import logging
import os

from openai import AsyncOpenAI
import jinja2

from .models import (
    UncodedPatient,
    PatientRecipe,
    GenerationResponse,
    CodeSystem,
    RecordType,
    State,
    CohortSpec,
    Stage,
    Cohort,
)
from .batch_llm import BatchLLM
from typing import Union, List


# Get the directory of the current file to construct template path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_generate_template_path = os.path.join(_project_root, "templates", "generate.jinja2")

GENERATION_TEMPLATE = jinja2.Template(
    open(_generate_template_path, "r", encoding="utf-8").read()
)


def create_generation_prompts(
    record_type: RecordType, recipes: dict[int, PatientRecipe]
) -> list[tuple[int, str]]:
    """Create generation prompts from PatientRecipe objects, one per segment.

    Args:
        record_type: Type of record (RecordType enum)
        recipes: Dict of {patient_id: PatientRecipe}

    Returns list of (patient_id, prompt) tuples.
    """
    allowed_code_systems = CodeSystem.allowed_in(record_type)

    prompts = []
    for patient_id, recipe in recipes.items():
        for segment in recipe.segments:
            # Format dates for segment
            start_iso = segment.start_date.isoformat()
            end_iso = segment.end_date.isoformat()
            if record_type in [RecordType.CLAIMS, RecordType.EHR_OUTPATIENT]:
                formatted_start = start_iso.split("T")[0]
                formatted_end = end_iso.split("T")[0]
            else:
                formatted_start = start_iso
                formatted_end = end_iso

            prompt = GENERATION_TEMPLATE.render(
                record_type=record_type.value,
                start_date=formatted_start,
                end_date=formatted_end,
                recipe=recipe,
                segment=segment,
                allowed_code_systems=[cs.value for cs in allowed_code_systems],
                loinc_allowed=CodeSystem.LOINC in allowed_code_systems,
            )
            prompts.append((patient_id, prompt))

    return prompts


async def generate_patient(
    client: AsyncOpenAI,
    recipe: PatientRecipe,
    patient_id: int,
    record_type: RecordType,
    generator: str,
) -> UncodedPatient:
    """
    Generates an uncoded patient record from a given recipe.

    Args:
        client: AsyncOpenAI client instance
        recipe: The patient recipe
        patient_id: Unique patient identifier
        record_type: Type of record
        generator: Model name for generation

    Returns:
        UncodedPatient
    """
    logger = logging.getLogger(__name__)

    # Generate each segment and then concatenate into uncoded record
    prompts = create_generation_prompts(record_type, {patient_id: recipe})
    segment_prompts = [prompt for pid, prompt in prompts]

    combined_records = {}
    for seg_idx, prompt in enumerate(segment_prompts):
        logger.info(f"Generation prompt for segment {seg_idx}: {prompt}")
        response = await client.chat.completions.create(
            model=generator,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "generation_response",
                    "strict": True,
                    "schema": GenerationResponse.model_json_schema(),
                },
            },
            temperature=0.7,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No content in generation response")
        logger.info(f"Generation response for segment {seg_idx}: {content}")
        flat_generation_response = GenerationResponse.model_validate_json(content)
        segment_records = flat_generation_response.records.unflatten()
        # Combine records, assuming no overlapping times
        combined_records.update(segment_records.root)
    record_data = UncodedPatient(combined_records)

    return record_data


async def _handle_generation_stage(
    batch_llm: BatchLLM,
    state: State,
    cohort_specs: list[CohortSpec],
    generator: str,
    chroma_db,
    embedder,
    verifier: str,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Handle the initial generation stage using batch API."""
    # Prepare batch requests
    prompts_by_id = {}

    for cohort_idx, cohort_patient_ids in enumerate(state.sampled_ids):
        spec = cohort_specs[cohort_idx]
        recipes = {pid: state.sampled_recipes[pid] for pid in cohort_patient_ids}
        prompts = create_generation_prompts(spec.record_type, recipes)
        segment_prompts = {}
        for pid, prompt in prompts:
            segment_prompts.setdefault(pid, []).append(prompt)

        for pid, prompt_list in segment_prompts.items():
            for seg_idx, prompt in enumerate(prompt_list):
                custom_id = f"patient_{pid}_segment_{seg_idx}"
                prompts_by_id[custom_id] = prompt

    # Submit batch request
    try:
        schema = GenerationResponse.model_json_schema()
        batch_id = await batch_llm.ask(prompts_by_id, schema, generator)
        state.generation_batch_id = batch_id
        state.stage = Stage.CHECK_GENERATION
        return state
    except Exception as e:
        raise RuntimeError(f"Failed to submit batch generation request: {e}")


async def _handle_check_generation_stage(
    batch_llm: BatchLLM,
    state: State,
    cohort_specs: list[CohortSpec],
    chroma_db,
    generator: str,
    embedder,
    verifier: str,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Check if generation batch is ready and start verification if so."""
    if not state.generation_batch_id:
        raise ValueError("No generation batch ID found in state")

    batch_id = state.generation_batch_id

    results = await batch_llm.get(batch_id)
    if results is None:
        return state
    state.generated_records = _parse_generation_results(
        results, state.sampled_recipes, logger
    )

    state.stage = Stage.MATCHING
    return state


def _parse_generation_results(
    results: dict[str, str],
    sampled_recipes: dict[int, PatientRecipe],
    logger: logging.Logger,
) -> dict[int, UncodedPatient]:
    """Parse generation results and organize by patient, combining segments."""
    cohort_records = {}
    segment_data = {}  # patient_id -> list of (seg_idx, records)

    for custom_id, content in results.items():
        parts = custom_id.split("_")
        patient_id = int(parts[1])
        seg_idx = int(parts[3])  # segment_{seg_idx}
        flat_generation_response = GenerationResponse.model_validate_json(content)
        if patient_id not in segment_data:
            segment_data[patient_id] = []
        segment_data[patient_id].append(
            (seg_idx, flat_generation_response.records.unflatten())
        )

    # Combine segments for each patient
    for patient_id, segments in segment_data.items():
        combined_records = {}
        for seg_idx, records in sorted(segments, key=lambda x: x[0]):  # sort by seg_idx
            combined_records.update(records.root)
        cohort_records[patient_id] = UncodedPatient(combined_records)

    return cohort_records
