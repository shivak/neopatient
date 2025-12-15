import json
import logging
import os
import random
import string
import sys
import time
from typing import Dict, List, Union, Tuple
import pathlib
from openai import AsyncOpenAI
import jinja2

from chromadb.api import ClientAPI
from .matcher import code_patient, code_cohort
from .database import resolve_chroma_client
from .embed import create_embedder
from .models import (
    UncodedPatient,
    VerificationResponse,
    State,
    Patient,
    Cohort,
    PatientRecipe,
    GenerationResponse,
    SamplingResponse,
    CohortSpec,
    CodeSystem,
    RecordType,
    Stage,
)
from .sampler import (
    sample_recipes,
    SAMPLE_TEMPLATE,
    sample_patient_stats,
    _get_csv_path,
)
from .batch_llm import create_batch_llm, BatchLLM


# Load Jinja2 template from file
def load_template(template_path: str) -> jinja2.Template:
    """Load a Jinja2 template from a file path."""
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()
    return jinja2.Template(template_content)


# Get the directory of the current file to construct template path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_generate_template_path = os.path.join(_project_root, "templates", "generate.jinja2")
_verify_template_path = os.path.join(_project_root, "templates", "verify.jinja2")

GENERATION_TEMPLATE = load_template(_generate_template_path)
VERIFICATION_TEMPLATE = load_template(_verify_template_path)


def create_verification_prompt(record: Patient, positive: str, negative: str) -> str:
    """Create a verification prompt for a patient record."""
    record_tsv = (
        record.to_pandas()
        .drop(columns=["subject_id", "text_value"], errors="ignore")
        .to_csv(sep="\t", index=False)
    )
    return VERIFICATION_TEMPLATE.render(
        record_tsv=record_tsv, positive=positive, negative=negative
    )


def _generate_salt() -> str:
    """Generate a random salt string for batch request IDs."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=8))


def create_generation_prompts(
    record_type: RecordType, recipes: Dict[int, PatientRecipe]
) -> List[Tuple[int, str]]:
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


async def synthesize_patient(
    client: AsyncOpenAI,
    positive: str,
    negative: str,
    patient_id: int,
    chroma_db: Union[ClientAPI, pathlib.Path, None],
    embedder_model: str | None,
    embedder_batch_size: int | None,
    embedder_args: dict | None,
    embedder_base_url: str | None = None,
    seed: int | None = None,
    generator: str = "gpt-5",
    verifier: str = "gpt-5",
    record_type: RecordType = RecordType.EHR_OUTPATIENT,
    sampler: str = "gpt-5",
) -> Patient:
    """
    Generates a synthetic longitudinal patient record for an individual patient.

    First, samples an individualized patient description using the cohort-level positive and negative descriptions.
    Then, generates events based on the sampled description.
    Then, matches codes using ChromaDB.
    Finally, verifies the record satisfies the cohort-level positive and negative descriptions.

    Args:
        client: AsyncOpenAI client instance
        positive: Positive cohort description used for sampling and verification
        negative: Negative cohort description for sampling and verification
        patient_id: Unique patient identifier
        chroma_db: The ChromaDB client, path, or None for code matching
        embedder_model: Embedder model name for code matching
        embedder_batch_size: Batch size for embedding operations
        embedder_args: Embedder args dict for code matching
        seed: Optional seed for reproducibility
        generator: Model name for generation (default: "gpt-5")
        verifier: Model name for verification (default: "gpt-5")
        record_type: Type of record (RecordType enum) (default: RecordType.EHR_OUTPATIENT)
        sampler: Model name for sampling individualized descriptions (default: "gpt-5")

    Returns:
        Generated patient record as MEDS DataSchema table

    Raises:
        ValueError: If the generated record does not satisfy the cohort criteria
    """
    logger = logging.getLogger(__name__)
    chroma_client = resolve_chroma_client(chroma_db)
    embedder = create_embedder(
        embedder_model, embedder_batch_size, embedder_args, embedder_base_url
    )

    # Sample recipe
    sampled = await sample_recipes(
        client, positive, negative, 1, record_type, sampler, logger
    )
    recipe = sampled[0]

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
        logger.info(f"Generation response for segment {seg_idx}: {content}")
        flat_generation_response = GenerationResponse.model_validate_json(content)
        segment_records = flat_generation_response.records.unflatten()
        # Combine records, assuming no overlapping times
        combined_records.update(segment_records.root)
    record_data = UncodedPatient(combined_records)

    # Match codes and create Patient
    record = await code_patient(
        record_data, recipe, patient_id, chroma_client, embedder
    )

    # Verify the record satisfies positive and negative descriptions
    ver_prompt = create_verification_prompt(record, positive, negative)
    logger.info(f"Verification prompt: {ver_prompt}")
    ver_response = await client.chat.completions.create(
        model=verifier,
        messages=[{"role": "user", "content": ver_prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "verification_response",
                "strict": True,
                "schema": VerificationResponse.model_json_schema(),
            },
        },
        temperature=0.0,
    )
    ver_content = ver_response.choices[0].message.content
    logger.info(f"Verification response: {ver_content}")
    verification = VerificationResponse.model_validate_json(ver_content)

    # Step 4: Check satisfaction
    if not verification.satisfactory:
        raise ValueError(f"Record does not satisfy criteria: {verification.criticism}")

    return record


async def synthesize_cohorts(
    cohort_specs: List[CohortSpec],
    chroma_db: Union[ClientAPI, pathlib.Path, None],
    embedder_model: str | None,
    embedder_batch_size: int | None,
    embedder_args: dict | None,
    embedder_base_url: str | None = None,
    epsilon: float = 0.2,
    state: State | None = None,
    generator: str = "gpt-5",
    verifier: str = "gpt-5-nano",
    sampler: str = "gpt-5",
) -> Union[List[Cohort], State]:
    logger = logging.getLogger(__name__)

    # Create batch LLM instance
    batch_llm = create_batch_llm(generator)

    """
    Generates synthetic patient records in batch using OpenAI's batch API.

    Process:
    1. Sample individual patient descriptions for each cohort.
    2. Generate longitudinal records for each individual.
    3. Match codes using ChromaDB.
    4. Verify records satisfy cohort-level criteria.
    5. Return satisfactory records.

    Args:
        cohort_specs: List of cohort specifications, each containing:
            - count: Number of patients to generate for this cohort
            - positive: Positive cohort description
            - negative: Negative (anti-cohort) description
        chroma_db: The ChromaDB client, path, or None for code matching
        embedder_model: Embedder model name for code matching
        embedder_args: Embedder args dict for code matching
        epsilon: Over-generation factor (deprecated, now exact count from sampling)
        state: Optional state to resume from a previous batch operation
        generator: Model name for generation (default: "gpt-5")
        verifier: Model name for verification (default: "gpt-5-nano")
        sampler: Model name for sampling (default: "gpt-5")

    Returns:
        Either:
        - List of cohorts, where each cohort is a list of patient records
        - State dictionary for resuming if batch is not ready yet
    """
    chroma_db = resolve_chroma_client(chroma_db)
    embedder = create_embedder(
        embedder_model, embedder_batch_size, embedder_args, embedder_base_url
    )

    # If resuming from state, use existing state
    if state is not None:
        current_state = state
    else:
        # Initialize new state
        current_state = State(stage=Stage.SAMPLING)

    match current_state.stage:
        case Stage.SAMPLING:
            return await _handle_sampling_stage(
                batch_llm,
                current_state,
                cohort_specs,
                sampler,
                generator,
                chroma_db,
                embedder,
                verifier,
                logger,
            )
        case Stage.CHECK_SAMPLING:
            return await _handle_check_sampling_stage(
                batch_llm,
                current_state,
                cohort_specs,
                sampler,
                generator,
                chroma_db,
                embedder,
                verifier,
                logger,
            )
        case Stage.GENERATION:
            return await _handle_generation_stage(
                batch_llm,
                current_state,
                cohort_specs,
                generator,
                chroma_db,
                embedder,
                verifier,
                logger,
            )
        case Stage.CHECK_GENERATION:
            return await _handle_check_generation_stage(
                batch_llm,
                current_state,
                cohort_specs,
                chroma_db,
                generator,
                embedder,
                verifier,
                logger,
            )
        case Stage.MATCHING:
            return await _handle_matching_stage(
                batch_llm,
                current_state,
                chroma_db,
                embedder,
                cohort_specs,
                verifier,
                logger,
            )
        case Stage.VERIFICATION:
            return await _start_verification_stage(
                batch_llm, current_state, cohort_specs, verifier, logger
            )
        case Stage.CHECK_VERIFICATION:
            return await _handle_check_verification_stage(
                batch_llm, current_state, cohort_specs, verifier, logger
            )
        case Stage.FINALIZE:
            return _handle_finalize_stage(current_state, cohort_specs)
        case _:
            raise ValueError(f"Unknown stage: {current_state.stage}")


async def synthesize_cohorts_with_state_file(
    cohort_specs: List[CohortSpec],
    chroma_db: Union[ClientAPI, pathlib.Path, None],
    embedder_model: str | None,
    embedder_batch_size: int | None,
    embedder_args: dict | None,
    embedder_base_url: str | None = None,
    generator: str = "gpt-5-nano",
    verifier: str = "gpt-5",
    sampler: str = "gpt-5",
    state_file: Union[str, pathlib.Path, None] = None,
    poll_interval: int = 15 * 60,
) -> List[Cohort]:
    """
    Generates synthetic patient cohorts with state file management and polling.

    Args:
        cohort_specs: List of cohort specifications
        chroma_db: ChromaDB client, path, or None
        embedder_model: Embedder model name for code matching
        embedder_args: Embedder args dict for code matching
        generator: Model for generation
        verifier: Model for verification
        sampler: Model for sampling
        state_file: Path to state file for resuming
        poll_interval: Seconds to wait between polls

    Returns:
        List of cohorts (each cohort is list of Patient tables)
    """
    state = None
    if state_file and pathlib.Path(state_file).exists():
        with open(state_file, "r") as f:
            state = State.model_validate(json.load(f))

    while True:
        result = await synthesize_cohorts(
            cohort_specs=cohort_specs,
            chroma_db=chroma_db,
            embedder_model=embedder_model,
            embedder_batch_size=embedder_batch_size,
            embedder_args=embedder_args,
            embedder_base_url=embedder_base_url,
            generator=generator,
            verifier=verifier,
            sampler=sampler,
            state=state,
        )
        if isinstance(result, list):
            if state_file and pathlib.Path(state_file).exists():
                os.unlink(state_file)
            return result
        else:
            if state_file:
                with open(state_file, "w") as f:
                    json.dump(result.model_dump(), f)
            time.sleep(poll_interval)
            state = result


async def _handle_sampling_stage(
    batch_llm: BatchLLM,
    state: State,
    cohort_specs: List[CohortSpec],
    sampler: str,
    generator: str,
    chroma_db,
    embedder,
    verifier: str,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Sample individual patient recipes for each cohort using batch processing."""
    if state.sampled_recipes:
        # Already sampled, move to generation
        state.stage = Stage.GENERATION
    return state

    # Create batch sampling requests for all cohorts
    prompts_by_id = {}
    cohort_info = []  # Track (cohort_idx, expected_count) for validation

    for cohort_idx, spec in enumerate(cohort_specs):
        csv_path = _get_csv_path(spec.record_type)
        stats = sample_patient_stats(csv_path, spec.count)

        prompt = SAMPLE_TEMPLATE.render(
            positive_cohort=spec.positive,
            negative_cohort=spec.negative,
            n=spec.count,
            stats=stats,
        )

        custom_id = f"cohort_{cohort_idx}_sampling"
        prompts_by_id[custom_id] = prompt
        cohort_info.append((cohort_idx, spec.count))

        logger.info(f"Sampling prompt for cohort {cohort_idx}: {prompt[:100]}...")

    # Submit batch sampling request
    from .models import SamplingResponse

    schema = SamplingResponse.model_json_schema()
    batch_id = await batch_llm.ask(prompts_by_id, schema, sampler)
    state.sampling_batch_id = batch_id
    state.stage = Stage.CHECK_SAMPLING

    return state


async def _handle_check_sampling_stage(
    batch_llm: BatchLLM,
    state: State,
    cohort_specs: List[CohortSpec],
    sampler: str,
    generator: str,
    chroma_db,
    embedder,
    verifier: str,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Check if sampling batch is ready and parse results."""
    if not state.sampling_batch_id:
        raise ValueError("No sampling batch ID found in state")

    batch_id = state.sampling_batch_id
    is_done = await batch_llm.is_done(batch_id)
    if not is_done:
        return state
    results = await batch_llm.get(batch_id)

    cohort_results: dict[int, dict[int, PatientRecipe]] = {}
    for result in results:
        if result.get("response", {}).get("status_code") == 200:
            custom_id = result["custom_id"]
            # Extract cohort index from custom_id like "cohort_0_sampling"
            cohort_idx = int(custom_id.split("_")[1])

            content = result["response"]["body"]["choices"][0]["message"]["content"]
            logger.info(f"Sampling response for {custom_id}: {content[:100]}...")
            sampling_response = SamplingResponse.model_validate_json(content)

            # Validate we got the expected number of recipes
            expected_count = cohort_specs[cohort_idx].count
            if len(sampling_response.root) < expected_count:
                raise ValueError(
                    f"Cohort {cohort_idx}: Expected {expected_count} samples, got {len(sampling_response.root)}"
                )

            cohort_ids = random.sample(
                range(100000000, 1000000000), len(sampling_response.root)
            )
            cohort_results[cohort_idx] = dict(zip(cohort_ids, sampling_response.root))

    # Check that all cohorts have results
    if len(cohort_results) != len(cohort_specs):
        raise ValueError(
            f"Expected {len(cohort_specs)} cohort results, got {len(cohort_results)}"
        )

    state.sampled_recipes = [cohort_results[i] for i in range(len(cohort_specs))]
    state.stage = Stage.GENERATION
    return state


async def _handle_generation_stage(
    batch_llm: BatchLLM,
    state: State,
    cohort_specs: List[CohortSpec],
    generator: str,
    chroma_db,
    embedder,
    verifier: str,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Handle the initial generation stage using batch API."""
    # Prepare batch requests
    prompts_by_id = {}

    for cohort_idx, sampled in enumerate(state.sampled_recipes):
        spec = cohort_specs[cohort_idx]
        recipes = list(sampled.values())
        patient_ids = list(sampled.keys())
        prompts = create_generation_prompts(
            spec.record_type, dict(zip(patient_ids, recipes))
        )
        segment_prompts = {}
        for pid, prompt in prompts:
            segment_prompts.setdefault(pid, []).append(prompt)

        for pid, prompt_list in segment_prompts.items():
            for seg_idx, prompt in enumerate(prompt_list):
                salt = _generate_salt()
                custom_id = (
                    f"cohort_{cohort_idx}_patient_{pid}_segment_{seg_idx}_{salt}"
                )

                logger.info(f"Generation prompt for {custom_id}: {prompt}")
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
    cohort_specs: List[CohortSpec],
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

    is_done = await batch_llm.is_done(batch_id)
    if not is_done:
        return state

    results = await batch_llm.get(batch_id)
    state.generated_records = _parse_generation_results(
        results, state.sampled_recipes, logger
    )

    state.stage = Stage.MATCHING
    return state


async def _handle_matching_stage(
    batch_llm: BatchLLM,
    state: State,
    chroma_db,
    embedder,
    cohort_specs,
    verifier,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Handle code matching stage and start verification."""
    chroma_client = resolve_chroma_client(chroma_db)

    # Apply code matching to all generated records
    state.coded_cohorts = []
    for cohort_idx, cohort_dict in enumerate(state.generated_records):
        recipes = state.sampled_recipes[cohort_idx]
        matched = await code_cohort(cohort_dict, recipes, chroma_client, embedder)
        state.coded_cohorts.append(matched)

    # Start verification stage
    state.stage = Stage.VERIFICATION
    return state


async def _start_verification_stage(
    batch_llm: BatchLLM,
    state: State,
    cohort_specs: List[CohortSpec],
    verifier: str,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Start verification stage using batch API."""
    # Prepare verification requests
    prompts_by_id = {}

    for cohort_idx, cohort_records in enumerate(state.coded_cohorts):
        spec = cohort_specs[cohort_idx]
        positive = spec.positive
        negative = spec.negative

        for record_idx, record in enumerate(cohort_records):
            prompt = create_verification_prompt(record, positive, negative)
            salt = _generate_salt()
            custom_id = f"verify_cohort_{cohort_idx}_patient_{record_idx}_{salt}"
            logger.info(f"Verification prompt for {custom_id}: {prompt}")

            prompts_by_id[custom_id] = prompt

    # Submit verification batch
    try:
        schema = VerificationResponse.model_json_schema()
        batch_id = await batch_llm.ask(prompts_by_id, schema, verifier)
        state.verification_batch_id = batch_id
        state.stage = Stage.CHECK_VERIFICATION
        return state
    except Exception as e:
        raise RuntimeError(f"Failed to submit batch verification request: {e}")


async def _handle_check_verification_stage(
    batch_llm: BatchLLM,
    state: State,
    cohort_specs: List[CohortSpec],
    verifier: str,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Check if verification batch is ready."""
    if not state.verification_batch_id:
        raise ValueError("No verification batch ID found in state")

    batch_id = state.verification_batch_id
    is_done = await batch_llm.is_done(batch_id)
    if not is_done:
        return state

    results = await batch_llm.get(batch_id)
    state.verifications = _parse_verification_results(results, logger)

    state.stage = Stage.FINALIZE
    return state


def _handle_finalize_stage(
    state: State, cohort_specs: List[CohortSpec]
) -> List[Cohort]:
    """Finalize results by filtering satisfactory records."""
    final_results = []

    for cohort_idx, (cohort_records, cohort_verifications) in enumerate(
        zip(state.coded_cohorts, state.verifications)
    ):
        spec = cohort_specs[cohort_idx]
        target_count = spec.count
        satisfactory_records = [
            record
            for record, verification in zip(cohort_records, cohort_verifications)
            if verification.satisfactory
        ]
        capped_records = satisfactory_records[:target_count]
        if capped_records:
            final_results.append(capped_records)
        if len(satisfactory_records) < target_count:
            print(
                f"Warning: Cohort {cohort_idx} has only {len(satisfactory_records)} satisfactory records, expected {target_count}",
                file=sys.stderr,
            )

    return final_results


def _parse_generation_results(
    results: List[Dict],
    sampled_recipes: List[Dict[int, PatientRecipe]],
    logger: logging.Logger,
) -> List[Dict[int, UncodedPatient]]:
    """Parse generation results and organize by cohort, combining segments."""
    cohort_records = [{} for _ in sampled_recipes]
    segment_data = {}  # (cohort_idx, patient_id) -> list of (seg_idx, records)

    for result in results:
        if result.get("response", {}).get("status_code") == 200:
            custom_id = result["custom_id"]
            parts = custom_id.split("_")
            cohort_idx = int(parts[1])
            patient_id = int(parts[3])
            seg_idx = int(parts[5])  # segment_{seg_idx}
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            logger.info(f"Generation response for {custom_id}: {content}")
            flat_generation_response = GenerationResponse.model_validate_json(content)
            key = (cohort_idx, patient_id)
            if key not in segment_data:
                segment_data[key] = []
            segment_data[key].append(
                (seg_idx, flat_generation_response.records.unflatten())
            )

    # Combine segments for each patient
    for (cohort_idx, patient_id), segments in segment_data.items():
        combined_records = {}
        for seg_idx, records in sorted(segments, key=lambda x: x[0]):  # sort by seg_idx
            combined_records.update(records.root)
        cohort_records[cohort_idx][patient_id] = UncodedPatient(combined_records)

    return cohort_records


def _parse_verification_results(
    results: List[Dict], logger: logging.Logger
) -> List[List[VerificationResponse]]:
    """Parse verification results and organize by cohort."""
    # Determine number of cohorts from results
    cohort_indices = set()
    for result in results:
        if result.get("response", {}).get("status_code") == 200:
            custom_id = result["custom_id"]
            cohort_idx = int(custom_id.split("_")[2])
            cohort_indices.add(cohort_idx)

    max_cohort_idx = max(cohort_indices) if cohort_indices else 0
    cohort_verifications = [[] for _ in range(max_cohort_idx + 1)]

    for result in results:
        if result.get("response", {}).get("status_code") == 200:
            custom_id = result["custom_id"]
            cohort_idx = int(custom_id.split("_")[2])
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            logger.info(f"Verification response for {custom_id}: {content}")
            verification = VerificationResponse.model_validate_json(content)
            cohort_verifications[cohort_idx].append(verification)

    return cohort_verifications
