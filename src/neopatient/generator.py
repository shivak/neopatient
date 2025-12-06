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
    CohortSpec,
    CodeSystem,
    RecordType,
)
from .sampler import sample_recipes


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
    record_type: str, recipes: Dict[int, PatientRecipe]
) -> List[Tuple[int, str]]:
    """Create generation prompts from PatientRecipe objects, one per segment.

    Returns list of (patient_id, prompt) tuples.
    """
    # Convert string to enum
    record_type_enum = RecordType(record_type)
    allowed_code_systems = CodeSystem.allowed_in(record_type_enum)

    prompts = []
    for patient_id, recipe in recipes.items():
        for segment in recipe.segments:
            # Format dates for segment
            start_iso = segment.start_date.isoformat()
            end_iso = segment.end_date.isoformat()
            if record_type in ["claims", "ehr-outpatient"]:
                formatted_start = start_iso.split("T")[0]
                formatted_end = end_iso.split("T")[0]
            else:
                formatted_start = start_iso
                formatted_end = end_iso

            prompt = GENERATION_TEMPLATE.render(
                record_type=record_type,
                start_date=formatted_start,
                end_date=formatted_end,
                recipe=recipe,
                segment=segment,
                allowed_code_systems=[cs.value for cs in allowed_code_systems],
            )
            prompts.append((patient_id, prompt))

    return prompts


async def synthesize_patient(
    positive: str,
    negative: str,
    patient_id: int,
    chroma_db: Union[ClientAPI, pathlib.Path, None],
    embedder_model: str,
    embedder_batch_size: int,
    embedder_args: dict,
    seed: int | None = None,
    generator: str = "gpt-5",
    verifier: str = "gpt-5",
    record_type: str = "ehr-outpatient",
    sampler: str = "gpt-5",
) -> Patient:
    """
    Generates a synthetic longitudinal patient record for an individual patient.

    First, samples an individualized patient description using the cohort-level positive and negative descriptions.
    Then, generates events based on the sampled description.
    Then, matches codes using ChromaDB.
    Finally, verifies the record satisfies the cohort-level positive and negative descriptions.

    Args:
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
        record_type: Type of record ("claims", "ehr-inpatient", "ehr-outpatient") (default: "ehr-outpatient")
        sampler: Model name for sampling individualized descriptions (default: "gpt-5")

    Returns:
        Generated patient record as MEDS DataSchema table

    Raises:
        ValueError: If the generated record does not satisfy the cohort criteria
    """
    logger = logging.getLogger(__name__)
    chroma_client = resolve_chroma_client(chroma_db)
    client = AsyncOpenAI()  # Assume API key is set via environment
    embedder = create_embedder(embedder_model, embedder_batch_size, embedder_args)

    print(f"Generating record using: {generator}")

    sampled = await sample_recipes(positive, negative, 1, record_type, sampler, logger)
    recipe = next(iter(sampled.values()))

    # Step 1: Generate tuples with LLM for each segment
    prompts = create_generation_prompts(record_type, {patient_id: recipe})
    segment_prompts = {}
    for pid, prompt in prompts:
        segment_prompts.setdefault(pid, []).append(prompt)

    combined_records = {}
    for pid, prompt_list in segment_prompts.items():
        for seg_idx, prompt in enumerate(prompt_list):
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
                #        seed=seed,
                temperature=0.7,
            )
            content = response.choices[0].message.content
            logger.info(f"Generation response for segment {seg_idx}: {content}")
            flat_generation_response = GenerationResponse.model_validate_json(content)
            segment_records = flat_generation_response.records.unflatten()
            # Combine records, assuming no overlapping times
            combined_records.update(segment_records.root)
    record_data = UncodedPatient(combined_records)

    # Step 2: Match codes and create Patient
    record = await code_patient(
        record_data, recipe, patient_id, chroma_client, embedder
    )

    # Step 3: Verify the record satisfies cohort-level positive and negative descriptions (cohort-level, but description includes negatives)
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
        #        seed=seed,
        temperature=0.0,
    )
    ver_content = ver_response.choices[0].message.content
    logger.info(f"Verification response: {ver_content}")
    verification = VerificationResponse.model_validate_json(ver_content)

    # Step 4: Check satisfaction
    if not verification.satisfactory:
        raise ValueError(f"Record does not satisfy criteria: {verification.criticism}")

    return record


async def synthesize_cohort(
    cohort_specs: List[CohortSpec],
    chroma_db: Union[ClientAPI, pathlib.Path, None],
    embedder_model: str,
    embedder_batch_size: int,
    embedder_args: dict,
    epsilon: float = 0.2,
    state: State | None = None,
    generator: str = "gpt-5",
    verifier: str = "gpt-5-nano",
    sampler: str = "gpt-5",
) -> Union[List[Cohort], State]:
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
    client = AsyncOpenAI()
    embedder = create_embedder(embedder_model, embedder_batch_size, embedder_args)

    # If resuming from state, use existing state
    if state is not None:
        current_state = state.copy()
    else:
        # Initialize new state
        current_state = {
            "stage": "sampling",
            "sampled_descriptions": [],
            "generation_tickets": [],
            "generated_records": [],
            "verification_tickets": [],
            "verifications": [],
        }

    # Stage 1: Sample individual patients
    if current_state["stage"] == "sampling":
        return await _handle_sampling_stage(
            client, current_state, cohort_specs, sampler
        )

    # Stage 2: Generate records using batch API
    elif current_state["stage"] == "generation":
        return await _handle_generation_stage(
            client, current_state, cohort_specs, generator
        )

    # Stage 2: Check generation results and apply code matching
    elif current_state["stage"] == "check_generation":
        return await _handle_check_generation_stage(
            client, current_state, cohort_specs, generator
        )

    # Stage 3: Start verification with code-matched records
    elif current_state["stage"] == "matching":
        return await _handle_matching_stage(
            client, current_state, chroma_db, embedder, cohort_specs, verifier
        )

    # Stage 4: Check verification results
    elif current_state["stage"] == "check_verification":
        return await _handle_check_verification_stage(
            client, current_state, cohort_specs, verifier
        )

    # Stage 5: Process final results
    elif current_state["stage"] == "finalize":
        return _handle_finalize_stage(current_state, cohort_specs)

    else:
        raise ValueError(f"Unknown stage: {current_state['stage']}")


async def synthesize_cohort_with_state_file(
    cohort_specs: List[CohortSpec],
    chroma_db: Union[ClientAPI, pathlib.Path, None],
    embedder_model: str,
    embedder_batch_size: int,
    embedder_args: dict,
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
            state = json.load(f)

    while True:
        result = await synthesize_cohort(
            cohort_specs=cohort_specs,
            chroma_db=chroma_db,
            embedder_model=embedder_model,
            embedder_batch_size=embedder_batch_size,
            embedder_args=embedder_args,
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
                    json.dump(result, f)
            time.sleep(poll_interval)
            state = result


async def _handle_sampling_stage(
    client: AsyncOpenAI, state: State, cohort_specs: List[CohortSpec], sampler: str
) -> Union[List[Cohort], State]:
    """Sample individual patient recipes for each cohort using the sampler LLM."""
    if state["sampled_descriptions"]:
        # Already sampled, move to generation
        state["stage"] = "generation"
        return _handle_generation_stage(client, state, cohort_specs, sampler)

    # Sample for each cohort
    for spec in cohort_specs:
        record_type = spec.record_type.value
        sampled = await sample_recipes(
            spec.positive,
            spec.negative,
            spec.count,
            record_type,
            sampler,
        )
        state["sampled_descriptions"].append(sampled)

    state["stage"] = "generation"
    return _handle_generation_stage(client, state, cohort_specs, sampler)


async def _handle_generation_stage(
    client: AsyncOpenAI, state: State, cohort_specs: List[CohortSpec], generator: str
) -> Union[List[Cohort], State]:
    """Handle the initial generation stage using batch API."""
    if state["generation_tickets"]:
        # Already submitted generation requests, move to checking
        state["stage"] = "check_generation"
        return _handle_check_generation_stage(client, state, cohort_specs, generator)

    # Prepare batch requests
    batch_requests = []

    for cohort_idx, sampled in enumerate(state["sampled_descriptions"]):
        spec = cohort_specs[cohort_idx]
        recipes = list(sampled.values())
        patient_ids = list(sampled.keys())
        prompts = create_generation_prompts(
            spec.record_type.value, dict(zip(patient_ids, recipes))
        )
        segment_prompts = {}
        for pid, prompt in prompts:
            segment_prompts.setdefault(pid, []).append(prompt)

        for pid, prompt_list in segment_prompts.items():
            for seg_idx, prompt in enumerate(prompt_list):
                salt = _generate_salt()

                batch_requests.append(
                    {
                        "custom_id": f"cohort_{cohort_idx}_patient_{pid}_segment_{seg_idx}_{salt}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": generator,
                            "messages": [{"role": "user", "content": prompt}],
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "generation_response",
                                    "strict": True,
                                    "schema": GenerationResponse.model_json_schema(),
                                },
                            },
                            "temperature": 0.7,
                        },
                    }
                )

    # Submit batch request
    try:
        batch_response = await client.batches.create(
            input_file_id=await _create_jsonl_file(batch_requests),
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        state["generation_tickets"].append(batch_response.id)
        state["stage"] = "check_generation"
        return state
    except Exception as e:
        raise RuntimeError(f"Failed to submit batch generation request: {e}")


async def _handle_check_generation_stage(
    client: AsyncOpenAI,
    state: State,
    cohort_specs: List[CohortSpec],
    chroma_db,
    generator: str,
    embedder,
    verifier: str,
) -> Union[List[Cohort], State]:
    """Check if generation batch is ready and start verification if so."""
    if not state["generation_tickets"]:
        raise ValueError("No generation tickets found in state")

    batch_id = state["generation_tickets"][0]

    try:
        batch_status = client.batches.retrieve(batch_id)

        if batch_status.status == "completed":
            # Download results
            batch_output = client.batches.retrieve(batch_id).output_file_id
            results = _download_batch_results(client, batch_output)

            # Parse generation results
            state["generated_records"] = _parse_generation_results(
                results, state["sampled_descriptions"]
            )

            # Move to matching stage
            state["stage"] = "matching"
            return _handle_matching_stage(
                client, state, chroma_db, embedder, cohort_specs, verifier
            )

        elif batch_status.status in ["failed", "expired", "cancelled"]:
            raise RuntimeError(
                f"Batch generation failed with status: {batch_status.status}"
            )

        else:
            # Still processing, return state to resume later
            return state

    except Exception as e:
        raise RuntimeError(f"Failed to check generation batch status: {e}")


async def _handle_matching_stage(
    client: AsyncOpenAI, state: State, chroma_db, embedder, cohort_specs, verifier
) -> Union[List[Cohort], State]:
    """Handle code matching stage and start verification."""
    chroma_client = resolve_chroma_client(chroma_db)

    # Apply code matching to all generated records
    state["coded_cohorts"] = []
    for cohort_idx, cohort_dict in enumerate(state["generated_records"]):
        recipes = state["sampled_descriptions"][cohort_idx]
        matched = await code_cohort(cohort_dict, recipes, chroma_client, embedder)
        state["coded_cohorts"].append(matched)

    # Start verification stage
    return _start_verification_stage(client, state, cohort_specs, verifier)


async def _start_verification_stage(
    client: AsyncOpenAI, state: State, cohort_specs: List[CohortSpec], verifier: str
) -> Union[List[Cohort], State]:
    """Start verification stage using batch API."""
    # Prepare verification requests
    batch_requests = []

    for cohort_idx, cohort_records in enumerate(state["coded_cohorts"]):
        spec = cohort_specs[cohort_idx]
        positive = spec.positive
        negative = spec.negative

        for record_idx, record in enumerate(cohort_records):
            prompt = create_verification_prompt(record, positive, negative)
            salt = _generate_salt()

            batch_requests.append(
                {
                    "custom_id": f"verify_cohort_{cohort_idx}_patient_{record_idx}_{salt}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": verifier,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "verification_response",
                                "strict": True,
                                "schema": VerificationResponse.model_json_schema(),
                            },
                        },
                    },
                }
            )

    # Submit verification batch
    try:
        batch_response = client.batches.create(
            input_file_id=_create_jsonl_file(batch_requests),
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        state["verification_tickets"].append(batch_response.id)
        state["stage"] = "check_verification"
        return state
    except Exception as e:
        raise RuntimeError(f"Failed to submit batch verification request: {e}")


async def _handle_check_verification_stage(
    client: AsyncOpenAI, state: State, cohort_specs: List[CohortSpec], verifier: str
) -> Union[List[Cohort], State]:
    """Check if verification batch is ready."""
    if not state["verification_tickets"]:
        raise ValueError("No verification tickets found in state")

    batch_id = state["verification_tickets"][0]

    try:
        batch_status = client.batches.retrieve(batch_id)

        if batch_status.status == "completed":
            # Download results
            batch_output = client.batches.retrieve(batch_id).output_file_id
            results = _download_batch_results(client, batch_output)

            # Parse verification results
            state["verifications"] = _parse_verification_results(results)

            # Move to finalization
            state["stage"] = "finalize"
            return _handle_finalize_stage(state, cohort_specs)

        elif batch_status.status in ["failed", "expired", "cancelled"]:
            raise RuntimeError(
                f"Batch verification failed with status: {batch_status.status}"
            )

        else:
            # Still processing, return state to resume later
            return state

    except Exception as e:
        raise RuntimeError(f"Failed to check verification batch status: {e}")


def _handle_finalize_stage(
    state: State, cohort_specs: List[CohortSpec]
) -> List[Cohort]:
    """Finalize results by filtering satisfactory records."""
    final_results = []

    for cohort_idx, (cohort_records, cohort_verifications) in enumerate(
        zip(state["coded_cohorts"], state["verifications"])
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


async def _create_jsonl_file(requests: List[Dict]) -> str:
    """Create a JSONL file from batch requests and upload to OpenAI."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for request in requests:
            f.write(json.dumps(request, default=str) + "\n")
        temp_path = f.name

    try:
        with open(temp_path, "rb") as f:
            file_response = await AsyncOpenAI().files.create(file=f, purpose="batch")
        return file_response.id
    finally:
        os.unlink(temp_path)


async def _download_batch_results(client: AsyncOpenAI, file_id: str) -> List[Dict]:
    """Download and parse batch results from OpenAI."""
    file_content = await client.files.content(file_id)
    results = []

    for line in file_content.text.split("\n"):
        if line.strip():
            results.append(json.loads(line))

    return results


def _parse_generation_results(
    results: List[Dict], sampled_descriptions: List[Dict[int, PatientRecipe]]
) -> List[Dict[int, UncodedPatient]]:
    """Parse generation results and organize by cohort, combining segments."""
    cohort_records = [{} for _ in sampled_descriptions]
    segment_data = {}  # (cohort_idx, patient_id) -> list of (seg_idx, records)

    for result in results:
        if result.get("response", {}).get("status_code") == 200:
            custom_id = result["custom_id"]
            parts = custom_id.split("_")
            cohort_idx = int(parts[1])
            patient_id = int(parts[3])
            seg_idx = int(parts[5])  # segment_{seg_idx}
            flat_generation_response = GenerationResponse.model_validate_json(
                result["response"]["body"]["choices"][0]["message"]["content"]
            )
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
    results: List[Dict],
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
            verification = VerificationResponse.model_validate_json(
                result["response"]["body"]["choices"][0]["message"]["content"]
            )
            cohort_verifications[cohort_idx].append(verification)

    return cohort_verifications
