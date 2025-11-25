import json
import os
import random
import string
import sys
import time
from typing import Dict, List, Union, Any
import pathlib
import datetime
import openai
import jinja2
import chromadb
from chromadb.api import ClientAPI
import pandas as pd
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from .matcher import batch_find_best_matching_codes
from .models import UncodedPatient, VerificationResponse, State, Patient, Cohort, PatientRecipe, GenerationResponse, CohortSpec
from .sampler import sample_individual_descriptions
from meds.schema import DataSchema


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


def _get_csv_path(record_type: str) -> str:
    if record_type == "ehr-inpatient":
        return os.path.join(_project_root, "stats", "ehr-inpatient.csv")
    else:
        return os.path.join(_project_root, "stats", "ehr-outpatient.csv")


def sample_patient_stats(csv_path: str, n: int) -> List[Dict[str, Any]]:
    """Sample n rows with replacement from CSV, returning list of dicts."""
    df = pd.read_csv(csv_path)
    sampled_df = df.sample(n=n, replace=True)
    return sampled_df.to_dict('records')


def _resolve_chroma_client(chroma_db: Union[chromadb.ClientAPI, pathlib.Path, None]) -> chromadb.ClientAPI:
    """Resolve chroma_db parameter to a ChromaDB client."""
    from .database import load_chroma_client
    
    if chroma_db is None:
        # Download pre-generated ChromaDB files from Hugging Face
        chroma_path = snapshot_download("cab-harvard/neopatient")
        return load_chroma_client(chroma_path)
    elif isinstance(chroma_db, pathlib.Path):
        return load_chroma_client(str(chroma_db))
    elif isinstance(chroma_db, chromadb.ClientAPI):
        return chroma_db
    else:
        raise ValueError("chroma_db must be a ChromaDB client, a pathlib.Path, or None")


def synthesize_patient(
    positive: str,
    negative: str,
    patient_id: int,
    chroma_db: Union[chromadb.ClientAPI, pathlib.Path, None],
    seed: int | None = None,
    generator: str = "gpt-5",
    verifier: str = "gpt-5",
    record_type: str = "ehr-outpatient",
    end_date: str | None = None,
) -> Patient:
    """
    Generates a synthetic longitudinal patient record for an individual patient.

    First, generates events based on the positive cohort description.
    Then, matches codes using ChromaDB.
    Finally, verifies the record satisfies the cohort-level positive and negative descriptions.

    Args:
        positive: Positive cohort description used for generation and verification
        negative: Negative cohort description for verification
        patient_id: Unique patient identifier
        chroma_db: The ChromaDB client, path, or None for code matching
        seed: Optional seed for reproducibility
        generator: Model name for generation (default: "gpt-5")
        verifier: Model name for verification (default: "gpt-5")
        record_type: Type of record ("claims", "ehr-inpatient", "ehr-outpatient") (default: "ehr-outpatient")
        end_date: End date for the record (ISO string), defaults to current time

    Returns:
        Generated patient record as MEDS DataSchema table

    Raises:
        ValueError: If the generated record does not satisfy the cohort criteria
    """
    chroma_client = _resolve_chroma_client(chroma_db)
    client = openai.OpenAI()  # Assume API key is set via environment

    print(f"Generating record using: {generator}")

    # Sample stats
    csv_path = _get_csv_path(record_type)
    stat = sample_patient_stats(csv_path, 1)[0]
    duration = stat['duration']
    avg_codes_per_time = stat['avg_codes_per_time']
    if end_date is None:
        end_date = datetime.datetime.now().isoformat()
    days = int(duration.split()[0])
    start_date = (datetime.datetime.fromisoformat(end_date) - datetime.timedelta(days=days)).isoformat()
    if record_type in ["claims", "ehr-outpatient"]:
        start_date = start_date.split('T')[0]  # Truncate to YYYY-MM-DD
        end_date = end_date.split('T')[0]      # Truncate to YYYY-MM-DD
    individual_description = positive

    # Step 1: Generate tuples with LLM
    prompt = GENERATION_TEMPLATE.render(individual_description=individual_description, record_type=record_type, start_date=start_date, end_date=end_date, avg_codes_per_time=avg_codes_per_time)
    response = client.chat.completions.create(
        model=generator,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": GenerationResponse.model_json_schema()},
        #        seed=seed,
        temperature=0.7,
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM response content is None")
    generation_response = GenerationResponse.model_validate_json(content)
    if not generation_response.finished:
        raise ValueError("Generation not finished, discarding incomplete record")
    record_data = generation_response.records
    # Step 2: Match codes and create DataSchema
    record = _match_codes([record_data], [patient_id], chroma_client)[0]

    # Step 3: Verify the record satisfies cohort-level positive and negative descriptions (cohort-level, but description includes negatives)
    record_tsv = record.to_pandas().to_csv(sep="\t", index=False)
    ver_prompt = VERIFICATION_TEMPLATE.render(
        record_tsv=record_tsv, positive=positive, negative=negative
    )
    ver_response = client.chat.completions.create(
        model=verifier,
        messages=[{"role": "user", "content": ver_prompt}],
        response_format={"type": "json_schema", "json_schema": VerificationResponse.model_json_schema()},
        #        seed=seed,
        temperature=0.0,
    )
    verification = VerificationResponse.model_validate_json(ver_response.choices[0].message.content)

    # Step 4: Check satisfaction
    if not verification.satisfactory:
        raise ValueError(
            f"Record does not satisfy criteria: {verification.criticism}"
        )

    return record


def synthesize_cohort(
    cohort_specs: List[CohortSpec],
    chroma_db: Union[chromadb.ClientAPI, pathlib.Path, None],
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
    chroma_db = _resolve_chroma_client(chroma_db)
    client = openai.OpenAI()

    # If resuming from state, use existing state
    if state is not None:
        current_state = state.copy()
        chroma_db = current_state["chroma_db"]
    else:
        # Initialize new state
        current_state = {
            "stage": "sampling",
            "cohort_specs": cohort_specs,
            "chroma_db": chroma_db,
            "epsilon": epsilon,
            "generator": generator,
            "verifier": verifier,
            "sampler": sampler,
            "sampled_descriptions": [],
            "generation_tickets": [],
            "generated_records": [],
            "verification_tickets": [],
            "verifications": [],
        }

    # Stage 1: Sample individual patients
    if current_state["stage"] == "sampling":
        return _handle_sampling_stage(client, current_state)

    # Stage 2: Generate records using batch API
    elif current_state["stage"] == "generation":
        return _handle_generation_stage(client, current_state)

    # Stage 2: Check generation results and apply code matching
    elif current_state["stage"] == "check_generation":
        return _handle_check_generation_stage(client, current_state)

    # Stage 3: Start verification with code-matched records
    elif current_state["stage"] == "matching":
        return _handle_matching_stage(client, current_state)

    # Stage 4: Check verification results
    elif current_state["stage"] == "check_verification":
        return _handle_check_verification_stage(client, current_state)

    # Stage 5: Process final results
    elif current_state["stage"] == "finalize":
        return _handle_finalize_stage(current_state)

    else:
        raise ValueError(f"Unknown stage: {current_state['stage']}")


def synthesize_cohort_with_state_file(
    cohort_specs: List[CohortSpec],
    chroma_db: Union[chromadb.ClientAPI, pathlib.Path, None],
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
        with open(state_file, 'r') as f:
            state = json.load(f)
    while True:
        result = synthesize_cohort(
            cohort_specs=cohort_specs,
            chroma_db=chroma_db,
            generator=generator,
            verifier=verifier,
            sampler=sampler,
            state=state
        )
        if isinstance(result, list):
            if state_file and pathlib.Path(state_file).exists():
                os.unlink(state_file)
            return result
        else:
            if state_file:
                with open(state_file, 'w') as f:
                    json.dump(result, f)
            time.sleep(poll_interval)
            state = result


def _handle_sampling_stage(
    client: openai.OpenAI, state: State
) -> Union[List[Cohort], State]:
    """Sample individual patient recipes for each cohort using the sampler LLM."""
    if state["sampled_descriptions"]:
        # Already sampled, move to generation
        state["stage"] = "generation"
        return _handle_generation_stage(client, state)

    # Sample for each cohort
    for spec in state["cohort_specs"]:
        record_type = spec.get("record_type", "ehr-outpatient")
        csv_path = _get_csv_path(record_type)
        stats = sample_patient_stats(csv_path, spec["count"])
        duration = stats[0]['duration']  # assuming same duration for cohort
        sampled = sample_individual_descriptions(
            spec["positive"], spec["negative"], spec["count"], duration, state["sampler"]
        )
        state["sampled_descriptions"].append(sampled)

    state["stage"] = "generation"
    return _handle_generation_stage(client, state)


def _handle_generation_stage(
    client: openai.OpenAI, state: State
) -> Union[List[Cohort], State]:
    """Handle the initial generation stage using batch API."""
    if state["generation_tickets"]:
        # Already submitted generation requests, move to checking
        state["stage"] = "check_generation"
        return _handle_check_generation_stage(client, state)

    # Prepare batch requests
    batch_requests = []

    for cohort_idx, sampled in enumerate(state["sampled_descriptions"]):
        spec = state["cohort_specs"][cohort_idx]
        record_type = spec.get("record_type", "ehr-outpatient")
        csv_path = _get_csv_path(record_type)
        stats = sample_patient_stats(csv_path, spec["count"])
        for (patient_id, pr), stat in zip(sampled.items(), stats):
            avg_codes_per_time = stat['avg_codes_per_time']
            individual_description = pr.description
            start_date_str = pr.start_date.isoformat()
            end_date_str = pr.end_date.isoformat()
            if record_type in ["claims", "ehr-outpatient"]:
                start_date_str = start_date_str.split('T')[0]  # Truncate to YYYY-MM-DD
                end_date_str = end_date_str.split('T')[0]      # Truncate to YYYY-MM-DD
            prompt = GENERATION_TEMPLATE.render(individual_description=individual_description, record_type=record_type, start_date=start_date_str, end_date=end_date_str, avg_codes_per_time=avg_codes_per_time)
            salt = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

            batch_requests.append(
                {
                    "custom_id": f"cohort_{cohort_idx}_patient_{patient_id}_{salt}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                      "body": {
                          "model": state.get("generator"),
                          "messages": [{"role": "user", "content": prompt}],
                          "response_format": {"type": "json_schema", "json_schema": GenerationResponse.model_json_schema()},
                         "temperature": 1.0,
                     },
                }
            )

    # Submit batch request
    try:
        batch_response = client.batches.create(
            input_file_id=_create_jsonl_file(batch_requests),
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        state["generation_tickets"].append(batch_response.id)
        state["stage"] = "check_generation"
        return state
    except Exception as e:
        raise RuntimeError(f"Failed to submit batch generation request: {e}")


def _handle_check_generation_stage(
    client: openai.OpenAI, state: State
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
            state["generated_records"], state["patient_ids"] = _parse_generation_results(
                results, state["sampled_descriptions"]
            )

            # Move to matching stage
            state["stage"] = "matching"
            return _handle_matching_stage(client, state)

        elif batch_status.status in ["failed", "expired", "cancelled"]:
            raise RuntimeError(
                f"Batch generation failed with status: {batch_status.status}"
            )

        else:
            # Still processing, return state to resume later
            return state

    except Exception as e:
        raise RuntimeError(f"Failed to check generation batch status: {e}")


def _handle_matching_stage(
    client: openai.OpenAI, state: State
) -> Union[List[Cohort], State]:
    """Handle code matching stage and start verification."""
    chroma_client = _resolve_chroma_client(state["chroma_db"])
    
    # Apply code matching to all generated records
    state["coded_cohorts"] = []
    for cohort_records, cohort_patient_ids in zip(state["generated_records"], state["patient_ids"]):
        matched = _match_codes(cohort_records, cohort_patient_ids, chroma_client)
        state["coded_cohorts"].append(matched)

    # Start verification stage
    return _start_verification_stage(client, state)


def _start_verification_stage(
    client: openai.OpenAI, state: State
) -> Union[List[Cohort], State]:
    """Start verification stage using batch API."""
    # Prepare verification requests
    batch_requests = []

    for cohort_idx, cohort_records in enumerate(state["coded_cohorts"]):
        spec = state["cohort_specs"][cohort_idx]
        positive = spec["positive"]
        negative = spec["negative"]

        for record_idx, record in enumerate(cohort_records):
            record_tsv = record.to_pandas().to_csv(sep="\t", index=False)
            prompt = VERIFICATION_TEMPLATE.render(
                record_tsv=record_tsv, positive=positive, negative=negative
            )
            salt = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

            batch_requests.append(
                {
                    "custom_id": f"verify_cohort_{cohort_idx}_patient_{record_idx}_{salt}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                     "body": {
                         "model": state.get("verifier"),
                         "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_schema", "json_schema": VerificationResponse.model_json_schema()},
                        "temperature": 0.0,
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


def _handle_check_verification_stage(
    client: openai.OpenAI, state: State
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
            return _handle_finalize_stage(state)

        elif batch_status.status in ["failed", "expired", "cancelled"]:
            raise RuntimeError(
                f"Batch verification failed with status: {batch_status.status}"
            )

        else:
            # Still processing, return state to resume later
            return state

    except Exception as e:
        raise RuntimeError(f"Failed to check verification batch status: {e}")


def _handle_finalize_stage(state: State) -> List[Cohort]:
    """Finalize results by filtering satisfactory records."""
    final_results = []

    for cohort_idx, (cohort_records, cohort_verifications) in enumerate(
        zip(state["coded_cohorts"], state["verifications"])
    ):
        spec = state["cohort_specs"][cohort_idx]
        target_count = spec["count"]
        satisfactory_records = [
            record for record, verification in zip(cohort_records, cohort_verifications)
            if verification.satisfactory
        ]
        capped_records = satisfactory_records[:target_count]
        if capped_records:
            final_results.append(capped_records)
        if len(satisfactory_records) < target_count:
            print(f"Warning: Cohort {cohort_idx} has only {len(satisfactory_records)} satisfactory records, expected {target_count}", file=sys.stderr)

    return final_results


def _create_jsonl_file(requests: List[Dict]) -> str:
    """Create a JSONL file from batch requests and upload to OpenAI."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for request in requests:
            f.write(json.dumps(request, default=str) + "\n")
        temp_path = f.name

    try:
        with open(temp_path, "rb") as f:
            file_response = openai.OpenAI().files.create(file=f, purpose="batch")
        return file_response.id
    finally:
        os.unlink(temp_path)


def _download_batch_results(client: openai.OpenAI, file_id: str) -> List[Dict]:
    """Download and parse batch results from OpenAI."""
    file_content = client.files.content(file_id)
    results = []

    for line in file_content.text.split("\n"):
        if line.strip():
            results.append(json.loads(line))

    return results


def _parse_generation_results(
    results: List[Dict], sampled_descriptions: List[Dict[int, PatientRecipe]]
) -> tuple[List[List[UncodedPatient]], List[List[int]]]:
    """Parse generation results and organize by cohort."""
    cohort_records = [[] for _ in sampled_descriptions]
    patient_ids = [[] for _ in sampled_descriptions]

    for result in results:
        if result.get("response", {}).get("status_code") == 200:
            custom_id = result["custom_id"]
            cohort_idx = int(custom_id.split("_")[1])
            patient_id = int(custom_id.split("_")[3])
            generation_response = GenerationResponse.model_validate_json(
                result["response"]["body"]["choices"][0]["message"]["content"]
            )
            if generation_response.finished:
                cohort_records[cohort_idx].append(generation_response.records)
                patient_ids[cohort_idx].append(patient_id)

    return cohort_records, patient_ids


def _parse_verification_results(results: List[Dict]) -> List[List[VerificationResponse]]:
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


def _match_codes(
    cohort_records: List[UncodedPatient], patient_ids: List[int], chroma_client: ClientAPI
) -> Cohort:
    """Match code descriptions to standardized codes using ChromaDB and return list of patient records."""
    if not cohort_records:
        return []

    # Collect all descriptions to match
    all_descriptions = []
    systems = []
    for record in cohort_records:
        for time_str, events in record.root.items():
            for row in events:
                systems.append(row.code_system)
                all_descriptions.append(row.code_desc)

    queries = list(zip(systems, all_descriptions))

    # Perform batch matching
    model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")
    batch_results = batch_find_best_matching_codes(queries, chroma_client, model)

    # Build tables per patient
    cohort = []
    idx = 0
    for record, patient_id in zip(cohort_records, patient_ids):
        rows = []
        first_entry = True
        for time_str, events in record.root.items():
            if first_entry:
                # Handle birthday: Add MEDS_BIRTH row with time_str, and static events with time=None
                rows.append({
                    "subject_id": patient_id,
                    "time": time_str,  # Birthday time for MEDS_BIRTH
                    "code": "MEDS_BIRTH",
                    "numeric_value": None,
                    "unit": None,
                    "text_value": None,
                })
                for row in events:
                    code, matched_desc = batch_results[idx]
                    rows.append({
                        "subject_id": patient_id,
                        "time": None,  # Untimestamped for static
                        "code": code,
                        "numeric_value": row.numeric_value,
                        "unit": row.unit,
                        "text_value": row.text_value,
                    })
                    idx += 1
                first_entry = False
            else:
                # Regular events with time_str
                for row in events:
                    code, matched_desc = batch_results[idx]
                    rows.append({
                        "subject_id": patient_id,
                        "time": time_str,
                        "code": code,
                        "numeric_value": row.numeric_value,
                        "unit": row.unit,
                        "text_value": row.text_value,
                    })
                    idx += 1
        table = pa.Table.from_pylist(rows, schema=DataSchema.schema)
        cohort.append(table)

    return cohort
