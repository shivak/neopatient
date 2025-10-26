import json
import os
from typing import Dict, List, Union, Any
import openai
import jinja2
import chromadb
from chromadb.api import ClientAPI
import pandas as pd
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from .matcher import batch_find_best_matching_codes
from .models import PatientRecord, GenerationRecord
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


def generate_synthetic_patient_record(
    positive: str,
    negative: str,
    chroma_client: chromadb.ClientAPI,
    seed: int | None = None,
    generator: str = "gpt-4o",
    verifier: str = "gpt-4o",
) -> pa.Table:
    """
    Generates a synthetic longitudinal patient record based on positive and negative descriptions.

    Args:
        positive: Positive cohort description
        negative: Negative anti-cohort description
        chroma_client: The ChromaDB client
        seed: Optional seed for reproducibility
        generator: Model name for generation (default: "gpt-4o")
        verifier: Model name for verification (default: "gpt-4o")

    Returns:
        Generated patient record JSON

    Raises:
        ValueError: If the generated record does not satisfy the criteria
    """
    client = openai.OpenAI()  # Assume API key is set via environment

    print(f"Generating record using: {generator}")

    # Step 1: Generate tuples with LLM using structured output
    prompt = GENERATION_TEMPLATE.render(positive_cohort=positive)
    response = client.beta.chat.completions.parse(
        model=generator,
        messages=[{"role": "user", "content": prompt}],
        response_format=GenerationRecord,
        #        seed=seed,
        temperature=0.7,
    )
    record = response.choices[0].message.parsed

    # Step 2: Match codes and create DataSchema
    record = _match_codes([record], chroma_client)

    # Step 3: Verify with LLM
    record_tsv = record.to_pandas().to_csv(sep="\t", index=False)
    ver_prompt = VERIFICATION_TEMPLATE.render(
        record_tsv=record_tsv, positive=positive, negative=negative
    )
    ver_response = client.chat.completions.create(
        model=verifier,
        messages=[{"role": "user", "content": ver_prompt}],
        response_format={"type": "json_object"},
        #        seed=seed,
        temperature=0.0,
    )
    verification = json.loads(ver_response.choices[0].message.content)

    # Step 4: Check satisfaction
    if not verification["satisfactory"]:
        raise ValueError(
            f"Record does not satisfy criteria: {verification['criticism']}"
        )

    return record


def generate_synthetic_patient_records_batch(
    cohort_specs: List[Dict[str, Any]],
    chroma_client: chromadb.PersistentClient,
    epsilon: float = 0.2,
    state: Dict[str, Any] | None = None,
    generator: str = "gpt-5",
    verifier: str = "gpt-5-nano",
) -> Union[List[List[Dict]], Dict[str, Any]]:
    """
    Generates synthetic patient records in batch using OpenAI's batch API.

    Args:
        cohort_specs: List of cohort specifications, each containing:
            - count: Number of patients to generate for this cohort
            - positive: Positive cohort description
            - negative: Negative (anti-cohort) description
            - seeds: Optional list of seeds (length should match count)
        chroma_client: The ChromaDB client
        epsilon: Over-generation factor (1 + epsilon) to account for failed verifications
        state: Optional state to resume from a previous batch operation
        generator: Model name for generation (default: "gpt-4o")
        verifier: Model name for verification (default: "gpt-4o")

    Returns:
        Either:
        - List of lists of generated patient records (one list per cohort)
        - State dictionary for resuming if batch is not ready yet
    """
    client = openai.OpenAI()

    # If resuming from state, use existing state
    if state is not None:
        current_state = state.copy()
    else:
        # Initialize new state
        current_state = {
            "stage": "generation",
            "cohort_specs": cohort_specs,
            "chroma_client": chroma_client,
            "epsilon": epsilon,
            "generator": generator,
            "verifier": verifier,
            "generation_tickets": [],
            "generated_records": [],
            "verification_tickets": [],
            "verified_records": [],
            "completed_cohorts": [],
        }

    # Stage 1: Generate records using batch API
    if current_state["stage"] == "generation":
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


def _handle_generation_stage(
    client: openai.OpenAI, state: Dict[str, Any]
) -> Union[List[List[Dict]], Dict[str, Any]]:
    """Handle the initial generation stage using batch API."""
    if state["generation_tickets"]:
        # Already submitted generation requests, move to checking
        state["stage"] = "check_generation"
        return _handle_check_generation_stage(client, state)

    # Prepare batch requests
    batch_requests = []
    request_id = 0

    for cohort_idx, spec in enumerate(state["cohort_specs"]):
        count = spec["count"]
        positive = spec["positive"]
        seeds = spec.get("seeds", [None] * count)

        # Over-generate by epsilon factor
        target_count = int(count * (1 + state["epsilon"]))

        for i in range(target_count):
            seed = seeds[i] if i < len(seeds) else None
            prompt = GENERATION_TEMPLATE.render(positive_cohort=positive)

            batch_requests.append(
                {
                    "custom_id": f"cohort_{cohort_idx}_patient_{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": state.get("generator", state.get("generation_model")),
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        #                    "seed": seed,
                        "temperature": 1.0,
                    },
                }
            )
            request_id += 1

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
    client: openai.OpenAI, state: Dict[str, Any]
) -> Union[List[List[Dict]], Dict[str, Any]]:
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
                results, state["cohort_specs"]
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
    client: openai.OpenAI, state: Dict[str, Any]
) -> Union[List[List[Dict]], Dict[str, Any]]:
    """Handle code matching stage and start verification."""
    # Apply code matching to all generated records using true batch processing
    state["code_matched_records"] = _match_codes(
        state["generated_records"], state["chroma_client"]
    )

    # Start verification stage
    return _start_verification_stage(client, state)


def _start_verification_stage(
    client: openai.OpenAI, state: Dict[str, Any]
) -> Union[List[List[Dict]], Dict[str, Any]]:
    """Start verification stage using batch API."""
    # Prepare verification requests
    batch_requests = []

    for cohort_idx, cohort_records in enumerate(state["code_matched_records"]):
        spec = state["cohort_specs"][cohort_idx]
        positive = spec["positive"]
        negative = spec["negative"]

        for record_idx, record in enumerate(cohort_records):
            record_str = json.dumps(record, indent=2, default=str)
            prompt = VERIFICATION_TEMPLATE.render(
                record_json=record_str, positive=positive, negative=negative
            )

            batch_requests.append(
                {
                    "custom_id": f"verify_cohort_{cohort_idx}_patient_{record_idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": state.get("verifier", state.get("verification_model")),
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
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
    client: openai.OpenAI, state: Dict[str, Any]
) -> Union[List[List[Dict]], Dict[str, Any]]:
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
            state["verified_records"] = _parse_verification_results(
                results, state["code_matched_records"]
            )

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


def _handle_finalize_stage(state: Dict[str, Any]) -> List[List[Dict]]:
    """Finalize results by filtering satisfactory records."""
    final_results = []

    for cohort_idx, (cohort_records, cohort_verifications) in enumerate(
        zip(state["code_matched_records"], state["verified_records"])
    ):
        spec = state["cohort_specs"][cohort_idx]
        target_count = spec["count"]
        satisfactory_records = []

        for record, verification in zip(cohort_records, cohort_verifications):
            if verification["satisfactory"]:
                # Records already have code matching applied
                satisfactory_records.append(record)

                if len(satisfactory_records) >= target_count:
                    break

        final_results.append(satisfactory_records)

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
    results: List[Dict], cohort_specs: List[Dict]
) -> List[List[Dict]]:
    """Parse generation results and organize by cohort."""
    cohort_records = [[] for _ in cohort_specs]

    for result in results:
        if result.get("response", {}).get("status_code") == 200:
            custom_id = result["custom_id"]
            cohort_idx = int(custom_id.split("_")[1])
            record_data = json.loads(
                result["response"]["body"]["choices"][0]["message"]["content"]
            )

            # Validate the record against the Pydantic model
            try:
                validated_record = PatientRecord.model_validate(record_data)
                cohort_records[cohort_idx].append(validated_record.model_dump())
            except Exception as e:
                print(f"Warning: Failed to validate record for {custom_id}: {e}")
                # Skip invalid records
                continue

    return cohort_records


def _parse_verification_results(
    results: List[Dict], generated_records: List[List[Dict]]
) -> List[List[Dict]]:
    """Parse verification results and organize by cohort."""
    cohort_verifications = [[] for _ in generated_records]

    for result in results:
        if result.get("response", {}).get("status_code") == 200:
            custom_id = result["custom_id"]
            cohort_idx = int(custom_id.split("_")[2])
            verification = json.loads(
                result["response"]["body"]["choices"][0]["message"]["content"]
            )
            cohort_verifications[cohort_idx].append(verification)

    return cohort_verifications


def _match_codes(
    cohort_records: List[GenerationRecord], chroma_client: ClientAPI
) -> pa.Table:
    """Apply code matching to cohort records and return DataSchema."""
    if not cohort_records:
        empty_df = pd.DataFrame(
            columns=["subject_id", "time", "code", "numeric_value", "text_value"]
        )
        return pa.Table.from_pandas(empty_df, schema=DataSchema.schema)

    # Collect all descriptions to match
    all_descriptions = []
    for record in cohort_records:
        for row in record:
            all_descriptions.append(row[2])  # code_desc

    # Assume default system 'lnc' for matching
    systems = ["lnc"] * len(all_descriptions)
    queries = list(zip(systems, all_descriptions))

    # Perform batch matching
    model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")
    batch_results = batch_find_best_matching_codes(queries, chroma_client, model)

    # Build rows for DataSchema
    rows = []
    idx = 0
    for record in cohort_records:
        for row in record:
            subject_id, time, code_desc, numeric_value, text_value = row
            matched_code, matched_desc = batch_results[idx]
            code = matched_code if matched_code else code_desc  # fallback to desc
            text_value = matched_desc[:128] if matched_desc else text_value
            rows.append(
                {
                    "subject_id": subject_id,
                    "time": time,
                    "code": code,
                    "numeric_value": numeric_value,
                    "text_value": text_value,
                }
            )
            idx += 1

    df = pd.DataFrame(rows)
    return pa.Table.from_pandas(df, schema=DataSchema.schema)
