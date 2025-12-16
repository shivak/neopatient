import logging
import os
import random
import string
from typing import Union, List
import jinja2
from openai import AsyncOpenAI

from .models import Patient, VerificationResponse, State, CohortSpec, Cohort, Stage
from .batch_llm import BatchLLM

# Get the directory of the current file to construct template path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_verify_template_path = os.path.join(_project_root, "templates", "verify.jinja2")

VERIFICATION_TEMPLATE = jinja2.Template(
    open(_verify_template_path, "r", encoding="utf-8").read()
)


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


async def verify_patient(
    record: Patient,
    positive: str,
    negative: str,
    client: AsyncOpenAI,
    verifier: str,
) -> VerificationResponse:
    """
    Verifies a patient record against positive and negative descriptions.

    Args:
        record: The patient record to verify
        positive: Positive cohort description
        negative: Negative cohort description
        client: AsyncOpenAI client instance
        verifier: Model name for verification

    Returns:
        VerificationResponse
    """
    logger = logging.getLogger(__name__)

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
    if ver_content is None:
        raise ValueError("No content in verification response")
    logger.info(f"Verification response: {ver_content}")
    verification = VerificationResponse.model_validate_json(ver_content)

    return verification


def _generate_salt() -> str:
    """Generate a random salt string for batch request IDs."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=8))


async def _start_verification_stage(
    batch_llm: BatchLLM,
    state: State,
    cohort_specs: list[CohortSpec],
    verifier: str,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Start verification stage using batch API."""
    # Prepare verification requests
    prompts_by_id = {}

    for spec, cohort, cohort_ids in zip(
        cohort_specs, state.coded_cohorts, state.sampled_ids
    ):
        for record, pid in zip(cohort, cohort_ids):
            prompt = create_verification_prompt(record, spec.positive, spec.negative)
            salt = _generate_salt()
            custom_id = f"verify_patient_{pid}_{salt}"
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
    cohort_specs: list[CohortSpec],
    verifier: str,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Check if verification batch is ready."""
    if not state.verification_batch_id:
        raise ValueError("No verification batch ID found in state")

    batch_id = state.verification_batch_id
    results = await batch_llm.get(batch_id)
    if results is None:
        return state
    state.verifications = _parse_verification_results(results, logger)

    state.stage = Stage.FINALIZE
    return state


def _parse_verification_results(
    results: dict[str, str], logger: logging.Logger
) -> dict[int, VerificationResponse]:
    """Parse verification results and organize by patient."""
    verifications = {}

    for custom_id, content in results.items():
        pid = int(custom_id.split("_")[2])
        verification = VerificationResponse.model_validate_json(content)
        verifications[pid] = verification

    return verifications
