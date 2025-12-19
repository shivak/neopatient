import json
import logging

import pathlib

from openai import AsyncOpenAI

from chromadb.api import ClientAPI
from .matcher import code_patient
from .database import resolve_chroma_client
from .embed import Embed, create_embedder
from .models import (
    State,
    Patient,
    Cohort,
    CohortSpec,
    RecordType,
    Stage,
)
from .batch_llm import BatchLLM, create_batch_llm
from .generator import (
    generate_patient,
    _handle_generation_stage,
    _handle_check_generation_stage,
)
from .verifier import verify_patient
from .sampler import (
    sample_recipes,
    _handle_sampling_stage,
    _handle_check_sampling_stage,
)
from .matcher import _handle_matching_stage
from .verifier import (
    _start_verification_stage,
    _handle_check_verification_stage,
)


logger = logging.getLogger(__name__)


def _synthesis_setup(
    chroma_db: ClientAPI | pathlib.Path | None,
    embedder_model: str | None,
    embedder_batch_size: int | None,
    embedder_args: dict | None,
    embedder_base_url: str | None,
    embedder_api_key: str | None,
    sampler: str,
    generator: str,
    verifier: str,
    poll_interval: int | None,
) -> tuple[ClientAPI, Embed, BatchLLM, BatchLLM, BatchLLM]:
    """Set up shared dependencies for synthesis functions."""
    chroma_db = resolve_chroma_client(chroma_db)
    embedder = create_embedder(
        embedder_model,
        embedder_batch_size,
        embedder_args,
        embedder_base_url,
        embedder_api_key,
    )
    sampler_llm = create_batch_llm(sampler, poll_interval)
    generator_llm = create_batch_llm(generator, poll_interval)
    verifier_llm = create_batch_llm(verifier, poll_interval)
    return chroma_db, embedder, sampler_llm, generator_llm, verifier_llm


async def synthesize_patient(
    client: AsyncOpenAI,
    positive: str,
    negative: str,
    patient_id: int,
    chroma_db: ClientAPI | pathlib.Path | None,
    embedder_model: str | None,
    embedder_batch_size: int | None,
    embedder_args: dict | None,
    embedder_base_url: str | None = None,
    embedder_api_key: str | None = None,
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
    chroma_client = resolve_chroma_client(chroma_db)
    embedder = create_embedder(
        embedder_model,
        embedder_batch_size,
        embedder_args,
        embedder_base_url,
        embedder_api_key,
    )

    # Sample recipe
    sampled = await sample_recipes(client, positive, negative, 1, record_type, sampler)
    recipe = sampled[0]

    # Generate uncoded patient
    uncoded = await generate_patient(client, recipe, patient_id, record_type, generator)

    # Match codes and create Patient
    record = await code_patient(uncoded, recipe, patient_id, chroma_client, embedder)

    # Verify the record
    verification = await verify_patient(record, positive, negative, client, verifier)

    # Check satisfaction
    if not verification.satisfactory:
        raise ValueError(f"Record does not satisfy criteria: {verification.criticism}")

    return record


async def _synthesize_cohorts(
    cohort_specs: list[CohortSpec],
    chroma_db,
    embedder,
    sampler: BatchLLM,
    generator: BatchLLM,
    verifier: BatchLLM,
    state: State | None = None,
) -> list[Cohort] | State:
    # Round up cohort sizes to at least 3 to avoid generating null cohorts
    # due to individual failure probabilities
    for spec in cohort_specs:
        spec.count = max(spec.count, 3)
    # If resuming from state, use existing state
    if state is not None:
        current_state = state
    else:
        # Initialize new state
        current_state = State(stage=Stage.SAMPLING)

    logger.info(f"Current stage: {current_state.stage}")
    match current_state.stage:
        case Stage.SAMPLING:
            return await _handle_sampling_stage(
                sampler,
                current_state,
                cohort_specs,
                chroma_db,
                embedder,
            )
        case Stage.CHECK_SAMPLING:
            return await _handle_check_sampling_stage(
                sampler,
                current_state,
                cohort_specs,
                chroma_db,
                embedder,
            )
        case Stage.GENERATION:
            return await _handle_generation_stage(
                generator,
                current_state,
                cohort_specs,
                chroma_db,
                embedder,
            )
        case Stage.CHECK_GENERATION:
            return await _handle_check_generation_stage(
                generator,
                current_state,
                cohort_specs,
                chroma_db,
                embedder,
            )
        case Stage.MATCHING:
            return await _handle_matching_stage(
                current_state,
                chroma_db,
                embedder,
                cohort_specs,
            )
        case Stage.VERIFICATION:
            return await _start_verification_stage(
                verifier, current_state, cohort_specs
            )
        case Stage.CHECK_VERIFICATION:
            return await _handle_check_verification_stage(
                verifier, current_state, cohort_specs
            )
        case Stage.FINALIZE:
            return _handle_finalize_stage(current_state, cohort_specs)
        case _:
            raise ValueError(f"Unknown stage: {current_state.stage}")


async def synthesize_cohorts(
    cohort_specs: list[CohortSpec],
    chroma_db: ClientAPI | pathlib.Path | None,
    embedder_model: str | None,
    embedder_batch_size: int | None,
    embedder_args: dict | None,
    embedder_base_url: str | None = None,
    embedder_api_key: str | None = None,
    state: State | None = None,
    generator: str = "gpt-5",
    verifier: str = "gpt-5-nano",
    sampler: str = "gpt-5",
    poll_interval: int | None = None,
) -> list[Cohort] | State:
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
        state: Optional state to resume from a previous batch operation
        generator: Model name for generation (default: "gpt-5")
        verifier: Model name for verification (default: "gpt-5-nano")
        sampler: Model name for sampling (default: "gpt-5")
        poll_interval: Optional poll interval for rate limiting batch checks

    Returns:
        Either:
        - List of cohorts, where each cohort is a list of patient records
        - State dictionary for resuming if batch is not ready yet
    """
    chroma_db, embedder, sampler_llm, generator_llm, verifier_llm = _synthesis_setup(
        chroma_db,
        embedder_model,
        embedder_batch_size,
        embedder_args,
        embedder_base_url,
        embedder_api_key,
        sampler,
        generator,
        verifier,
        poll_interval,
    )

    return await _synthesize_cohorts(
        cohort_specs,
        chroma_db,
        embedder,
        sampler_llm,
        generator_llm,
        verifier_llm,
        state,
    )


async def synthesize_cohorts_with_state_file(
    cohort_specs: list[CohortSpec],
    chroma_db: ClientAPI | pathlib.Path | None,
    embedder_model: str | None,
    embedder_batch_size: int | None,
    embedder_args: dict | None,
    state_file: pathlib.Path,
    embedder_base_url: str | None = None,
    embedder_api_key: str | None = None,
    generator: str = "gpt-5-nano",
    verifier: str = "gpt-5",
    sampler: str = "gpt-5",
    poll_interval: int = 15 * 60,
) -> list[Cohort]:
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
        state_file: pathlib.Path to state file for resuming
        poll_interval: Seconds to wait between polls

    Returns:
        List of cohorts (each cohort is list of Patient tables)
    """
    # Resolve dependencies once outside the loop
    chroma_db, embedder, sampler_llm, generator_llm, verifier_llm = _synthesis_setup(
        chroma_db,
        embedder_model,
        embedder_batch_size,
        embedder_args,
        embedder_base_url,
        embedder_api_key,
        sampler,
        generator,
        verifier,
        poll_interval,
    )

    state = None
    if state_file.exists():
        with open(state_file, "r") as f:
            state = State.model_validate(json.load(f))

    while True:
        result = await _synthesize_cohorts(
            cohort_specs=cohort_specs,
            chroma_db=chroma_db,
            embedder=embedder,
            sampler=sampler_llm,
            generator=generator_llm,
            verifier=verifier_llm,
            state=state,
        )
        if isinstance(result, list):
            return result
        else:
            with open(state_file, "w") as f:
                # Serialize to string first to avoid partial writes if serialization fails
                state_json = json.dumps(result.model_dump(), default=str)
                f.write(state_json)
            state = result


def _handle_finalize_stage(
    state: State, cohort_specs: list[CohortSpec]
) -> list[Cohort]:
    """Finalize results by filtering satisfactory records."""
    final_results = []

    for spec, cohort, cohort_ids in zip(
        cohort_specs, state.coded_cohorts, state.sampled_ids
    ):
        satisfactory_patients = [
            record
            for record, pid in zip(cohort, cohort_ids)
            if state.verifications[pid].satisfactory
        ]
        final_results.append(satisfactory_patients)

    return final_results
