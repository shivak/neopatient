from chromadb.api import ClientAPI

from .models import (
    CodeSystem,
    UncodedPatient,
    Patient,
    PatientRecipe,
    State,
    Cohort,
    Stage,
)
from .embed import Embed
from .database import resolve_chroma_client
import pyarrow as pa
import datetime
import logging
from typing import Union, List


def _format_code(code_system: CodeSystem, code: str) -> str:
    return f"{code_system.value}//{code}"


def _convert_time(time_str: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(time_str)


def query_with_instructions(code_desc: str, code_system: CodeSystem) -> str:
    if code_system == CodeSystem.LOINC:
        instruct = "find the closest medical code description. Focus on the medical observation, measurement or order. Be cognizant of the fact that some medical terminology may be abbreviated or unfamiliar to you."
    else:
        instruct = "find the closest medical code description. Focus on the medical concept, diagnosis, process, severity, and/or other medical particulars. Be cognizant of the fact that some medical terminology may be unfamiliar to you."
    return f"Instruct: {instruct}\nQuery:{code_desc}"


async def match_codes_in_system(
    coding_system: CodeSystem,
    descriptions: list[str],
    chroma_client: ClientAPI,
    embedder: Embed,
) -> list[tuple[str, str]]:
    """
    Find the best matching medical codes and descriptions for multiple descriptions in a single batch operation.

    Args:
        coding_system (CodeSystem): The medical coding system
        descriptions (List[str]): List of descriptions to match against
        chroma_client (ClientAPI): The ChromaDB client

    Returns:
        List[Tuple[str, str]]: List of (code, description) tuples for each input description.
    """
    if not descriptions:
        return []

    collection = chroma_client.get_collection(coding_system.value)

    # Encode all descriptions using embedder
    matched_results = []

    instructed_descriptions = [
        query_with_instructions(desc, coding_system) for desc in descriptions
    ]
    async for embedding_batch in embedder(instructed_descriptions):
        results = collection.query(
            query_embeddings=embedding_batch, n_results=1, include=["documents"]
        )
        codes_and_descrs = [
            (ids[0], docs[0]) for ids, docs in zip(results["ids"], results["documents"])
        ]
        matched_results.extend(codes_and_descrs)

    return matched_results


async def match_codes(
    queries: list[tuple[CodeSystem, str]],
    chroma_client: ClientAPI,
    embedder: Embed,
) -> list[tuple[CodeSystem, str, str]]:
    """
    Find the best matching medical codes and descriptions for multiple (coding_system, description) pairs.
    Groups queries by coding system for optimal batch processing.

    Args:
        queries (List[Tuple[str, str]]): List of (coding_system, description) tuples
        chroma_client (chromadb.PersistentClient): The ChromaDB client

    Returns:
        List[Tuple[CodeSystem, str, str]]: List of (code_system, code, description) tuples in the same order as input queries
    """
    if not queries:
        return []

    # Group queries by coding system
    system_groups: dict[CodeSystem, list[tuple[int, str]]] = {}
    for i, (system, desc) in enumerate(queries):
        if system not in system_groups:
            system_groups[system] = []
        system_groups[system].append((i, desc))

    # Collect results with indices
    results_with_indices = []
    for system, system_queries in system_groups.items():
        indices, descriptions = zip(*system_queries)
        batch_results = await match_codes_in_system(
            system, list(descriptions), chroma_client, embedder
        )

        # Collect results with their original indices
        for idx, (code, descr) in zip(indices, batch_results):
            results_with_indices.append((idx, (system, code, descr)))

    # Sort by original index and extract results
    results_with_indices.sort(key=lambda x: x[0])
    results = [result for _, result in results_with_indices]

    return results


async def code_patient(
    patient: UncodedPatient,
    recipe: PatientRecipe,
    patient_id: int,
    chroma_client: ClientAPI,
    embedder: Embed,
) -> Patient:
    """Match code descriptions for a single patient and return MEDS table."""

    # Collect static queries
    static_queries = []
    if recipe.gender:
        static_queries.append((CodeSystem("snomed"), f"Gender ({recipe.gender})"))
    if recipe.race:
        static_queries.append((CodeSystem("snomed"), f"Race ({recipe.race})"))
    if recipe.ethnicity:
        static_queries.append((CodeSystem("snomed"), f"Ethnicity ({recipe.ethnicity})"))

    # Collect longitudinal queries
    longitudinal_queries = [
        (row.code_system, row.code_desc)
        for time_str, events in patient.root.items()
        for row in events
    ]

    # Combine queries
    all_queries = static_queries + longitudinal_queries

    # Perform batch matching
    batch_results = await match_codes(all_queries, chroma_client, embedder)

    # Split results
    static_results = batch_results[: len(static_queries)]
    longitudinal_results = batch_results[len(static_queries) :]

    # Build rows
    rows = []

    # MEDS_BIRTH
    rows.append(
        {
            "subject_id": patient_id,
            "time": recipe.birthday,
            "code": "MEDS_BIRTH",
            "numeric_value": None,
            "unit": None,
            "code_descr": None,
        }
    )

    # Static events
    for (query_cs, code_desc), (code_system, matched_code, matched_descr) in zip(
        static_queries, static_results
    ):
        rows.append(
            {
                "subject_id": patient_id,
                "time": None,
                "code": _format_code(code_system, matched_code),
                "numeric_value": None,
                "unit": None,
                "code_descr": matched_descr[:128],
            }
        )

    # Longitudinal events
    idx = 0
    for time_str, events in patient.root.items():
        for row in events:
            code_system, matched_code, matched_descr = longitudinal_results[idx]
            rows.append(
                {
                    "subject_id": patient_id,
                    "time": _convert_time(time_str),
                    "code": _format_code(code_system, matched_code),
                    "numeric_value": row.numeric_value,
                    "unit": row.unit,
                    "code_descr": matched_descr[:128],
                }
            )
            idx += 1

    # Don't validate against PatientSchema because code_descr gets dropped
    table = pa.Table.from_pylist(rows)
    return table


async def code_cohort(
    patients: dict[int, UncodedPatient],
    recipes: dict[int, PatientRecipe],
    chroma_client: ClientAPI,
    embedder: Embed,
) -> dict[int, Patient]:
    """Match code descriptions for a cohort of patients and return dict of patient records."""

    cohort = {}
    for patient_id, patient in patients.items():
        recipe = recipes[patient_id]
        table = await code_patient(patient, recipe, patient_id, chroma_client, embedder)
        cohort[patient_id] = table
    return cohort


async def _handle_matching_stage(
    batch_llm,
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
    matched = await code_cohort(
        state.generated_records, state.sampled_recipes, chroma_client, embedder
    )
    state.coded_cohorts = [
        [matched[pid] for pid in cohort_ids] for cohort_ids in state.sampled_ids
    ]

    # Start verification stage
    state.stage = Stage.VERIFICATION
    return state
