from chromadb.api import ClientAPI
from typing import List, Tuple, Dict
from .models import CodeSystem, UncodedPatient, Cohort
from .embed import Embed
from meds.schema import DataSchema as PatientSchema
import pyarrow as pa
import datetime


def _format_code(code_system: CodeSystem, code: str) -> str:
    return f"{code_system.value}//{code}"


def _convert_time(time_str: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(time_str)


async def match_codes_in_system(
    coding_system: CodeSystem,
    descriptions: list[str],
    chroma_client: ClientAPI,
    embedder: Embed,
) -> list[tuple[CodeSystem, str, str]]:
    """
    Find the best matching medical codes and descriptions for multiple descriptions in a single batch operation.

    Args:
        coding_system (CodeSystem): The medical coding system
        descriptions (List[str]): List of descriptions to match against
        chroma_client (ClientAPI): The ChromaDB client

    Returns:
        List[Tuple[CodeSystem, str, str]]: List of tuples containing (code_system, code, description) for each input description.
    """
    if not descriptions:
        return []

    try:
        collection = chroma_client.get_collection(coding_system.value)
    except Exception:
        raise ValueError(f"No collection found for coding system: {coding_system}")

    # Encode all descriptions using embedder
    query_embs = []
    async for batch in embedder(descriptions):
        query_embs.extend(batch)

    if not query_embs:
        return []

    # Perform batch search
    results = collection.query(
        query_embeddings=query_embs, n_results=1, include=["documents"]
    )

    # Process results
    matched_results = []
    for i in range(len(descriptions)):
        code = results["ids"][i][0]
        document = results["documents"][i][0]
        matched_results.append((coding_system, code, document))

    return matched_results


async def match_codes(
    queries: List[Tuple[CodeSystem, str]],
    chroma_client: ClientAPI,
    embedder: Embed,
) -> List[Tuple[CodeSystem, str, str]]:
    """
    Find the best matching medical codes for multiple (coding_system, description) pairs.
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
    system_groups: Dict[CodeSystem, List[Tuple[int, str]]] = {}
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
        for idx, result in zip(indices, batch_results):
            results_with_indices.append((idx, result))

    # Sort by original index and extract results
    results_with_indices.sort(key=lambda x: x[0])
    results = [result for _, result in results_with_indices]

    return results


async def code_patient(
    patients: Dict[int, UncodedPatient],
    chroma_client: ClientAPI,
    embedder: Embed,
) -> Cohort:
    """Match code descriptions to standardized codes using ChromaDB and return list of patient records."""

    if not patients:
        return []

    # Collect all descriptions to match
    queries = [
        (row.code_system, row.code_desc)
        for record in patients.values()
        for time_str, events in record.root.items()
        for row in events
    ]

    # Perform batch matching
    batch_results = await match_codes(queries, chroma_client, embedder)

    # Build tables per patient
    cohort = []
    idx = 0
    for patient_id, record in patients.items():
        rows = []
        first_entry = True
        for time_str, events in record.root.items():
            if first_entry:
                # Handle birthday: Add MEDS_BIRTH row with time_str, and static events with time=None
                rows.append(
                    {
                        "subject_id": patient_id,
                        "time": _convert_time(time_str),  # Birthday time for MEDS_BIRTH
                        "code": "MEDS_BIRTH",
                        "numeric_value": None,
                        "unit": None,
                    }
                )
                for row in events:
                    code_system, code, matched_desc = batch_results[idx]
                    rows.append(
                        {
                            "subject_id": patient_id,
                            "time": None,  # Untimestamped for static
                            "code": _format_code(code_system, code),
                            "numeric_value": row.numeric_value,
                            "unit": row.unit,
                        }
                    )
                    idx += 1
                first_entry = False
            else:
                # Regular events with time_str
                for row in events:
                    code_system, code, matched_desc = batch_results[idx]
                    rows.append(
                        {
                            "subject_id": patient_id,
                            "time": _convert_time(time_str),
                            "code": _format_code(code_system, code),
                            "numeric_value": row.numeric_value,
                            "unit": row.unit,
                        }
                    )
                    idx += 1
        table = pa.Table.from_pylist(rows, schema=PatientSchema.schema())
        cohort.append(table)

    return cohort
