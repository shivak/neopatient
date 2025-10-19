from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Tuple, Dict


def find_best_matching_code(
    coding_system: str,
    description: str,
    chroma_client: chromadb.PersistentClient,
    model: SentenceTransformer,
) -> tuple[str | None, str | None]:
    """
    Find the best matching medical code and description for a given coding system and description.

    Args:
        coding_system (str): The medical coding system (e.g., 'snomed', 'rxnorm', 'icd10_proc')
        description (str): The description to match against
        chroma_client (chromadb.PersistentClient): The ChromaDB client

    Returns:
        tuple[str, str]: The best matching medical code and its associated description (first 256 chars)
    """

    try:
        collection = chroma_client.get_collection(coding_system)
    except:
        raise ValueError(f"No collection found for coding system: {coding_system}")

    query_desc = description[:256]
    query_emb = model.encode([query_desc], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_emb], n_results=1, include=["metadatas", "documents"]
    )

    if results["metadatas"] and results["metadatas"][0]:
        metadata = results["metadatas"][0][0]
        return metadata["med_code"], metadata["desc"]
    return None, None


def find_best_matching_codes(
    coding_system: str,
    descriptions: list[str],
    chroma_client: chromadb.PersistentClient,
    model: SentenceTransformer,
) -> list[tuple[str, str]]:
    """
    Find the best matching medical codes and descriptions for multiple descriptions in a single batch operation.

    Args:
        coding_system (str): The medical coding system (e.g., 'snomed', 'rxnorm', 'icd10_proc')
        descriptions (List[str]): List of descriptions to match against
        chroma_client (chromadb.PersistentClient): The ChromaDB client

    Returns:
        List[Tuple[str, str]]: List of tuples containing (code, description) for each input description.
                              Returns (None, None) for descriptions that couldn't be matched.
    """
    if not descriptions:
        return []

    try:
        collection = chroma_client.get_collection(coding_system)
    except:
        raise ValueError(f"No collection found for coding system: {coding_system}")

    # Prepare all query descriptions (truncate to 256 chars)
    query_descs = [desc[:256] for desc in descriptions]

    # Encode all descriptions in a single batch
    query_embs = model.encode(query_descs, normalize_embeddings=True).tolist()

    # Perform batch search
    results = collection.query(
        query_embeddings=query_embs, n_results=1, include=["metadatas", "documents"]
    )

    # Process results
    matched_results = []
    for i in range(len(descriptions)):
        if (
            results["metadatas"]
            and results["metadatas"][i]
            and len(results["metadatas"][i]) > 0
        ):
            metadata = results["metadatas"][i][0]
            matched_results.append((metadata["med_code"], metadata["desc"]))
        else:
            matched_results.append((None, None))

    return matched_results


def batch_find_best_matching_codes(
    queries: List[Tuple[str, str]],
    chroma_client: chromadb.PersistentClient,
    model: SentenceTransformer,
) -> List[Tuple[str | None, str | None]]:
    """
    Find the best matching medical codes for multiple (coding_system, description) pairs.
    Groups queries by coding system for optimal batch processing.

    Args:
        queries (List[Tuple[str, str]]): List of (coding_system, description) tuples
        chroma_client (chromadb.PersistentClient): The ChromaDB client

    Returns:
        List[Tuple[str, str]]: List of (code, description) tuples in the same order as input queries
    """
    if not queries:
        return []

    # Group queries by coding system
    system_groups: Dict[str, List[Tuple[int, str]]] = {}
    for i, (system, desc) in enumerate(queries):
        if system not in system_groups:
            system_groups[system] = []
        system_groups[system].append((i, desc))

    # Initialize results list
    results = [(None, None) for _ in range(len(queries))]

    # Process each coding system in batch
    for system, system_queries in system_groups.items():
        indices, descriptions = zip(*system_queries)
        batch_results = find_best_matching_codes(
            system, list(descriptions), chroma_client, model
        )

        # Map results back to original positions
        for idx, result in zip(indices, batch_results):
            results[idx] = result

    return results
