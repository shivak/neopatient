import duckdb
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils.batch_utils import create_batches

import os


def setup_databases(parquet_path: str, chroma_db_path: str = "clinprime_chroma"):
    """
    Initialize ChromaDB databases for each coding system from a parquet file.

    Args:
        parquet_path (str): Path to the clinprime_mapping.parquet file
        chroma_db_path (str): Path to the ChromaDB database directory

    Raises:
        FileNotFoundError: If the parquet file does not exist at the specified path
    """
    # Validate that the parquet file exists before attempting to connect
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at path: {parquet_path}")

    con = duckdb.connect()
    coding_systems = con.query(
        "SELECT DISTINCT code_system FROM read_parquet($parquet_path)",
        params={"parquet_path": parquet_path},
    ).fetchall()
    coding_systems = [row[0] for row in coding_systems]

    model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")

    # Initialize ChromaDB client with persistent storage
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=chroma_db_path, settings=settings)

    print(coding_systems)
    for system in coding_systems:
        # Delete existing collection if it exists
        try:
            client.delete_collection(system)
        except:
            pass  # Collection doesn't exist, which is fine

        # Create new collection
        collection = client.create_collection(
            name=system,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        sub_data = con.query(
            "SELECT med_code, t.desc FROM read_parquet($parquet_path) AS t WHERE code_system = $system",
            params={"parquet_path": parquet_path, "system": system},
        ).to_df()

        descs = sub_data["desc"].apply(lambda x: x[:256] if x else "").tolist()
        embeddings = model.encode(descs, normalize_embeddings=True)

        # Prepare data for ChromaDB
        ids = [f"{system}_{i}" for i in range(len(sub_data))]
        metadatas = [
            {"med_code": str(sub_data.loc[i, "med_code"]), "desc": descs[i]}
            for i in range(len(sub_data))
        ]

        # Add documents to collection in batches
        batches = create_batches(
            api=client,
            ids=ids,
            documents=descs,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        for batch in batches:
            print(f"Adding batch of size {len(batch[0])}")
            collection.add(
                ids=batch[0],
                embeddings=batch[1],
                metadatas=batch[2],
                documents=batch[3],
            )

    con.close()


def load_chroma_client(
    chroma_db_path: str = "clinprime_chroma",
) -> chromadb.PersistentClient:
    """
    Load a ChromaDB client for the specified database path.

    Args:
        chroma_db_path (str): Path to the ChromaDB database directory

    Returns:
        chromadb.PersistentClient: The ChromaDB client

    Raises:
        FileNotFoundError: If the ChromaDB database directory does not exist
    """
    if not os.path.exists(chroma_db_path):
        raise FileNotFoundError(
            f"ChromaDB database not found at path: {chroma_db_path}"
        )
    settings = Settings(anonymized_telemetry=False)
    return chromadb.PersistentClient(path=chroma_db_path, settings=settings)
