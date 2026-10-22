import duckdb
import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings


import pathlib
from huggingface_hub import snapshot_download
from .models import CodeSystem
from .embed import Embed

import os


async def create_database(parquet_path: str, embedder: Embed, db_dir: str):
    """
    Initialize ChromaDB databases for each coding system from a parquet file.

    Args:
        parquet_path (str): Path to the clinprime_mapping.parquet file
        db_dir (str): Path to the ChromaDB database directory
        embedder: Embedder function to use for creating embeddings

    Raises:
        FileNotFoundError: If the parquet file does not exist at the specified path
    """
    # Validate that the parquet file exists before attempting to connect
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at path: {parquet_path}")

    con = duckdb.connect()

    # Initialize ChromaDB client with persistent storage
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=db_dir, settings=settings)

    for system in CodeSystem:
        print(f"Creating collection for {system}")
        # Delete existing collection if it exists
        try:
            client.delete_collection(system.value)
        except Exception:
            pass  # Collection doesn't exist, which is fine

        # Create new collection
        collection = client.create_collection(
            name=system.value,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        results = con.query(
            "SELECT med_code, t.desc FROM read_parquet($parquet_path) AS t WHERE code_system = $system",
            params={"parquet_path": parquet_path, "system": system.value},
        )
        rows_per_fetch = client.get_max_batch_size()
        for chunk in results.fetch_record_batch(rows_per_batch=rows_per_fetch):
            med_codes = chunk["med_code"].to_pylist()
            descs = chunk["desc"].to_pylist()

            embeddings = []
            async for batch in embedder(descs):
                embeddings.extend(batch)

            # Add documents to collection
            collection.add(
                ids=med_codes,
                embeddings=embeddings,
                documents=descs,
            )

    con.close()


def load_chroma_client(
    db_dir: str,
) -> ClientAPI:
    """
    Load a ChromaDB client for the specified database path.

    Args:
        db_dir (str): Path to the ChromaDB database directory

    Returns:
        chromadb.PersistentClient: The ChromaDB client

    Raises:
        FileNotFoundError: If the ChromaDB database directory does not exist
    """
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"ChromaDB database not found at path: {db_dir}")
    settings = Settings(anonymized_telemetry=False)
    return chromadb.PersistentClient(path=db_dir, settings=settings)


def resolve_chroma_client(
    chroma_db: ClientAPI | pathlib.Path | None,
) -> ClientAPI:
    """Resolve chroma_db parameter to a ChromaDB client."""

    if chroma_db is None:
        # Download pre-generated ChromaDB files from Hugging Face into a dedicated
        # local_dir rather than the default HF hub cache. By default, snapshot_download
        # returns a snapshot directory containing symlinks into ~/.cache/huggingface/hub/blobs/,
        # where files are marked read-only (0444) to protect the content-addressed cache.
        # However, ChromaDB's Rust SQLite backend unconditionally executes write statements
        # on startup (CREATE TABLE IF NOT EXISTS migrations, INSERT OR REPLACE INTO
        # embeddings_queue_config, and schema migrations across ChromaDB versions), which
        # fails with "SQLITE_READONLY (code: 8): attempt to write a readonly database" if
        # pointed at read-only HF cache blobs. Passing local_dir materializes real writable
        # files outside the immutable blob store.
        # We only call snapshot_download if chroma.sqlite3 does not yet exist locally,
        # because ChromaDB modifies chroma.sqlite3 on startup (updating its mtime and
        # sha256), which would otherwise cause snapshot_download(local_dir=...) to think
        # chroma.sqlite3 is outdated and re-download the 1.92 GB unmigrated file every run.
        default_local_dir = (
            pathlib.Path.home()
            / ".cache"
            / "neopatient"
            / "CAB-Harvard--neopatient-Qwen3-Embedding-8B"
        )
        if not (default_local_dir / "chroma.sqlite3").exists():
            snapshot_download(
                "CAB-Harvard/neopatient-Qwen3-Embedding-8B",
                repo_type="dataset",
                revision="3eca3e5c188cd5a4ea54811a9271798262f2b101",
                local_dir=default_local_dir,
            )
        return load_chroma_client(str(default_local_dir))
    elif isinstance(chroma_db, pathlib.Path):
        return load_chroma_client(str(chroma_db))
    elif isinstance(chroma_db, ClientAPI):
        return chroma_db
    else:
        raise ValueError("chroma_db must be a ChromaDB client, a pathlib.Path, or None")
