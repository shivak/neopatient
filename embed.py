import argparse
import asyncio
import logging
import sys
from neopatient.database import create_database
from neopatient.embed import create_embedder


def main():
    parser = argparse.ArgumentParser(
        description="Neopatient embed tool for creating ChromaDB from parquet file"
    )

    parser.add_argument(
        "--parquet_path", required=True, help="Path to clinprime_mapping.parquet file"
    )
    parser.add_argument(
        "--db_dir",
        help="Path to ChromaDB database directory",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    try:
        embedder = create_embedder("sentence-transformers/all-MiniLM-L6-v2", "")
        logger.info("Setting up databases...")
        asyncio.run(create_database(args.parquet_path, embedder, args.db_dir))
        logger.info("Database setup completed")
        print(f"ChromaDB created successfully at {args.db_dir}")
    except Exception as e:
        logger.error(f"Error during database setup: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
