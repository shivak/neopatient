import argparse
import logging
import sys
from neopatient import setup_databases

def main():
    parser = argparse.ArgumentParser(description="Neopatient embed tool for creating ChromaDB from parquet file")

    parser.add_argument("--parquet_path", required=True, help="Path to clinprime_mapping.parquet file")
    parser.add_argument("--chroma_db_path", default="clinprime_chroma", help="Path to ChromaDB database directory")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("Setting up databases...")
        setup_databases(args.parquet_path, args.chroma_db_path)
        logger.info("Database setup completed")
        print(f"ChromaDB created successfully at {args.chroma_db_path}")
    except Exception as e:
        logger.error(f"Error during database setup: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()