import argparse
import asyncio
import json
import logging
import sys
from .database import create_database
from .embed import create_embedder


async def _main():
    parser = argparse.ArgumentParser(
        description="Create neopatient vector database for looking up medical codes"
    )

    parser.add_argument(
        "--parquet_path",
        required=True,
        help="Path to parquet file with med_code, desc columns",
    )
    parser.add_argument(
        "--db_dir",
        help="Path to ChromaDB database directory",
    )
    parser.add_argument(
        "--embedder",
        default="Qwen/Qwen3-Embedding-4B",
        help="Embedder model name (HF if contains '/', OpenAI otherwise)",
    )
    parser.add_argument(
        "--embedder-args",
        type=json.loads,
        default={"model_kwargs": {"device_map": "auto"}},
        help="Embedder arguments as JSON dict",
    )
    parser.add_argument(
        "--embedder-batch-size",
        type=int,
        default=128,
        help="Batch size for embedding operations",
    )

    args = parser.parse_args()

    # Validate embedder_batch_size
    if args.embedder_batch_size <= 0:
        print(
            f"Error: --embedder-batch-size must be a positive integer, got {args.embedder_batch_size}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    try:
        embedder = create_embedder(
            args.embedder, args.embedder_batch_size, args.embedder_args
        )
        logger.info(f"Using embedder: {args.embedder}")

        logger.info("Setting up databases...")
        await create_database(args.parquet_path, embedder, args.db_dir)
        logger.info("Database setup completed")
        print(f"ChromaDB created successfully at {args.db_dir}")
    except Exception as e:
        logger.error(f"Error during database setup: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
