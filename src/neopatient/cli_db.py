import argparse
import asyncio
import logging
import os
from .database import create_database
from .embed import create_embedder
from .cli_common import add_embedder_args


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
    add_embedder_args(parser)
    parser.set_defaults(embedder_args={"model_kwargs": {"device_map": "auto"}})

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(
            logging, os.environ.get("LOGLEVEL", "INFO").upper(), logging.INFO
        ),
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    embedder = create_embedder(
        args.embedder,
        args.embedder_batch_size,
        args.embedder_args,
        args.embedder_base_url,
    )
    logger.info(f"Using embedder: {args.embedder}")

    logger.info("Setting up databases...")
    await create_database(args.parquet_path, embedder, args.db_dir)
    logger.info("Database setup completed")
    print(f"ChromaDB created successfully at {args.db_dir}")


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
