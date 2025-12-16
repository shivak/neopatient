import argparse
import asyncio
import logging
import os
import pathlib
import sys
import pyarrow as pa
from pyarrow import parquet
from openai import AsyncOpenAI
from . import synthesize_patient, synthesize_cohorts_with_state_file
from .models import CohortSpec, RecordType
from .cli_common import (
    add_embedder_args,
    add_specification_args,
    add_synthesis_args,
    add_state_args,
)
from .llm import apply_rate_limiting


async def _main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Single patient subcommand
    single_parser = subparsers.add_parser(
        "single", help="Generate a single patient record"
    )
    add_embedder_args(single_parser)
    add_specification_args(single_parser)
    add_synthesis_args(single_parser)

    # Cohort subcommand
    cohort_parser = subparsers.add_parser(
        "cohort", help="Generate a cohort of patient records"
    )
    add_embedder_args(cohort_parser)
    add_specification_args(cohort_parser)
    add_synthesis_args(cohort_parser)
    add_state_args(cohort_parser)
    cohort_parser.add_argument(
        "--size", type=int, required=True, help="Size of the cohort"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=getattr(
            logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING
        ),
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Set ChromaDB parameter
    chroma_db = pathlib.Path(args.db_dir) if args.db_dir else None

    if args.command == "single":
        # Create OpenAI client
        client = AsyncOpenAI(max_retries=0)

        # Apply rate limiting if requested
        if hasattr(args, "llm_per_min") and args.llm_per_min is not None:
            apply_rate_limiting(client, requests_per_minute=args.llm_per_min)

        record = await synthesize_patient(
            client,
            positive=args.positive,
            negative=args.negative,
            patient_id=1,
            chroma_db=chroma_db,
            embedder_model=args.embedder,
            embedder_batch_size=args.embedder_batch_size,
            embedder_args=args.embedder_args,
            embedder_base_url=args.embedder_base_url,
            generator=args.generator,
            verifier=args.verifier,
            record_type=RecordType(args.record_type),
            sampler=args.sampler,
        )
        logger.info("Patient record generated")
        parquet.write_table(record, args.out)

    elif args.command == "cohort":
        cohort_specs = [
            CohortSpec(
                positive=args.positive,
                negative=args.negative,
                count=args.size,
                record_type=RecordType(args.record_type),
            )
        ]
        cohorts = await synthesize_cohorts_with_state_file(
            cohort_specs=cohort_specs,
            chroma_db=chroma_db,
            embedder_model=args.embedder,
            embedder_batch_size=args.embedder_batch_size,
            embedder_args=args.embedder_args,
            embedder_base_url=args.embedder_base_url,
            generator=args.generator,
            verifier=args.verifier,
            sampler=args.sampler,
            state_file=pathlib.Path(args.state_file),
            poll_interval=args.poll_interval,
        )
        cohort = cohorts[0]  # list of Patient tables
        big_table = pa.concat_tables(cohort)
        parquet.write_table(big_table, args.out)
        logger.info("Cohort generated and written to %s", args.out)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
