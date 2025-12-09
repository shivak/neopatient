import argparse
import asyncio
import logging
import pathlib
import sys
import pyarrow as pa
from pyarrow import parquet
from . import synthesize_patient, synthesize_cohort_with_state_file
from .models import CohortSpec, RecordType
from .cli_common import (
    add_embedder_args,
    add_specification_args,
    add_synthesis_args,
    add_state_args,
)


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
    single_parser.set_defaults(embedder_args={"model_kwargs": {"dtype": "bfloat16"}})

    # Cohort subcommand
    cohort_parser = subparsers.add_parser(
        "cohort", help="Generate a cohort of patient records"
    )
    add_embedder_args(cohort_parser)
    add_specification_args(cohort_parser)
    add_synthesis_args(cohort_parser)
    add_state_args(cohort_parser)
    cohort_parser.set_defaults(embedder="sentence-transformers/all-MiniLM-L6-v2")
    cohort_parser.add_argument(
        "--size", type=int, required=True, help="Size of the cohort"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Validate embedder_batch_size
    if args.embedder_batch_size <= 0:
        print(
            f"Error: --embedder-batch-size must be a positive integer, got {args.embedder_batch_size}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Set ChromaDB parameter
    chroma_db = pathlib.Path(args.db_dir) if args.db_dir else None

    if args.command == "single":
        try:
            logger.info("Generating patient record...")
            record = await synthesize_patient(
                positive=args.positive,
                negative=args.negative,
                patient_id=1,
                chroma_db=chroma_db,
                embedder_model=args.embedder,
                embedder_batch_size=args.embedder_batch_size,
                embedder_args=args.embedder_args,
                embedder_base_url=args.embedder_base_url,
                seed=args.seed,
                generator=args.generator,
                verifier=args.verifier,
                record_type=args.record_type,
                sampler=args.sampler,
            )
            logger.info("Patient record generated")
            parquet.write_table(record, args.out)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "cohort":
        try:
            logger.info("Generating patient cohort...")
            cohort_specs = [
                CohortSpec(
                    positive=[args.positive],
                    negative=[args.negative],
                    count=args.size,
                    record_type=RecordType(args.record_type),
                )
            ]
            result = await synthesize_cohort_with_state_file(
                cohort_specs=cohort_specs,
                chroma_db=chroma_db,
                embedder_model=args.embedder,
                embedder_batch_size=args.embedder_batch_size,
                embedder_args=args.embedder_args,
                embedder_base_url=args.embedder_base_url,
                generator=args.generator,
                verifier=args.verifier,
                sampler=args.sampler,
                state_file=args.state_file,
                poll_interval=args.poll_interval,
            )
            cohort = result[0]  # list of Patient tables
            big_table = pa.concat_tables(cohort)
            parquet.write_table(big_table, args.out)
            logger.info("Cohort generated and written to %s", args.out)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
