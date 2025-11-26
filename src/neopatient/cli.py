import argparse
import asyncio
import json
import logging
import pathlib
import sys
import pyarrow as pa
from pyarrow import parquet
from . import synthesize_patient, synthesize_cohort_with_state_file
from .models import CohortSpec, RecordType


async def _main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Single patient subcommand
    single_parser = subparsers.add_parser(
        "single", help="Generate a single patient record"
    )
    single_parser.add_argument(
        "--positive", required=True, help="Positive patient description"
    )
    single_parser.add_argument(
        "--negative", required=True, help="Negative patient description"
    )
    single_parser.add_argument(
        "--out", required=True, help="Output path for parquet file"
    )
    single_parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducibility"
    )
    single_parser.add_argument(
        "--db_dir",
        default=None,
        help="Path to ChromaDB database directory, or None to download from Hugging Face",
    )
    single_parser.add_argument(
        "--generator", default="gpt-5-nano", help="Model name for generation"
    )
    single_parser.add_argument(
        "--verifier", default="gpt-5", help="Model name for verification"
    )
    single_parser.add_argument(
        "--record-type",
        default="ehr-outpatient",
        choices=[e.value for e in RecordType],
        help="Type of record",
    )
    single_parser.add_argument(
        "--end-date",
        default=None,
        help="End date for the record (ISO string), defaults to current time",
    )
    single_parser.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedder model name for code matching",
    )
    single_parser.add_argument(
        "--embedder-args",
        type=json.loads,
        default={},
        help="JSON dict for embedder configuration",
    )
    single_parser.add_argument(
        "--embedder-batch-size",
        type=int,
        default=128,
        help="Batch size for embedding operations",
    )

    # Cohort subcommand
    cohort_parser = subparsers.add_parser(
        "cohort", help="Generate a cohort of patient records"
    )
    cohort_parser.add_argument(
        "--positive", required=True, help="Positive cohort description"
    )
    cohort_parser.add_argument(
        "--negative", required=True, help="Negative cohort description"
    )
    cohort_parser.add_argument(
        "--size", type=int, required=True, help="Size of the cohort"
    )
    cohort_parser.add_argument(
        "--out", required=True, help="Output path for parquet file"
    )
    cohort_parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducibility"
    )
    cohort_parser.add_argument(
        "--db_dir",
        default=None,
        help="Path to ChromaDB database directory, or None to download from Hugging Face",
    )
    cohort_parser.add_argument(
        "--generator", default="gpt-5-nano", help="Model name for generation"
    )
    cohort_parser.add_argument(
        "--verifier", default="gpt-5", help="Model name for verification"
    )
    cohort_parser.add_argument(
        "--sampler", default="gpt-5", help="Model name for sampling"
    )
    cohort_parser.add_argument(
        "--record-type",
        default="ehr-outpatient",
        choices=[e.value for e in RecordType],
        help="Type of record",
    )
    cohort_parser.add_argument(
        "--poll_interval",
        type=int,
        default=15 * 60,
        help="Polling interval in seconds for batch completion",
    )
    cohort_parser.add_argument(
        "--state-file",
        required=True,
        help="Path to state file for resuming batch operations",
    )
    cohort_parser.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedder model name for code matching",
    )
    cohort_parser.add_argument(
        "--embedder-args",
        type=json.loads,
        default={},
        help="JSON dict for embedder configuration",
    )
    cohort_parser.add_argument(
        "--embedder-batch-size",
        type=int,
        default=128,
        help="Batch size for embedding operations",
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
        level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
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
                seed=args.seed,
                generator=args.generator,
                verifier=args.verifier,
                record_type=args.record_type,
                end_date=args.end_date,
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
