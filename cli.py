import argparse
import json
import logging
import pathlib
import sys
from neopatient import generate_synthetic_patient_record

def main():
    parser = argparse.ArgumentParser(description="Neopatient CLI for generating synthetic patient records")


    # Arguments
    parser.add_argument("--positive", help="Positive cohort description")
    parser.add_argument("--negative", help="Negative (anti-cohort) description")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")

    # Common arguments
    parser.add_argument("--chroma_db_path", default=None, help="Path to ChromaDB database directory, or None to download from Hugging Face")
    parser.add_argument("--generator", default="gpt-5-nano", help="Model name for generation")
    parser.add_argument("--verifier", default="gpt-5", help="Model name for verification")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Set ChromaDB parameter
    chroma_db = pathlib.Path(args.chroma_db_path) if args.chroma_db_path else None

    if not args.positive or not args.negative:
        print("Error: --positive and --negative are required", file=sys.stderr)
        sys.exit(1)

    try:
        logger.info("Generating patient record...")
        record = generate_synthetic_patient_record(
            positive=args.positive,
            negative=args.negative,
            patient_id=1,
            chroma_db=chroma_db,
            seed=args.seed,
            generator=args.generator,
            verifier=args.verifier
        )
        logger.info("Patient record generated")
        print(json.dumps(record, indent=2, default=str))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()