import argparse
import json
import logging
import pathlib
import sys
from neopatient import generate_synthetic_patient_record, generate_synthetic_patient_records_batch
from neopatient.sampler import sample_individual_patients

def main():
    parser = argparse.ArgumentParser(description="Neopatient CLI for generating synthetic patient records")
    
    # Mode selection
    parser.add_argument("--mode", choices=["single", "batch"], default="single", 
                       help="Generation mode: single record or batch")
    
    # Single mode arguments
    parser.add_argument("--positive", help="Positive cohort description")
    parser.add_argument("--negative", help="Negative (anti-cohort) description")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    
    # Batch mode arguments
    parser.add_argument("--batch_config", help="Path to batch configuration JSON file")
    parser.add_argument("--state_file", help="Path to state file for resuming batch generation")
    parser.add_argument("--epsilon", type=float, default=0.2, 
                       help="Over-generation factor for batch mode (default: 0.2)")
    
    # Common arguments
    parser.add_argument("--chroma_db_path", default=None, help="Path to ChromaDB database directory, or None to download from Hugging Face")
    parser.add_argument("--generator", default="gpt-5-nano", help="Model name for generation")
    parser.add_argument("--verifier", default="gpt-5", help="Model name for verification")
    parser.add_argument("--sampler", default="gpt-5", help="Model name for sampling")
    
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

    if args.mode == "single":
        if not args.positive or not args.negative:
            print("Error: --positive and --negative are required for single mode", file=sys.stderr)
            sys.exit(1)
        
        try:
            logger.info("Sampling individual patient...")
            sampled = sample_individual_patients(
                positive=args.positive,
                negative=args.negative,
                n=1,
                sampler_model=args.sampler
            )
            patient_id, individual_description = next(iter(sampled.items()))
            logger.info("Generating patient record...")
            record = generate_synthetic_patient_record(
                positive=args.positive,
                negative=args.negative,
                individual_description=individual_description,
                patient_id=patient_id,
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
    
    elif args.mode == "batch":
        if not args.batch_config and not args.state_file:
            print("Error: Either --batch_config or --state_file is required for batch mode", file=sys.stderr)
            sys.exit(1)
        
        try:
            # Load state if resuming
            state = None
            if args.state_file:
                logger.info("Loading state...")
                with open(args.state_file, 'r') as f:
                    state = json.load(f)
            
            # Load batch config if starting new
            cohort_specs = None
            if args.batch_config:
                logger.info("Loading batch configuration...")
                with open(args.batch_config, 'r') as f:
                    cohort_specs = json.load(f)
            elif state:
                cohort_specs = state.get("cohort_specs")
            
            if not cohort_specs:
                print("Error: No cohort specifications found", file=sys.stderr)
                sys.exit(1)
            
            logger.info("Starting batch generation...")
            result = generate_synthetic_patient_records_batch(
                cohort_specs=cohort_specs,
                chroma_db=chroma_db,
                epsilon=args.epsilon,
                state=state,
                generator=args.generator,
                verifier=args.verifier,
                sampler=args.sampler
            )
            
            # Check if result is state (needs resuming) or final results
            if isinstance(result, dict) and "stage" in result:
                # Save state for resuming
                if args.state_file:
                    logger.info("Saving state...")
                    with open(args.state_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    print(f"Batch generation in progress. State saved to {args.state_file}")
                    print(f"Current stage: {result['stage']}")
                    if result.get("generation_tickets"):
                        print(f"Generation batch ID: {result['generation_tickets'][0]}")
                    if result.get("verification_tickets"):
                        print(f"Verification batch ID: {result['verification_tickets'][0]}")
                else:
                    print("Error: State file required for resuming batch operations", file=sys.stderr)
                    sys.exit(1)
            else:
                # Final results
                logger.info("Batch generation completed")
                print(json.dumps(result, indent=2, default=str))
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
