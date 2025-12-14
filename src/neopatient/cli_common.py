import json
from .models import RecordType


def add_embedder_args(parser):
    parser.add_argument(
        "--embedder",
        help="Embedder model name",
    )
    parser.add_argument(
        "--embedder-base-url",
        help="Base URL for OpenAI-compatible API (uses OpenAI embedder if provided)",
    )
    parser.add_argument(
        "--embedder-args",
        type=json.loads,
        help="JSON dict for embedder configuration",
    )
    parser.add_argument(
        "--embedder-batch-size",
        type=int,
        help="Batch size for embedding operations",
    )


def add_specification_args(parser):
    parser.add_argument(
        "--positive", required=True, help="Positive patient description"
    )
    parser.add_argument(
        "--negative", required=True, help="Negative patient description"
    )
    parser.add_argument("--out", required=True, help="Output path for parquet file")
    parser.add_argument(
        "--record-type",
        default="ehr-outpatient",
        choices=[e.value for e in RecordType],
        help="Type of record",
    )


def add_synthesis_args(parser):
    parser.add_argument(
        "--generator", default="gpt-5-nano", help="Model name for generation"
    )
    parser.add_argument(
        "--verifier", default="gpt-5", help="Model name for verification"
    )
    parser.add_argument(
        "--sampler",
        default="gpt-5",
        help="Model name for sampling individualized descriptions",
    )
    parser.add_argument(
        "--llm-per-min",
        type=int,
        help="Maximum LLM requests per minute (optional rate limiting)",
    )
    parser.add_argument(
        "--db_dir",
        default=None,
        help="Path to ChromaDB database directory, or None to download from Hugging Face",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )


def add_state_args(parser):
    parser.add_argument("--state-file", required=True, help="State file for resuming")
    parser.add_argument(
        "--poll-interval", type=int, default=900, help="Poll interval in seconds"
    )
