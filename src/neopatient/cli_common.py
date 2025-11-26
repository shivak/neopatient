import json
from .models import RecordType


def add_embedder_args(parser):
    parser.add_argument(
        "--embedder",
        default="abhinand/MedEmbed-large-v0.1",
        help="Embedder model name (HF if contains '/', OpenAI otherwise)",
    )
    parser.add_argument(
        "--embedder-args",
        type=json.loads,
        default={},
        help="JSON dict for embedder configuration",
    )
    parser.add_argument(
        "--embedder-batch-size",
        type=int,
        default=128,
        help="Batch size for embedding operations",
    )


def add_generation_args(parser):
    parser.add_argument(
        "--positive", required=True, help="Positive patient description"
    )
    parser.add_argument(
        "--negative", required=True, help="Negative patient description"
    )
    parser.add_argument("--out", required=True, help="Output path for parquet file")
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducibility"
    )
    parser.add_argument(
        "--db_dir",
        default=None,
        help="Path to ChromaDB database directory, or None to download from Hugging Face",
    )
    parser.add_argument(
        "--generator", default="gpt-5-nano", help="Model name for generation"
    )
    parser.add_argument(
        "--verifier", default="gpt-5", help="Model name for verification"
    )
    parser.add_argument(
        "--record-type",
        default="ehr-outpatient",
        choices=[e.value for e in RecordType],
        help="Type of record",
    )
