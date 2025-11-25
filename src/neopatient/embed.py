import argparse
import asyncio
import logging
import sys
from typing import AsyncGenerator, Callable, List
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI

# Type alias for an embedder function
Embed = Callable[[List[str]], AsyncGenerator[List[List[float]], None]]


def sentence_embedder(model_name: str, **kwargs) -> Embed:
    """Create a sentence transformer embedder.

    Args:
        model_name: HuggingFace model name
        **kwargs: Additional arguments passed to SentenceTransformer constructor

    Returns:
        Async embedder function that yields batches of embeddings
    """
    model = SentenceTransformer(model_name, **kwargs)

    async def embed(texts: List[str]) -> AsyncGenerator[List[List[float]], None]:
        # Truncate texts to 256 characters
        truncated_texts = [text[:256] for text in texts]

        # Encode in batches using chromadb's create_batches
        embeddings = model.encode(truncated_texts, normalize_embeddings=True)

        # Yield the full embeddings list (chromadb create_batches expects this format)
        yield embeddings.tolist()

    return embed


def openai_embedder(model_name: str, **kwargs) -> Embed:
    """Create an OpenAI embedder.

    Args:
        model_name: OpenAI model name
        **kwargs: Additional arguments passed to embeddings.create() call

    Returns:
        Async embedder function that yields batches of embeddings
    """
    client = AsyncOpenAI()

    async def embed(texts: List[str]) -> AsyncGenerator[List[List[float]], None]:
        # Truncate texts to 256 characters
        truncated_texts = [text[:256] for text in texts]

        # Batch size for OpenAI API (conservative limit)
        batch_size = 100

        for i in range(0, len(truncated_texts), batch_size):
            batch_texts = truncated_texts[i : i + batch_size]

            response = await client.embeddings.create(
                model=model_name, input=batch_texts, **kwargs
            )

            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            yield embeddings

    return embed


def parse_embedder_args(args_str: str) -> dict:
    """Parse embedder args string into kwargs dict.

    Args:
        args_str: Comma-separated key=value pairs (e.g., "a=b,c=d")

    Returns:
        Dictionary of parsed arguments
    """
    if not args_str:
        return {}

    kwargs = {}
    for pair in args_str.split(","):
        if "=" not in pair:
            raise ValueError(f"Invalid argument format: {pair}. Expected key=value")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to convert to appropriate type
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        elif value.replace(".", "").isdigit():
            value = float(value)

        kwargs[key] = value

    return kwargs


def create_embedder(embedder_model: str, embedder_args_str: str) -> Embed:
    """Create an embedder from model name and args string.

    Args:
        embedder_model: Model name (HF if contains '/', OpenAI otherwise)
        embedder_args_str: Comma-separated key=value pairs

    Returns:
        Embedder function
    """
    embedder_kwargs = parse_embedder_args(embedder_args_str)
    if "/" in embedder_model:
        # HuggingFace model
        return sentence_embedder(embedder_model, **embedder_kwargs)
    else:
        # OpenAI model
        return openai_embedder(embedder_model, **embedder_kwargs)
