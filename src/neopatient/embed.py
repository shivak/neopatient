from typing import AsyncGenerator, Callable, List
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI

# Type alias for an embedder function
Embed = Callable[[List[str]], AsyncGenerator[List[List[float]], None]]


def sentence_embedder(model_name: str, batch_size: int, **kwargs) -> Embed:
    """Create a sentence transformer embedder.

    Args:
        model_name: HuggingFace model name
        batch_size: Batch size for encoding operations
        **kwargs: Additional arguments passed to SentenceTransformer constructor

    Returns:
        Async embedder function that yields batches of embeddings
    """
    kwargs.setdefault("tokenizer_kwargs", {})
    kwargs["tokenizer_kwargs"]["padding_side"] = "left"
    model = SentenceTransformer(model_name, **kwargs)

    async def embed(texts: List[str]) -> AsyncGenerator[List[List[float]], None]:
        # Truncate texts to 256 characters
        truncated_texts = [text[:256] for text in texts]

        # Encode in batches using chromadb's create_batches
        embeddings = model.encode(
            truncated_texts, normalize_embeddings=True, batch_size=batch_size
        )

        yield embeddings.tolist()

    return embed


def openai_embedder(
    model_name: str, batch_size: int, embedder_base_url: str | None = None, **kwargs
) -> Embed:
    """Create an OpenAI embedder.

    Args:
        model_name: OpenAI model name
        batch_size: Batch size for API calls (default: 100)
        embedder_base_url: Base URL for OpenAI-compatible API (optional)
        **kwargs: Additional arguments passed to embeddings.create() call

    Returns:
        Async embedder function that yields batches of embeddings
    """
    client = AsyncOpenAI(base_url=embedder_base_url, max_retries=0)

    async def embed(texts: List[str]) -> AsyncGenerator[List[List[float]], None]:
        # Truncate texts to 256 characters
        truncated_texts = [text[:256] for text in texts]

        # Batch size for OpenAI API
        for i in range(0, len(truncated_texts), batch_size):
            batch_texts = truncated_texts[i : i + batch_size]

            response = await client.embeddings.create(
                model=model_name, input=batch_texts, **kwargs
            )

            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            yield embeddings

    return embed


def create_embedder(
    embedder_model: str,
    batch_size: int,
    embedder_args: dict,
    embedder_base_url: str | None = None,
) -> Embed:
    """Create an embedder from model name and args dict.

    Args:
        embedder_model: Model name
        batch_size: Batch size for embedding operations
        embedder_args: Dictionary of embedder arguments
        embedder_base_url: Base URL for OpenAI-compatible API (uses OpenAI embedder if provided)

    Returns:
        Embedder function
    """
    if embedder_base_url is not None:
        return openai_embedder(
            embedder_model,
            batch_size=batch_size,
            embedder_base_url=embedder_base_url,
            **embedder_args,
        )
    else:
        return sentence_embedder(embedder_model, batch_size=batch_size, **embedder_args)
