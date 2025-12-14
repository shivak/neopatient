from openai import AsyncOpenAI
from limiter import Limiter


def apply_rate_limiting(client: AsyncOpenAI, requests_per_minute: int) -> None:
    """Apply rate limiting to an AsyncOpenAI client's chat completions.

    Args:
        client: The AsyncOpenAI client to modify
        requests_per_minute: Maximum requests per minute
    """
    limiter = Limiter(rate=requests_per_minute / 60, capacity=requests_per_minute)
    original_create = client.chat.completions.create

    async def rate_limited_create(**kwargs):
        async with limiter:
            return await original_create(**kwargs)

    client.chat.completions.create = rate_limited_create
