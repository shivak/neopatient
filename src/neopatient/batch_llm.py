"""Batch LLM abstraction for different providers."""

import io
import json
import logging
import os
from abc import ABC, abstractmethod


from limiter import Limiter
from openai import AsyncOpenAI
from google import genai


logger = logging.getLogger(__name__)


class BatchLLM(ABC):
    """Abstract base class for batch LLM operations."""

    def __init__(self, model: str, poll_interval: int | None = None):
        self.model = model
        if poll_interval is not None:
            self.limiter = Limiter(
                rate=1, capacity=poll_interval, consume=poll_interval
            )
        else:
            self.limiter = None

    async def _wait_for_limiter(self):
        if self.limiter:
            async with self.limiter:
                pass

    @abstractmethod
    async def ask(self, prompts_by_id: dict[str, str], response_schema: dict) -> str:
        """Submit batch requests and return batch ID."""
        pass

    @abstractmethod
    async def get(self, batch_id: str) -> dict[str, str] | None:
        """Retrieve completed batch results. Returns a dict where keys are prompt IDs and values are JSON response strings, or None if the batch is still processing. Raises RuntimeError if the batch failed, was cancelled, or expired."""
        pass


class BatchOpenAI(BatchLLM):
    """OpenAI batch implementation."""

    def __init__(self, model: str, poll_interval: int | None = None):
        super().__init__(model, poll_interval)
        self.client = AsyncOpenAI()

    async def _create_jsonl_file(
        self, prompts_by_id: dict[str, str], response_schema: dict
    ) -> str:
        """Create JSONL file for OpenAI batch API."""
        requests = []
        for custom_id, prompt in prompts_by_id.items():
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "strict": True,
                            "schema": response_schema,
                        },
                    },
                    "temperature": 0.7,
                },
            }
            requests.append(request)

        jsonl_content = (
            "\n".join(json.dumps(request, default=str) for request in requests) + "\n"
        )

        jsonl_bytes = jsonl_content.encode("utf-8")
        bytes_buffer = io.BytesIO(jsonl_bytes)
        file_response = await self.client.files.create(
            file=bytes_buffer, purpose="batch"
        )
        return file_response.id

    async def ask(self, prompts_by_id: dict[str, str], response_schema: dict) -> str:
        """Submit OpenAI batch requests."""
        logger.info(
            f"Submitting OpenAI batch request: prompts={json.dumps(prompts_by_id)}, schema={json.dumps(response_schema)}, model={self.model}"
        )
        input_file_id = await self._create_jsonl_file(prompts_by_id, response_schema)

        batch_response = await self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        return batch_response.id

    async def _download_batch_results(self, file_id: str) -> list[dict]:
        """Download and parse batch results from OpenAI."""
        file_content = await self.client.files.content(file_id)
        results = []

        for line in file_content.text.split("\n"):
            if line.strip():
                results.append(json.loads(line))

        return results

    async def ask(self, prompts_by_id: dict[str, str], response_schema: dict) -> str:
        """Submit OpenAI batch requests."""
        logger.info(
            f"Submitting OpenAI batch request: prompts={json.dumps(prompts_by_id)}, schema={json.dumps(response_schema)}, model={self.model}"
        )
        input_file_id = await self._create_jsonl_file(prompts_by_id, response_schema)

        batch_response = await self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        logger.info(
            f"Submitted OpenAI batch {batch_response.id} with {len(prompts_by_id)} prompts using model {self.model}"
        )
        return batch_response.id

    async def get(self, batch_id: str) -> dict[str, str] | None:
        """Retrieve OpenAI batch results."""
        await self._wait_for_limiter()
        batch_info = await self.client.batches.retrieve(batch_id)
        status = batch_info.status

        if status == "completed":
            logger.info(f"OpenAI batch {batch_id} status: completed")
        elif status in ["failed", "expired", "cancelled"]:
            logger.error(f"OpenAI batch {batch_id} failed with status: {status}")
            raise RuntimeError(f"OpenAI batch {batch_id} failed with status: {status}")
        else:
            # Still processing (running, validating, etc.)
            logger.info(f"OpenAI batch {batch_id} status: {status} (still processing)")
            return None

        if not hasattr(batch_info, "output_file_id") or not batch_info.output_file_id:
            logger.error(f"No output file for OpenAI batch {batch_id}")
            raise ValueError(f"No output file for batch {batch_id}")

        results = await self._download_batch_results(batch_info.output_file_id)
        response_data = {}
        for result in results:
            status_code = result["response"]["status_code"]
            if status_code != 200:
                raise RuntimeError(
                    f"OpenAI batch result failed with status {status_code}"
                )
            custom_id = result["custom_id"]
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            response_data[custom_id] = content
        logger.info(
            f"Retrieved results for OpenAI batch {batch_id}: {len(response_data)} results"
        )
        logger.debug(response_data)
        return response_data


class BatchGemini(BatchLLM):
    """Gemini batch implementation."""

    def __init__(self, model: str, poll_interval: int | None = None):
        super().__init__(model, poll_interval)
        self.client = genai.Client(api_key=os.getenv("OPENAI_API_KEY")).aio

    async def _create_jsonl_file(self, jsonl_content: str) -> str:
        """Upload JSONL content for Gemini batch API."""
        jsonl_bytes = jsonl_content.encode("utf-8")
        bytes_buffer = io.BytesIO(jsonl_bytes)
        uploaded_file = await self.client.files.upload(
            file=bytes_buffer,
            config=genai.types.UploadFileConfig(mime_type="application/jsonl"),
        )
        return uploaded_file.name

    async def ask(self, prompts_by_id: dict[str, str], response_schema: dict) -> str:
        """Submit Gemini batch requests."""
        requests = []
        for custom_id, prompt in prompts_by_id.items():
            request = {
                "key": custom_id,
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generation_config": {
                        "response_mime_type": "application/json",
                        "response_json_schema": response_schema,
                    },
                },
            }
            requests.append(request)

        jsonl_content = "\n".join(json.dumps(request) for request in requests) + "\n"
        logger.debug(f"Uploading Gemini batch JSONL: {jsonl_content}")
        input_file = await self._create_jsonl_file(jsonl_content)

        batch_job = await self.client.batches.create(
            model=self.model,
            src=input_file,
        )

        logger.info(
            f"Submitted Gemini batch {batch_job.name} with {len(prompts_by_id)} prompts using model {self.model}"
        )
        return batch_job.name

    async def get(self, batch_id: str) -> dict[str, str] | None:
        """Retrieve Gemini batch results."""
        await self._wait_for_limiter()
        batch_job = await self.client.batches.get(name=batch_id)
        state = batch_job.state

        # Gemini batch states
        if state == "JOB_STATE_SUCCEEDED":
            logger.info(f"Gemini batch {batch_id} status: succeeded")
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
            logger.error(f"Gemini batch {batch_id} failed with state: {state}")
            raise RuntimeError(f"Gemini batch {batch_id} failed with state: {state}")
        else:
            # Still processing (JOB_STATE_RUNNING, JOB_STATE_PENDING, etc.)
            logger.info(f"Gemini batch {batch_id} status: {state} (still processing)")
            return None

        response_data = {}

        if batch_job.dest and batch_job.dest.file_name:
            file_content_bytes = await self.client.files.download(
                file=batch_job.dest.file_name
            )
            file_content = file_content_bytes.decode("utf-8")
            for line in file_content.split("\n"):
                if line.strip():
                    result = json.loads(line)
                    custom_id = result["key"]
                    content = result["response"]["candidates"][0]["content"]["parts"][
                        0
                    ]["text"]
                    response_data[custom_id] = content

        logger.info(
            f"Retrieved results for Gemini batch {batch_id}: {len(response_data)} results"
        )
        return response_data


def create_batch_llm(model: str, poll_interval: int | None = None) -> BatchLLM:
    """Factory function to create appropriate BatchLLM implementation."""
    if "gemini" in model.lower():
        return BatchGemini(model, poll_interval)
    else:
        return BatchOpenAI(model, poll_interval)
