"""Batch LLM abstraction for different providers."""

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod


from openai import AsyncOpenAI
from google import genai


logger = logging.getLogger(__name__)


class BatchLLM(ABC):
    """Abstract base class for batch LLM operations."""

    @abstractmethod
    async def ask(
        self, prompts_by_id: dict[str, str], response_schema: dict, model: str
    ) -> str:
        """Submit batch requests and return batch ID."""
        pass

    @abstractmethod
    async def is_done(self, batch_id: str) -> bool:
        """Check if batch is completed. Returns True if done, False if still processing, raises RuntimeError if failed."""
        pass

    @abstractmethod
    async def get(self, batch_id: str) -> dict[str, str]:
        """Retrieve completed batch results."""
        pass


class BatchOpenAI(BatchLLM):
    """OpenAI batch implementation."""

    def __init__(self):
        self.client = AsyncOpenAI()

    async def _create_jsonl_file(
        self, prompts_by_id: dict[str, str], response_schema: dict, model: str
    ) -> str:
        """Create JSONL file for OpenAI batch API."""
        requests = []
        for custom_id, prompt in prompts_by_id.items():
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
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

        jsonl_content = "\n".join(
            json.dumps(request, default=str) for request in requests
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content + "\n")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                file_response = await self.client.files.create(file=f, purpose="batch")
            return file_response.id
        finally:
            os.unlink(temp_path)

    async def _download_batch_results(self, file_id: str) -> list[dict]:
        """Download and parse batch results from OpenAI."""
        file_content = await self.client.files.content(file_id)
        results = []

        for line in file_content.text.split("\n"):
            if line.strip():
                results.append(json.loads(line))

        return results

    async def ask(
        self, prompts_by_id: dict[str, str], response_schema: dict, model: str
    ) -> str:
        """Submit OpenAI batch requests."""
        logger.info(
            f"Submitting OpenAI batch request: prompts={json.dumps(prompts_by_id)}, schema={json.dumps(response_schema)}, model={model}"
        )
        input_file_id = await self._create_jsonl_file(
            prompts_by_id, response_schema, model
        )

        batch_response = await self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        logger.info(
            f"Submitted OpenAI batch {batch_response.id} with {len(prompts_by_id)} prompts using model {model}"
        )
        return batch_response.id

    async def is_done(self, batch_id: str) -> bool:
        """Check if OpenAI batch is completed."""
        batch_status = await self.client.batches.retrieve(batch_id)
        status = batch_status.status

        if status == "completed":
            logger.info(f"OpenAI batch {batch_id} status: completed")
            return True
        elif status in ["failed", "expired", "cancelled"]:
            logger.error(f"OpenAI batch {batch_id} failed with status: {status}")
            raise RuntimeError(f"OpenAI batch {batch_id} failed with status: {status}")
        else:
            # Still processing (running, validating, etc.)
            logger.info(f"OpenAI batch {batch_id} status: {status} (still processing)")
            return False

    async def get(self, batch_id: str) -> dict[str, str]:
        """Retrieve OpenAI batch results."""
        batch_info = await self.client.batches.retrieve(batch_id)
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
        return response_data


class BatchGemini(BatchLLM):
    """Gemini batch implementation."""

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("OPENAI_API_KEY")).aio

    async def ask(
        self, prompts_by_id: dict[str, str], response_schema: dict, model: str
    ) -> str:
        """Submit Gemini batch requests."""
        logger.info(
            f"Submitting Gemini batch request: prompts={json.dumps(prompts_by_id)}, schema={json.dumps(response_schema)}, model={model}"
        )
        gemini_requests = []
        for custom_id, prompt in prompts_by_id.items():
            gemini_req = {
                "key": custom_id,
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "config": {
                        "response_mime_type": "application/json",
                        "response_json_schema": response_schema,
                    },
                },
            }
            gemini_requests.append(gemini_req)

        batch_job = await self.client.batches.create(
            model=model,
            src=gemini_requests,
        )

        logger.info(
            f"Submitted Gemini batch {batch_job.name} with {len(prompts_by_id)} prompts using model {model}"
        )
        return batch_job.name

    async def is_done(self, batch_id: str) -> bool:
        """Check if Gemini batch is completed."""
        batch_job = await self.client.batches.get(name=batch_id)
        state = batch_job.state

        # Gemini batch states
        if state == "JOB_STATE_SUCCEEDED":
            logger.info(f"Gemini batch {batch_id} status: succeeded")
            return True
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
            logger.error(f"Gemini batch {batch_id} failed with state: {state}")
            raise RuntimeError(f"Gemini batch {batch_id} failed with state: {state}")
        else:
            # Still processing (JOB_STATE_RUNNING, JOB_STATE_PENDING, etc.)
            logger.info(f"Gemini batch {batch_id} status: {state} (still processing)")
            return False

    async def get(self, batch_id: str) -> dict[str, str]:
        """Retrieve Gemini batch results."""
        batch_job = await self.client.batches.get(name=batch_id)

        # Gemini returns results inline, not as a file
        response_data = {}

        if hasattr(batch_job, "results") and batch_job.results:
            for i, result in enumerate(batch_job.results):
                custom_id = getattr(result, "key", f"gemini_result_{i}")
                content = result.text
                response_data[custom_id] = content

        logger.info(
            f"Retrieved results for Gemini batch {batch_id}: {len(response_data)} results"
        )
        return response_data


def create_batch_llm(model: str) -> BatchLLM:
    """Factory function to create appropriate BatchLLM implementation."""
    if "gemini" in model.lower():
        return BatchGemini()
    else:
        return BatchOpenAI()
