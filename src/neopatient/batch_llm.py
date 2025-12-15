"""Batch LLM abstraction for different providers."""

import json
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List

from openai import AsyncOpenAI
from google import genai


class BatchLLM(ABC):
    """Abstract base class for batch LLM operations."""

    @abstractmethod
    async def ask(
        self, prompts_by_id: Dict[str, str], response_schema: dict, model: str
    ) -> str:
        """Submit batch requests and return batch ID."""
        pass

    @abstractmethod
    async def is_done(self, batch_id: str) -> bool:
        """Check if batch is completed. Returns True if done, False if still processing, raises RuntimeError if failed."""
        pass

    @abstractmethod
    async def get(self, batch_id: str) -> List[Dict]:
        """Retrieve completed batch results."""
        pass


class BatchOpenAI(BatchLLM):
    """OpenAI batch implementation."""

    def __init__(self):
        self.client = AsyncOpenAI()

    async def _create_jsonl_file(
        self, prompts_by_id: Dict[str, str], response_schema: dict, model: str
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

    async def _download_batch_results(self, file_id: str) -> List[Dict]:
        """Download and parse batch results from OpenAI."""
        file_content = await self.client.files.content(file_id)
        results = []

        for line in file_content.text.split("\n"):
            if line.strip():
                results.append(json.loads(line))

        return results

    async def ask(
        self, prompts_by_id: Dict[str, str], response_schema: dict, model: str
    ) -> str:
        """Submit OpenAI batch requests."""
        input_file_id = await self._create_jsonl_file(
            prompts_by_id, response_schema, model
        )

        batch_response = await self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        return batch_response.id

    async def is_done(self, batch_id: str) -> bool:
        """Check if OpenAI batch is completed."""
        batch_status = await self.client.batches.retrieve(batch_id)
        status = batch_status.status

        if status == "completed":
            return True
        elif status in ["failed", "expired", "cancelled"]:
            raise RuntimeError(f"OpenAI batch {batch_id} failed with status: {status}")
        else:
            # Still processing (running, validating, etc.)
            return False

    async def get(self, batch_id: str) -> List[Dict]:
        """Retrieve OpenAI batch results."""
        batch_info = await self.client.batches.retrieve(batch_id)
        if not hasattr(batch_info, "output_file_id") or not batch_info.output_file_id:
            raise ValueError(f"No output file for batch {batch_id}")

        return await self._download_batch_results(batch_info.output_file_id)


class BatchGemini(BatchLLM):
    """Gemini batch implementation."""

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        self._batch_id_to_custom_ids = {}  # batch_id -> list of custom_ids

    async def ask(
        self, prompts_by_id: Dict[str, str], response_schema: dict, model: str
    ) -> str:
        """Submit Gemini batch requests."""
        gemini_requests = []
        custom_ids = []
        for custom_id, prompt in prompts_by_id.items():
            gemini_req = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "config": {
                    "response_mime_type": "application/json",
                    "response_json_schema": response_schema,
                },
            }
            gemini_requests.append(gemini_req)
            custom_ids.append(custom_id)

        batch_job = await self.client.batches.create(
            model=model,
            src=gemini_requests,
        )

        # Store custom_ids for this batch
        self._batch_id_to_custom_ids[batch_job.name] = custom_ids

        return batch_job.name

    async def is_done(self, batch_id: str) -> bool:
        """Check if Gemini batch is completed."""
        batch_job = await self.client.batches.get(name=batch_id)
        state = batch_job.state

        # Gemini batch states
        if state == "JOB_STATE_SUCCEEDED":
            return True
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
            raise RuntimeError(f"Gemini batch {batch_id} failed with state: {state}")
        else:
            # Still processing (JOB_STATE_RUNNING, JOB_STATE_PENDING, etc.)
            return False

    async def get(self, batch_id: str) -> List[Dict]:
        """Retrieve Gemini batch results."""
        batch_job = await self.client.batches.get(name=batch_id)

        # Gemini returns results inline, not as a file
        results = []
        custom_ids = self._batch_id_to_custom_ids.get(batch_id, [])

        if hasattr(batch_job, "results") and batch_job.results:
            for i, result in enumerate(batch_job.results):
                custom_id = (
                    custom_ids[i] if i < len(custom_ids) else f"gemini_result_{i}"
                )
                # Convert Gemini result format to OpenAI-style format
                openai_result = {
                    "custom_id": custom_id,
                    "response": {
                        "body": {
                            "choices": [
                                {"message": {"content": getattr(result, "text", "")}}
                            ]
                        }
                    },
                }
                results.append(openai_result)

        return results


def create_batch_llm(model: str) -> BatchLLM:
    """Factory function to create appropriate BatchLLM implementation."""
    if "gemini" in model.lower():
        return BatchGemini()
    else:
        return BatchOpenAI()
