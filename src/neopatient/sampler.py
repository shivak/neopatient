import logging
import os
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
import jinja2
import pandas as pd
from .models import PatientRecipe, SamplingResponse, RecordType

# Get the directory of the current file to construct template path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_sample_template_path = os.path.join(_project_root, "templates", "sample.jinja2")

SAMPLE_TEMPLATE = jinja2.Template(
    open(_sample_template_path, "r", encoding="utf-8").read()
)


def _get_csv_path(record_type: RecordType) -> str:
    if record_type == RecordType.EHR_INPATIENT:
        return os.path.join(_project_root, "stats", "ehr-inpatient.csv")
    else:
        return os.path.join(_project_root, "stats", "ehr-outpatient.csv")


def sample_patient_stats(csv_path: str, n: int) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    sampled_df = df.sample(n=n, replace=True)
    return sampled_df.to_dict("records")


async def sample_recipes(
    client: AsyncOpenAI,
    positive: str,
    negative: str,
    n: int,
    record_type: RecordType,
    sampler_model: str = "gpt-5",
    logger: Optional[logging.Logger] = None,
) -> Dict[int, PatientRecipe]:
    """
    Samples individual patient recipes that satisfy cohort criteria.

    Uses LLM to generate n self-contained recipes, each including start_date, end_date, and description,
    constrained by the sampled duration.

    Args:
        client: AsyncOpenAI client instance
        positive: Positive cohort description
        negative: Negative anti-cohort description
        n: Number of patients to sample
         record_type: Type of record (RecordType enum)
        sampler_model: Model name for sampling (default: "gpt-5")
        logger: Optional logger for logging the LLM response

    Returns:
        Dict of {patient_id: PatientRecipe}
    """

    csv_path = _get_csv_path(record_type)
    stats = sample_patient_stats(csv_path, n)

    prompt = SAMPLE_TEMPLATE.render(
        positive_cohort=positive, negative_cohort=negative, n=n, stats=stats
    )
    if logger:
        logger.info(f"Sampling prompt: {prompt}")
    response = await client.chat.completions.create(
        model=sampler_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "patient_recipes",
                "strict": True,
                "schema": SamplingResponse.model_json_schema(),
            },
        },
        temperature=0.7,
    )
    content = response.choices[0].message.content
    if logger:
        logger.info(f"Sampling response: {content}")
    if content is None:
        raise ValueError("LLM response content is None")
    sampling_response = SamplingResponse.model_validate_json(content)

    # Validate that we have n entries
    if len(sampling_response.root) < n:
        raise ValueError(f"Expected {n} samples, got {len(sampling_response.root)}")

    return sampling_response.root
