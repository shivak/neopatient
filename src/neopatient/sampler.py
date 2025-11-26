import json
import os
from typing import Dict
from openai import AsyncOpenAI
import jinja2
from .models import PatientRecipe

# Get the directory of the current file to construct template path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_sample_template_path = os.path.join(_project_root, "templates", "sample.jinja2")

SAMPLE_TEMPLATE = jinja2.Template(
    open(_sample_template_path, "r", encoding="utf-8").read()
)


async def sample_individual_descriptions(
    positive: str, negative: str, n: int, duration: str, sampler_model: str = "gpt-5"
) -> Dict[int, PatientRecipe]:
    """
    Samples individual patient recipes that satisfy cohort criteria.

    Uses LLM to generate n self-contained recipes, each including start_date, end_date, and description,
    constrained by the provided duration.

    Args:
        positive: Positive cohort description
        negative: Negative anti-cohort description
        n: Number of patients to sample
        duration: Approximate duration for each patient's record (e.g., "1000 days")
        sampler_model: Model name for sampling (default: "gpt-5")

    Returns:
        Dict of {patient_id: PatientRecipe}
    """
    client = AsyncOpenAI()  # Assume API key is set via environment

    prompt = SAMPLE_TEMPLATE.render(
        positive_cohort=positive, negative_cohort=negative, n=n, duration=duration
    )
    response = await client.chat.completions.create(
        model=sampler_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "patient_recipes",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": PatientRecipe.model_json_schema(),
                },
            },
        },
        temperature=0.7,
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM response content is None")
    sampled_dict = json.loads(content)

    # Validate that we have n entries
    if len(sampled_dict) != n:
        raise ValueError(f"Expected {n} samples, got {len(sampled_dict)}")

    # Parse to PatientRecipe
    result = {}
    for key, value in sampled_dict.items():
        result[int(key)] = PatientRecipe.model_validate(value)

    return result
