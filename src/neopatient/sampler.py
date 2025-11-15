import json
import os
from typing import Dict
import openai
import jinja2

# Get the directory of the current file to construct template path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_sample_template_path = os.path.join(_project_root, "templates", "sample.jinja2")

SAMPLE_TEMPLATE = jinja2.Template(open(_sample_template_path, "r", encoding="utf-8").read())


def sample_individual_descriptions(
    positive: str,
    negative: str,
    n: int,
    sampler_model: str = "gpt-4o"
) -> Dict[int, str]:
    """
    Samples individual patient descriptions that satisfy cohort criteria.

    Uses LLM to generate n self-contained descriptions, each including what the individual
    should and should not be like, along with unique patient IDs.

    Args:
        positive: Positive cohort description
        negative: Negative anti-cohort description
        n: Number of patients to sample
        sampler_model: Model name for sampling (default: "gpt-4o")

    Returns:
        Dict of {patient_id: individual_description}
    """
    client = openai.OpenAI()  # Assume API key is set via environment

    prompt = SAMPLE_TEMPLATE.render(positive_cohort=positive, negative_cohort=negative, n=n)
    response = client.chat.completions.create(
        model=sampler_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM response content is None")
    sampled_dict = json.loads(content)

    # Validate that we have n entries
    if len(sampled_dict) != n:
        raise ValueError(f"Expected {n} samples, got {len(sampled_dict)}")

    return sampled_dict