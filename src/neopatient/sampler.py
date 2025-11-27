import json
import os
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
import jinja2
import datetime
import pandas as pd
from .models import PatientRecipe

# Get the directory of the current file to construct template path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
_sample_template_path = os.path.join(_project_root, "templates", "sample.jinja2")

SAMPLE_TEMPLATE = jinja2.Template(
    open(_sample_template_path, "r", encoding="utf-8").read()
)


def _get_csv_path(record_type: str) -> str:
    if record_type == "ehr-inpatient":
        return os.path.join(_project_root, "stats", "ehr-inpatient.csv")
    else:
        return os.path.join(_project_root, "stats", "ehr-outpatient.csv")


def sample_patient_stats(csv_path: str, n: int) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    sampled_df = df.sample(n=n, replace=True)
    return sampled_df.to_dict("records")


def sample_recipe(
    positive: str, record_type: str, end_date: Optional[str] = None
) -> PatientRecipe:
    if end_date is None:
        end_date = datetime.datetime.now().isoformat()
    csv_path = _get_csv_path(record_type)
    stat = sample_patient_stats(csv_path, 1)[0]
    duration = stat["duration"]
    days = int(duration.split()[0])
    start_date = (
        datetime.datetime.fromisoformat(end_date) - datetime.timedelta(days=days)
    ).isoformat()
    if record_type in ["claims", "ehr-outpatient"]:
        start_date = start_date.split("T")[0]
        end_date = end_date.split("T")[0]
    return PatientRecipe(
        description=positive,
        start_date=datetime.datetime.fromisoformat(start_date),
        end_date=datetime.datetime.fromisoformat(end_date),
        total_codes=stat["total_codes"],
        unique_codes=stat["unique_codes"],
        duration=stat["duration"],
        num_times=stat["num_times"],
        avg_codes_per_time=stat["avg_codes_per_time"],
    )


async def sample_recipes(
    positive: str, negative: str, n: int, record_type: str, sampler_model: str = "gpt-5"
) -> Dict[int, PatientRecipe]:
    """
    Samples individual patient recipes that satisfy cohort criteria.

    Uses LLM to generate n self-contained recipes, each including start_date, end_date, and description,
    constrained by the sampled duration.

    Args:
        positive: Positive cohort description
        negative: Negative anti-cohort description
        n: Number of patients to sample
        record_type: Type of record ("claims", "ehr-inpatient", "ehr-outpatient")
        sampler_model: Model name for sampling (default: "gpt-5")

    Returns:
        Dict of {patient_id: PatientRecipe}
    """
    client = AsyncOpenAI()  # Assume API key is set via environment

    csv_path = _get_csv_path(record_type)
    stats = sample_patient_stats(csv_path, n)

    prompt = SAMPLE_TEMPLATE.render(
        positive_cohort=positive, negative_cohort=negative, n=n, stats=stats
    )
    response = await client.chat.completions.create(
        model=sampler_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "patient_recipes",
                "strict": True,
                "schema": PatientRecipe.model_json_schema(),
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
