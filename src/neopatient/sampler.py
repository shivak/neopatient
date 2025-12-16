import logging
import os
import random
from typing import Any, Optional, Union, List
from openai import AsyncOpenAI
import jinja2
import pandas as pd
from .models import (
    PatientRecipe,
    SamplingResponse,
    RecordType,
    State,
    CohortSpec,
    Cohort,
    Stage,
)
from .batch_llm import BatchLLM

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


def sample_patient_stats(csv_path: str, n: int) -> list[dict[str, Any]]:
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
) -> list[PatientRecipe]:
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
        List of PatientRecipe objects
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


async def _handle_sampling_stage(
    batch_llm: BatchLLM,
    state: State,
    cohort_specs: list[CohortSpec],
    chroma_db,
    embedder,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Sample individual patient recipes for each cohort using batch processing."""
    # Create batch sampling requests for all cohorts
    prompts_by_id = {}
    cohort_info = []  # Track (cohort_idx, expected_count) for validation

    for cohort_idx, spec in enumerate(cohort_specs):
        csv_path = _get_csv_path(spec.record_type)
        stats = sample_patient_stats(csv_path, spec.count)

        prompt = SAMPLE_TEMPLATE.render(
            positive_cohort=spec.positive,
            negative_cohort=spec.negative,
            n=spec.count,
            stats=stats,
        )

        custom_id = f"cohort_{cohort_idx}_sampling"
        prompts_by_id[custom_id] = prompt
        cohort_info.append((cohort_idx, spec.count))

        logger.info(f"Sampling prompt for cohort {cohort_idx}: {prompt[:100]}...")

    # Pre-generate unique patient IDs for all cohorts
    total_patients = sum(spec.count for spec in cohort_specs)
    all_ids = random.sample(range(1, 1000000), total_patients)
    sampled_ids = []
    start = 0
    for spec in cohort_specs:
        sampled_ids.append(all_ids[start : start + spec.count])
        start += spec.count
    state.sampled_ids = sampled_ids

    # Submit batch sampling request
    schema = SamplingResponse.model_json_schema()
    batch_id = await batch_llm.ask(prompts_by_id, schema)
    state.sampling_batch_id = batch_id
    state.stage = Stage.CHECK_SAMPLING

    return state


async def _handle_check_sampling_stage(
    batch_llm: BatchLLM,
    state: State,
    cohort_specs: list[CohortSpec],
    chroma_db,
    embedder,
    logger: logging.Logger,
) -> Union[List[Cohort], State]:
    """Check if sampling batch is ready and parse results."""
    if not state.sampling_batch_id:
        raise ValueError("No sampling batch ID found in state")

    batch_id = state.sampling_batch_id
    results = await batch_llm.get(batch_id)
    if results is None:
        return state

    sampled_recipes = {}
    for custom_id, content in results.items():
        # Extract cohort index from custom_id like "cohort_0_sampling"
        cohort_idx = int(custom_id.split("_")[1])

        sampling_response = SamplingResponse.model_validate_json(content)

        # Validate we got the expected number of recipes
        expected_count = cohort_specs[cohort_idx].count
        if len(sampling_response.root) < expected_count:
            raise ValueError(
                f"Cohort {cohort_idx}: Expected {expected_count} samples, got {len(sampling_response.root)}"
            )

        cohort_ids = state.sampled_ids[cohort_idx]
        cohort_recipes = dict(zip(cohort_ids, sampling_response.root))
        sampled_recipes.update(cohort_recipes)

    # Check that all cohorts have results
    if len(results) != len(cohort_specs):
        raise ValueError(
            f"Expected {len(cohort_specs)} cohort results, got {len(results)}"
        )

    state.sampled_recipes = sampled_recipes
    state.stage = Stage.GENERATION
    return state
