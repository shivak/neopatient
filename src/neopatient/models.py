from typing import List, Union, TypedDict, Dict, Any
from enum import Enum
import pathlib
from pydantic import BaseModel, RootModel, Field
import pyarrow as pa


class CodeSystem(str, Enum):
    """Enumeration of supported medical vocabulary systems."""

    SNOMED = "snomed" # not in claims
    RXNORM = "rxnorm" # not in claims
    ICD9_PROC = "icd9_proc"
    LNC = "lnc" # not in claims
    ICD10_PROC = "icd10_proc"
    CPT = "cpt"
    ICD9 = "icd9"
    NDC = "ndc" # not in ehr
    ICD10 = "icd10"
    # not in EHR/claims data
    # PHECODE = "phecode"
    # ATC = "atc"
    # UMLS_CUI = "umls_cui"


class Event(BaseModel):
    """Individual patient event/measurement record."""
    time: str | None = Field(description="ISO timestamp or null for static measurements")
    code_system: CodeSystem = Field(description="vocabulary system for this code")
    code_desc: str = Field(description="brief textual description of the code/measurement/event")
    numeric_value: float | None = Field(description="numeric result if applicable")
    unit: str | None = Field(description="unit for the numeric_value if applicable")
    text_value: str | None = Field(description="text result if applicable")


class UncodedPatient(RootModel[List[Event]]):
    """List of uncoded patient events before code matching."""
    pass


# Type alias for a single patient's MEDS data table
type Patient = pa.Table

# Type alias for a cohort of patients
type Cohort = list[Patient]


class VerificationResponse(BaseModel):
    """Response from verification LLM."""
    satisfactory: bool
    criticism: str


class State(TypedDict, total=False):
    stage: str
    cohort_specs: List[Dict[str, Any]]
    chroma_db: Any
    epsilon: float
    generator: str
    verifier: str
    sampler: str
    sampled_descriptions: List[Dict[int, str]]
    generation_tickets: List[str]
    generated_records: List[List[UncodedPatient]]
    verification_tickets: List[str]
    verifications: List[List[VerificationResponse]]
    patient_ids: List[List[int]]
    coded_cohorts: List[Cohort]
