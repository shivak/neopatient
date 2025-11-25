from typing import List, TypedDict, Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, RootModel, Field, model_validator
import pyarrow as pa


class CodeSystem(str, Enum):
    """Enumeration of supported medical vocabulary systems."""

    SNOMED = "snomed"  # not in claims
    RXNORM = "rxnorm"  # not in claims
    ICD9_PROC = "icd9_proc"
    LNC = "lnc"  # not in claims
    ICD10_PROC = "icd10_proc"
    CPT = "cpt"
    ICD9 = "icd9"
    NDC = "ndc"  # not in ehr
    ICD10 = "icd10"
    # not in EHR/claims data
    # PHECODE = "phecode"
    # ATC = "atc"
    # UMLS_CUI = "umls_cui"


class RecordType(str, Enum):
    """Enumeration of supported record types."""

    EHR_INPATIENT = "ehr-inpatient"
    EHR_OUTPATIENT = "ehr-outpatient"
    CLAIMS = "claims"


class CohortSpec(BaseModel):
    """Specification for a cohort of patients."""

    positive: List[str] = Field(description="List of positive condition codes")
    negative: List[str] = Field(description="List of negative condition codes")
    count: int = Field(description="Number of patients to generate")
    record_type: RecordType = Field(description="Type of record")


class Event(BaseModel):
    """Individual patient event/measurement record (without time, as time is the dict key)."""

    code_system: CodeSystem = Field(description="vocabulary system for this code")
    code_desc: str = Field(
        description="brief textual description of the code/measurement/event"
    )
    numeric_value: float | None = Field(description="numeric result if applicable")
    unit: str | None = Field(description="unit for the numeric_value if applicable")
    text_value: str | None = Field(description="text result if applicable")


class UncodedPatient(RootModel[Dict[str, List[Event]]]):
    """Ordered dictionary of times (strings, '' for static) to lists of events."""

    pass


class GenerationResponse(BaseModel):
    """Response from generation LLM with finished flag."""

    finished: bool = Field(description="Whether the generation is complete")
    records: UncodedPatient = Field(description="The generated patient records")


# Type alias for a single patient's MEDS data table
type Patient = pa.Table

# Type alias for a cohort of patients
type Cohort = list[Patient]


class VerificationResponse(BaseModel):
    """Response from verification LLM."""

    satisfactory: bool
    criticism: str


class PatientRecipe(BaseModel):
    """Recipe for generating a patient record, including dates and description."""

    start_date: datetime
    end_date: datetime
    description: str

    @model_validator(mode="after")
    def validate_dates(self):
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


class State(TypedDict, total=False):
    stage: str
    sampled_descriptions: List[Dict[int, PatientRecipe]]
    generation_tickets: List[str]
    generated_records: List[List[UncodedPatient]]
    verification_tickets: List[str]
    verifications: List[List[VerificationResponse]]
    patient_ids: List[List[int]]
    coded_cohorts: List[Cohort]
