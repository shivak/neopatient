from pydantic import BaseModel, Field
from typing import List, Union
from datetime import datetime
from enum import Enum


class CodeSystem(str, Enum):
    """Enumeration of supported medical vocabulary systems."""

    SNOMED = "snomed"
    RXNORM = "rxnorm"
    ICD9_PROC = "icd9_proc"
    PHECODE = "phecode"
    LNC = "lnc"
    ICD10_PROC = "icd10_proc"
    CPT = "cpt"
    ATC = "atc"
    ICD9 = "icd9"
    UMLS_CUI = "umls_cui"
    NDC = "ndc"
    ICD10 = "icd10"


class StaticMeasurement(BaseModel):
    """Static measurement for gender, race, and ethnicity."""

    code_system: CodeSystem = Field(description="The vocabulary system for this code")
    code_desc: str = Field(
        description="Brief textual description of the code/measurement/event"
    )


class EventMeasurement(BaseModel):
    """Measurement within a medical event."""

    code_system: CodeSystem = Field(description="The vocabulary system for this code")
    code_desc: str = Field(
        description="Brief textual description of the code/measurement/event"
    )
    result_num: float | None = Field(
        None,
        description="Numeric result associated with this measurement (e.g., laboratory test result)",
    )


unit: str | None = Field(
    None,
    description="Units of the numerical measurement (e.g. mg/dL, /HPF, cm H2O, mU/L, pmol/L, Sec., mm Hg, lbs)",
)


class MedicalEvent(BaseModel):
    """A medical event with timestamp and associated measurements."""

    time: datetime = Field(description="ISO 8601 timestamp when this event occurred")
    measurements: List[EventMeasurement] = Field(
        description="List of all codes recorded during this event"
    )


class PatientRecord(BaseModel):
    """A synthetic longitudinal patient record."""

    patient_id: int = Field(description="Unique patient identifier")
    static_measurements: List[StaticMeasurement] = Field(
        description="Static measurements for gender, race, and ethnicity"
    )
    events: List[MedicalEvent] = Field(
        description="List of medical events with their associated measurements"
    )


# For MEDS generation
# GenerationRecord: List of tuples where each tuple is (time, code_system, code_desc, numeric_value, text_value)
# - time: str or None, ISO timestamp or null for static measurements
# - code_system: CodeSystem, the vocabulary system for this code (non-null)
# - code_desc: str, brief textual description of the code/measurement/event
# - numeric_value: float or None, numeric result associated with this measurement
# - text_value: str or None, text result or unit
GenerationRecord = List[tuple[Union[str, None], CodeSystem, str, Union[float, None], Union[str, None]]]

# Import meds.DataSchema
from meds.schema import DataSchema
