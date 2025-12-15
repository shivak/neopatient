from __future__ import annotations

from typing import List, Dict, Union, Annotated
from enum import Enum
from datetime import datetime
from pydantic import (
    BaseModel,
    RootModel,
    Field,
    model_validator,
    computed_field,
    PlainSerializer,
    PlainValidator,
)
import pyarrow as pa
import pyarrow.parquet as pq
import base64
from meds.schema import DataSchema


class CodeSystem(str, Enum):
    """Enumeration of supported medical vocabulary systems."""

    SNOMED = "snomed"
    RXNORM = "rxnorm"
    ICD9_PROC = "icd9_proc"
    LOINC = "loinc"
    # LNC = "lnc"  # just the underlying "parts" of loinc codes
    ICD10_PROC = "icd10_proc"
    CPT = "cpt"
    ICD9 = "icd9"
    NDC = "ndc"
    ICD10 = "icd10"
    # not in EHR/claims data
    # PHECODE = "phecode"
    # ATC = "atc"
    # UMLS_CUI = "umls_cui"

    @staticmethod
    def allowed_in(record_type: "RecordType") -> List["CodeSystem"]:
        """Return list of allowed code systems for the given record type."""
        if record_type == RecordType.CLAIMS:
            return [
                CodeSystem.ICD9_PROC,
                CodeSystem.ICD10_PROC,
                CodeSystem.CPT,
                CodeSystem.ICD9,
                CodeSystem.NDC,
                CodeSystem.ICD10,
            ]
        elif record_type in [RecordType.EHR_INPATIENT, RecordType.EHR_OUTPATIENT]:
            return [
                CodeSystem.SNOMED,
                CodeSystem.RXNORM,
                CodeSystem.ICD9_PROC,
                CodeSystem.LOINC,
                CodeSystem.ICD10_PROC,
                CodeSystem.CPT,
                CodeSystem.ICD9,
                CodeSystem.ICD10,
            ]
        else:
            raise ValueError(f"Unknown record type: {record_type}")


class RecordType(str, Enum):
    """Enumeration of supported record types."""

    EHR_INPATIENT = "ehr-inpatient"
    EHR_OUTPATIENT = "ehr-outpatient"
    CLAIMS = "claims"


class Stage(str, Enum):
    """Enumeration of synthesis stages."""

    SAMPLING = "sampling"
    CHECK_SAMPLING = "check_sampling"
    GENERATION = "generation"
    CHECK_GENERATION = "check_generation"
    MATCHING = "matching"
    VERIFICATION = "verification"
    CHECK_VERIFICATION = "check_verification"
    FINALIZE = "finalize"


class CohortSpec(BaseModel):
    """Specification for a cohort of patients."""

    positive: str = Field(description="Positive cohort description")
    negative: str = Field(description="Negative cohort description")
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


class UncodedPatient(RootModel[Dict[str, List[Event]]]):
    """Ordered dictionary of times (strings, '' for static) to lists of events."""

    pass


# FlatEvent is a union to represent events without nulls: 2-tuple for non-numeric events,
# 4-tuple for numeric events (ensuring unit is always present with numeric_value).
FlatEvent = Union[tuple[CodeSystem, str], tuple[CodeSystem, str, float, str]]


class FlatUncodedPatient(RootModel[Dict[str, List[FlatEvent]]]):
    """Flat representation of uncoded patient records (events as tuples)."""

    def unflatten(self) -> UncodedPatient:
        """Convert flat tuple-based events to structured Event objects."""
        records = {}
        for time, events in self.root.items():
            event_list = []
            for e in events:
                if len(e) == 2:
                    code_system, code_desc = e
                    numeric_value = None
                    unit = None
                elif len(e) == 4:
                    code_system, code_desc, numeric_value, unit = e
                else:
                    raise ValueError(f"Invalid event tuple length: {len(e)}")
                event_list.append(
                    Event(
                        code_system=code_system,
                        code_desc=code_desc,
                        numeric_value=numeric_value,
                        unit=unit,
                    )
                )
            records[time] = event_list
        return UncodedPatient(records)


class GenerationResponse(BaseModel):
    """Response from generation LLM (events as tuples)."""

    records: FlatUncodedPatient = Field(
        description="The generated patient records (flat)"
    )


def _serialize_table(table: pa.Table, compression: str = "zstd") -> str:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=compression)
    buf = sink.getvalue()
    return base64.b64encode(buf.to_pybytes()).decode("utf-8")


def _deserialize_table(b64_str: str) -> pa.Table:
    buf = pa.py_buffer(base64.b64decode(b64_str))
    return pq.read_table(buf)


def _serialize_patient_table(t: pa.Table) -> str:
    return _serialize_table(t, compression=None)


def _validate_patient_table(table) -> pa.Table:
    if isinstance(table, str):
        table = _deserialize_table(table)
    DataSchema.validate(table)
    return table


# Type alias for a single patient's MEDS data table
Patient = Annotated[
    pa.Table,
    PlainValidator(_validate_patient_table),
    PlainSerializer(_serialize_patient_table),
]

# Type alias for a cohort of patients
type Cohort = list[Patient]


class VerificationResponse(BaseModel):
    """Response from verification LLM."""

    satisfactory: bool
    criticism: str


class Segment(BaseModel):
    """A temporal segment of a patient's medical history."""

    start_date: datetime
    end_date: datetime
    num_times: int
    avg_codes_per_time: float
    description: str

    @model_validator(mode="after")
    def validate_dates(self):
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


class PatientRecipe(BaseModel):
    """Recipe for generating a patient record, including dates, description, and temporal segments."""

    birthday: datetime
    description: str
    gender: str | None = None
    race: str | None = None
    ethnicity: str | None = None
    segments: List[Segment]

    @computed_field
    def start_date(self) -> datetime:
        """Overall start date derived from earliest segment."""
        return min(segment.start_date for segment in self.segments)

    @computed_field
    def end_date(self) -> datetime:
        """Overall end date derived from latest segment."""
        return max(segment.end_date for segment in self.segments)

    @computed_field
    def total_codes(self) -> int:
        """Total codes derived from segments."""
        return sum(
            int(segment.num_times * segment.avg_codes_per_time)
            for segment in self.segments
        )

    @computed_field
    def num_times(self) -> int:
        """Total number of times derived from segments."""
        return sum(segment.num_times for segment in self.segments)

    @computed_field
    def avg_codes_per_time(self) -> float:
        """Average codes per time derived from segments."""
        total_times = sum(segment.num_times for segment in self.segments)
        if total_times == 0:
            return 0.0
        total_codes = sum(
            int(segment.num_times * segment.avg_codes_per_time)
            for segment in self.segments
        )
        return total_codes / total_times


class SamplingResponse(RootModel[Dict[int, PatientRecipe]]):
    """Response from sampling LLM with patient recipes."""

    pass


class State(BaseModel):
    stage: Stage
    sampled_recipes: list[dict[int, PatientRecipe]] = Field(default_factory=list)
    sampling_batch_id: str | None = None
    generation_batch_id: str | None = None
    generated_records: list[dict[int, UncodedPatient]] = Field(default_factory=list)
    verification_batch_id: str | None = None
    verifications: list[list[VerificationResponse]] = Field(default_factory=list)
    coded_cohorts: list[Cohort] = Field(default_factory=list)
