from typing import List, Union
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


# For MEDS generation
# GenerationRecord: List of tuples where each tuple is (time, code_system, code_desc, numeric_value, text_value)
# - time: str or None, ISO timestamp or null for static measurements
# - code_system: CodeSystem, the vocabulary system for this code (non-null)
# - code_desc: str, brief textual description of the code/measurement/event
# - numeric_value: float or None, numeric result associated with this measurement
# - text_value: str or None, text result or unit
GenerationRecord = List[tuple[Union[str, None], CodeSystem, str, Union[float, None], Union[str, None]]]
