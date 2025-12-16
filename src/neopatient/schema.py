import pyarrow as pa
from meds import DataSchema
from flexible_schema import Optional


class PatientSchema(DataSchema):
    """Extends DataSchema for EHRSHOT-style units and inline code descriptions."""

    unit: Optional(pa.string())
    code_descr: Optional(pa.large_string())
