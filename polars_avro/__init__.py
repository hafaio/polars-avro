"""Polars io plugin for reading and writing Apache Avro files.

Provides `scan_avro`, `read_avro`, `write_avro`, and `AvroWriter`. Most polars
types write directly (narrow ints widen, Time truncates to microseconds); only
Categorical, Enum, and out-of-range UInt64 values must be cast first. When
reading, the ``utf8_view`` option controls how UUIDs and nullable strings are
decoded — see `scan_avro` for details.
"""

from ._avro_rs import AvroError, AvroSpecError, Codec, EmptySources
from ._scan import read_avro, scan_avro
from ._sink import AvroWriter, write_avro

__all__ = (
    "AvroError",
    "AvroSpecError",
    "AvroWriter",
    "Codec",
    "EmptySources",
    "read_avro",
    "scan_avro",
    "write_avro",
)
