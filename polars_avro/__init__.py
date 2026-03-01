"""Polars io plugin for reading and writing Apache Avro files.

Provides `scan_avro`, `read_avro`, `write_avro`, and `AvroWriter`. Some polars
types (Int8, Int16, UInt8, UInt16, UInt32, UInt64, Time, Categorical, Enum)
must be cast before writing. When reading, the ``utf8_view`` option controls
how UUIDs and nullable strings are decoded — see `scan_avro` for details.
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
