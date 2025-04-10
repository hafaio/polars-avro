"""Scan and sink avro files using Polars.

This module provides three functions: `scan_avro`, `write_avro`, and `read_avro`
written as an io-plugin. Currently `sink` is not supported for io-plugins.

Not all types can be converted between Polars and Avro, or are supported but
with precision loss. In general this library will attempt to read and write
everything as long as there's no data loss, but that can mean promoting types to
make them more general for serialization / deserialization.
"""

from ._avro_rs import AvroError, AvroSpecError, Codec, EmptySources
from ._scan import read_avro, scan_avro
from ._sink import write_avro

__all__ = (
    "AvroError",
    "AvroSpecError",
    "Codec",
    "EmptySources",
    "read_avro",
    "scan_avro",
    "write_avro",
)
