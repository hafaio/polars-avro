from __future__ import annotations

from typing import BinaryIO

from polars import DataFrame
from polars._typing import SchemaDict  # type: ignore[reportPrivateImportUsage]

class Codec:
    """The codec to use when writing Avro files."""

    Null: Codec
    Bzip2: Codec
    Deflate: Codec
    Snappy: Codec
    Xz: Codec
    Zstandard: Codec

class AvroIter:
    def next(self) -> DataFrame | None: ...

class AvroSource:
    """A pseudo-iterator over Avro files."""

    def __init__(
        self,
        paths: list[str],
        buffs: list[BinaryIO],
        single_col_name: str | None,
    ) -> None: ...
    def schema(self) -> SchemaDict: ...
    def batch_iter(
        self, batch_size: int, with_columns: list[str] | None
    ) -> AvroIter: ...

def write_avro_file(
    frames: list[DataFrame],
    dest: str,
    codec: Codec,
    promote_ints: bool,
    promote_array: bool,
    truncate_time: bool,
    compression_level: int | None,
    cloud_options: list[tuple[str, str]] | None,
) -> None:
    """Write a DataFrame to an Avro file."""
    ...

def write_avro_buff(
    frames: list[DataFrame],
    buff: BinaryIO,
    codec: Codec,
    promote_ints: bool,
    promote_array: bool,
    truncate_time: bool,
    compression_level: int | None,
) -> None:
    """Write a DataFrame to bytes."""
    ...

class AvroError(Exception):
    """An exception thrown from the native avro reader and writer."""

class EmptySources(ValueError):
    """An exception for when no sources are given."""

class AvroSpecError(ValueError):
    """An exception raised when the spec doesn't align to the avro spec."""
