from __future__ import annotations

from typing import BinaryIO

from polars import DataFrame, DataType
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
        storage_options: list[tuple[str, str]],
    ) -> None: ...
    def schema(self) -> SchemaDict: ...
    def batch_iter(
        self,
        strict: bool,
        utf8_view: bool,
        batch_size: int,
        with_columns: list[str] | None,
    ) -> AvroIter: ...

class AvroFileSink:
    """A sink to a file."""

    def __init__(
        self,
        path: str,
        fields: list[tuple[str, DataType]],
        codec: Codec,
    ) -> None: ...
    def write(self, frame: DataFrame) -> None: ...
    def close(self) -> None: ...

class AvroBuffSink:
    """A sink to a buffer."""

    def __init__(
        self,
        buff: BinaryIO,
        fields: list[tuple[str, DataType]],
        codec: Codec,
    ) -> None: ...
    def write(self, frame: DataFrame) -> None: ...
    def close(self) -> None: ...

class AvroCloudSink:
    """A sink to cloud storage."""

    def __init__(
        self,
        url: str,
        fields: list[tuple[str, DataType]],
        codec: Codec,
        storage_options: list[tuple[str, str]],
    ) -> None: ...
    def write(self, frame: DataFrame) -> None: ...
    def close(self) -> None: ...

def py_is_cloud_url(url: str) -> bool: ...

class AvroError(Exception):
    """An exception thrown from the native avro reader and writer."""

class EmptySources(ValueError):
    """An exception for when no sources are given."""

class AvroSpecError(ValueError):
    """An exception raised when the spec doesn't align to the avro spec."""
