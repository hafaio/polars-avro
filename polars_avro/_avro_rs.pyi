from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import BinaryIO

import pyarrow as pa

class Codec:
    """A compression codec to use when writing Avro files.

    Pass ``None`` instead of a codec for no compression (the avro ``null``
    codec).
    """

    Bzip2: Codec
    Deflate: Codec
    Snappy: Codec
    Xz: Codec
    Zstandard: Codec

class AvroIter:
    """An iterator over the record batches of an avro source."""

    def __iter__(self) -> AvroIter: ...
    def __next__(self) -> pa.RecordBatch: ...

class AvroSource:
    """A pseudo-iterator over Avro files.

    ``paths`` are local files read natively. ``sources`` are factories, each
    returning a fresh single-use context manager whose ``__enter__`` yields a
    seekable binary file and whose ``__exit__`` releases it; the reader calls a
    factory once per scan (and re-call to rewind).
    """

    def __init__(
        self,
        paths: list[str],
        sources: list[Callable[[], AbstractContextManager[BinaryIO]]],
    ) -> None: ...
    def schema(self) -> pa.Schema: ...
    def batch_iter(
        self,
        strict: bool,
        utf8_view: bool,
        batch_size: int,
        with_columns: list[str] | None,
    ) -> AvroIter: ...

class AvroFileSink:
    """A sink that writes record batches to a file."""

    def __init__(self, path: str, schema: pa.Schema, codec: Codec | None) -> None: ...
    def write(self, batch: pa.RecordBatch) -> None: ...
    def close(self) -> None: ...

class AvroBuffSink:
    """A sink that writes record batches to a writable binary buffer."""

    def __init__(
        self, buff: BinaryIO, schema: pa.Schema, codec: Codec | None
    ) -> None: ...
    def write(self, batch: pa.RecordBatch) -> None: ...
    def close(self) -> None: ...

class AvroError(Exception):
    """An exception thrown from the native avro reader and writer."""

class EmptySources(ValueError):
    """An exception for when no sources are given."""

class AvroSpecError(ValueError):
    """An exception raised when data doesn't align to the avro spec."""
