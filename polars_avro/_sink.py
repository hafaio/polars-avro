from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from os import path
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self

import polars as pl
import pyarrow as pa
from polars import DataFrame, Schema

from ._avro_rs import AvroBuffSink, AvroFileSink, Codec


def create_writer(
    schema: pa.Schema,
    *,
    dest: str | Path | BinaryIO,
    codec: Codec | None = None,
) -> AvroBuffSink | AvroFileSink:
    """Create a sink writing avro records matching ``schema``."""
    match dest:
        case str() | Path():
            expanded = path.expanduser(path.expandvars(dest))
            return AvroFileSink(expanded, schema, codec)
        case _:
            return AvroBuffSink(dest, schema, codec)


class AvroWriter:
    """Incrementally write DataFrames to an Avro file.

    Some polars types (Int8, Int16, UInt8, UInt16, UInt32, UInt64, Time,
    Categorical, Enum) must be cast before writing — see the README for
    workarounds.
    """

    def __init__(
        self,
        dest: str | Path | BinaryIO,
        *,
        schema: Schema | None = None,
        codec: Codec | None = None,
    ) -> None:
        self._create: Callable[[pa.Schema], AvroBuffSink | AvroFileSink] = partial(
            create_writer,
            dest=dest,
            codec=codec,
        )
        self._sink: AvroBuffSink | AvroFileSink | None = (
            None
            if schema is None
            else self._create(pl.DataFrame(schema=schema).to_arrow().schema)
        )

    def __enter__(self) -> Self:
        return self

    def write(self, batch: DataFrame) -> None:
        # to_arrow exports through pyarrow, which (unlike polars' own Arrow C
        # export) emits spec-compliant Null arrays; see
        # https://github.com/pola-rs/polars/issues/22934
        table = batch.to_arrow()
        if self._sink is None:
            self._sink = self._create(table.schema)
        for record_batch in table.to_batches():
            self._sink.write(record_batch)

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.close()


def write_avro(
    batches: DataFrame | Iterable[DataFrame],
    dest: str | Path | BinaryIO,
    *,
    schema: Schema | None = None,
    codec: Codec | None = None,
) -> None:
    """Write a DataFrame or iterable of DataFrames to an Avro file.

    Some polars types (Int8, Int16, UInt8, UInt16, UInt32, UInt64, Time,
    Categorical, Enum) must be cast before writing — see the README for
    workarounds.

    Parameters
    ----------
    batches : A DataFrame or iterable of DataFrames to write.
    dest : The file path or writable binary buffer to write to.
    schema : The schema to use. If None, inferred from the first batch.
    codec : The compression codec to use, or None for no compression.
    """
    with AvroWriter(
        dest,
        schema=schema,
        codec=codec,
    ) as writer:
        if isinstance(batches, DataFrame):
            writer.write(batches)
        else:
            for batch in batches:
                writer.write(batch)
