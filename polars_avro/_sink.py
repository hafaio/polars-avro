from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from os import path
from pathlib import Path
from typing import BinaryIO

from polars import DataFrame, Schema

from ._avro_rs import AvroBuffSink, AvroFileSink, Codec


def create_writer(
    schema: Schema,
    *,
    dest: str | Path | BinaryIO,
    codec: Codec = Codec.Null,
) -> AvroBuffSink | AvroFileSink:
    fields = [*schema.items()]
    match dest:
        case str() | Path():
            expanded = path.expanduser(path.expandvars(dest))
            return AvroFileSink(expanded, fields, codec)
        case _:
            return AvroBuffSink(dest, fields, codec)


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
        codec: Codec = Codec.Null,
    ) -> None:
        self._create: Callable[[Schema], AvroBuffSink | AvroFileSink] = partial(
            create_writer,
            dest=dest,
            codec=codec,
        )
        self._sink = None if schema is None else self._create(schema)

    def write(self, batch: DataFrame) -> None:
        if self._sink is None:
            self._sink = self._create(batch.schema)
        self._sink.write(batch)


def write_avro(  # noqa: PLR0913
    batches: DataFrame | Iterable[DataFrame],
    dest: str | Path | BinaryIO,
    *,
    schema: Schema | None = None,
    codec: Codec = Codec.Null,
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
    codec : The compression codec to use.
    """
    writer = AvroWriter(dest, schema=schema, codec=codec)
    if isinstance(batches, DataFrame):
        writer.write(batches)
    else:
        for batch in batches:
            writer.write(batch)
