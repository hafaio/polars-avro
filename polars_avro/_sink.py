from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import partial
from os import path
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self

from polars import DataFrame, Schema

from ._avro_rs import AvroBuffSink, AvroCloudSink, AvroFileSink, Codec, py_is_cloud_url
from ._cloud import CredentialProviderInput, resolve_credentials


def create_writer(
    schema: Schema,
    *,
    dest: str | Path | BinaryIO,
    codec: Codec = Codec.Null,
    storage_options: Mapping[str, str] | None,
    credential_provider: CredentialProviderInput,
) -> AvroBuffSink | AvroFileSink | AvroCloudSink:
    fields = [*schema.items()]
    match dest:
        case str() | Path():
            expanded = path.expanduser(path.expandvars(dest))
            if py_is_cloud_url(expanded):
                options = resolve_credentials(
                    credential_provider, [expanded], storage_options
                )
                return AvroCloudSink(expanded, fields, codec, options)
            else:
                return AvroFileSink(expanded, fields, codec)
        case _:
            return AvroBuffSink(dest, fields, codec)


class AvroWriter:
    """Incrementally write DataFrames to an Avro file.

    This creates a context manager that needs to be used when writing cloud
    files.

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
        storage_options: Mapping[str, str] | None = None,
        credential_provider: CredentialProviderInput = "auto",
    ) -> None:
        self._create: Callable[
            [Schema], AvroBuffSink | AvroFileSink | AvroCloudSink
        ] = partial(
            create_writer,
            dest=dest,
            codec=codec,
            storage_options=storage_options,
            credential_provider=credential_provider,
        )
        self._sink: AvroBuffSink | AvroFileSink | AvroCloudSink | None = (
            None if schema is None else self._create(schema)
        )

    def __enter__(self) -> Self:
        return self

    def write(self, batch: DataFrame) -> None:
        if self._sink is None:
            self._sink = self._create(batch.schema)
        self._sink.write(batch)

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


def write_avro(  # noqa: PLR0913
    batches: DataFrame | Iterable[DataFrame],
    dest: str | Path | BinaryIO,
    *,
    schema: Schema | None = None,
    codec: Codec = Codec.Null,
    storage_options: Mapping[str, str] | None = None,
    credential_provider: CredentialProviderInput = "auto",
) -> None:
    """Write a DataFrame or iterable of DataFrames to an Avro file.

    Some polars types (Int8, Int16, UInt8, UInt16, UInt32, UInt64, Time,
    Categorical, Enum) must be cast before writing — see the README for
    workarounds.

    Parameters
    ----------
    batches : A DataFrame or iterable of DataFrames to write.
    dest : The file path, cloud URL, or writable binary buffer to write to.
    schema : The schema to use. If None, inferred from the first batch.
    codec : The compression codec to use.
    storage_options : Extra configuration passed to the cloud storage
        backend (same keys accepted by Polars, e.g. ``aws_region``).
    credential_provider : Credential provider for cloud storage. Set to
        ``"auto"`` (default) to use automatic credential detection, or
        ``None`` to disable.
    """
    with AvroWriter(
        dest,
        schema=schema,
        codec=codec,
        storage_options=storage_options,
        credential_provider=credential_provider,
    ) as writer:
        if isinstance(batches, DataFrame):
            writer.write(batches)
        else:
            for batch in batches:
                writer.write(batch)
