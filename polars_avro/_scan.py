from collections.abc import Iterator, Mapping, Sequence
from os import path
from pathlib import Path
from typing import BinaryIO, Literal, cast

import polars as pl
from polars import CredentialProviderFunction, DataFrame, Expr, LazyFrame
from polars.io.cloud.credential_provider._builder import (
    _init_credential_provider_builder,  # type: ignore[reportPrivateImportUsage]
)
from polars.io.plugins import register_io_source

from ._avro_rs import AvroSource


def scan_avro(  # noqa: PLR0913
    sources: Sequence[str | Path] | Sequence[BinaryIO] | str | Path | BinaryIO,
    *,
    batch_size: int = 32768,
    glob: bool = True,
    storage_options: Mapping[str, str] | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
    retries: int = 2,
    file_cache_ttl: int | None = None,
) -> LazyFrame:
    """Scan Avro files.

    Parameters
    ----------
    sources : The source(s) to scan.
    batch_size : How many rows to attempt to read at a time.
    glob : Whether to use globbing to find files.
    storage_options : Additional options for cloud operations.
    credential_provider : The credential provider to use for cloud operations.
        Defaults to "auto" which uses the default credential provider.
    retries : The number of times to retry cloud operations.
    file_cache_ttl : The time to live for cached cloud files.
    """
    # normalize sources
    match sources:
        case [*_]:
            strs: list[str] = []
            bins: list[BinaryIO] = []
            for source in sources:
                if isinstance(source, str | Path):
                    strs.append(path.expanduser(source))
                else:
                    bins.append(source)
            normed = strs or bins
        case str() | Path():
            normed = [path.expanduser(sources)]
        case _:
            normed = [sources]

    # normalize cloud options
    cloud_options = None if storage_options is None else [*storage_options.items()]
    credential_provider_builder = _init_credential_provider_builder(
        credential_provider,
        sources,
        cast(dict[str, str], storage_options),  # incorrect signature
        "scan_avro",
    )
    def_batch_size = batch_size

    src = AvroSource(
        normed,
        glob,
        cloud_options,
        credential_provider_builder,
        retries,
        file_cache_ttl,
    )

    def source_generator(
        with_columns: list[str] | None,
        predicate: Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[DataFrame]:
        avro_iter = src.batch_iter(batch_size or def_batch_size, with_columns)
        while (batch := avro_iter.next()) is not None:
            if predicate is not None:
                batch = batch.filter(predicate)  # type: ignore
            if n_rows is None:
                yield batch
            else:
                batch = batch[:n_rows]
                n_rows -= len(batch)
                yield batch
                if n_rows == 0:
                    break

    return register_io_source(source_generator, schema=lambda: pl.Schema(src.schema()))


def read_avro(  # noqa: PLR0913
    sources: Sequence[str | Path] | Sequence[BinaryIO] | str | Path | BinaryIO,
    *,
    columns: Sequence[int | str] | None = None,
    n_rows: int | None = None,
    row_index_name: str | None = None,
    row_index_offset: int = 0,
    rechunk: bool = False,
    batch_size: int = 32768,
    glob: bool = True,
    storage_options: Mapping[str, str] | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
    retries: int = 2,
    file_cache_ttl: int | None = None,
) -> DataFrame:
    """Read an Avro file into a DataFrame.

    Parameters
    ----------
    sources : The source(s) to scan.
    columns : The columns to select.
    n_rows : The number of rows to read.
    row_index_name : The name of the row index column, or None to not add one.
    row_index_offset : The offset to start the row index at.
    rechunk : Whether to rechunk the DataFrame after reading.
    batch_size : How many rows to attempt to read at a time.
    glob : Whether to use globbing to find files.
    storage_options : Additional options for cloud operations.
    credential_provider : The credential provider to use for cloud operations.
        Defaults to "auto" which uses the default credential provider.
    retries : The number of times to retry cloud operations.
    file_cache_ttl : The time to live for cached cloud files.
    """
    lazy = scan_avro(
        sources,
        batch_size=batch_size,
        glob=glob,
        storage_options=storage_options,
        credential_provider=credential_provider,
        retries=retries,
        file_cache_ttl=file_cache_ttl,
    )
    if columns is not None:
        lazy = lazy.select(
            [pl.nth(c) if isinstance(c, int) else pl.col(c) for c in columns]
        )
    if row_index_name is not None:
        lazy = lazy.with_row_index(row_index_name, offset=row_index_offset)
    if n_rows is not None:
        lazy = lazy.limit(n_rows)
    res = lazy.collect()
    return res.rechunk() if rechunk else res
