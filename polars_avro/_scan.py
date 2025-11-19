from collections.abc import Iterator, Sequence
from glob import iglob
from os import path
from pathlib import Path
from typing import BinaryIO

import polars as pl
from polars import DataFrame, Expr, LazyFrame
from polars.io.plugins import register_io_source

from ._avro_rs import AvroSource


def expand_str(source: str | Path, *, glob: bool) -> Iterator[str]:
    """Expand a string or Path to a list of file paths."""
    expanded = path.expanduser(source)
    if glob and "*" in expanded:
        yield from sorted(iglob(expanded))
    elif path.isdir(expanded):
        yield from sorted(iglob(path.join(expanded, "*")))
    else:
        yield expanded


def scan_avro(  # noqa: PLR0913
    sources: Sequence[str | Path] | Sequence[BinaryIO] | str | Path | BinaryIO,
    *,
    batch_size: int = 32768,
    glob: bool = True,
    single_col_name: str | None = None,
) -> LazyFrame:
    """Scan Avro files.

    Parameters
    ----------
    sources : The source(s) to scan.
    batch_size : How many rows to attempt to read at a time.
    glob : Whether to use globbing to find files.
    storage_options : Additional options for cloud operations.
    """
    # normalize sources
    strs: list[str] = []
    bins: list[BinaryIO] = []
    match sources:
        case [*_]:
            for source in sources:
                if isinstance(source, str | Path):
                    strs.extend(expand_str(source, glob=glob))
                else:
                    bins.append(source)
        case str() | Path():
            strs.extend(expand_str(sources, glob=glob))
        case _:
            bins.append(sources)

    # normalize cloud options
    def_batch_size = batch_size

    src = AvroSource(
        strs,
        bins,
        single_col_name,
    )

    def get_schema() -> pl.Schema:
        return pl.Schema(src.schema())

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

    try:
        return register_io_source(source_generator, schema=get_schema)
    except TypeError:  # pragma: no cover
        eager_schema = get_schema()
        return register_io_source(source_generator, schema=eager_schema)


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
    single_col_name: str | None = None,
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
        single_col_name=single_col_name,
    )
    if columns is not None:
        lazy = lazy.select(  # type: ignore
            [pl.nth(c) if isinstance(c, int) else pl.col(c) for c in columns]
        )
    if row_index_name is not None:
        lazy = lazy.with_row_index(row_index_name, offset=row_index_offset)
    if n_rows is not None:
        lazy = lazy.limit(n_rows)
    res = lazy.collect()
    return res.rechunk() if rechunk else res
