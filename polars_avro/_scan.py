from collections.abc import Iterator, Sequence
from glob import iglob
from os import path
from pathlib import Path
from typing import BinaryIO

import polars as pl
import pyarrow as pa
from polars import DataFrame, Expr, LazyFrame
from polars.io.plugins import register_io_source

from ._avro_rs import AvroSource


def _arrow_to_frame(data: pa.Table | pa.RecordBatch) -> DataFrame:
    """Import a pyarrow table or record batch as a DataFrame."""
    frame = pl.from_arrow(data)
    assert isinstance(frame, DataFrame)
    return frame


def expand_str(source: str | Path, *, glob: bool) -> Iterator[str]:
    """Expand a string or Path to a list of file paths."""
    expanded = path.expanduser(path.expandvars(source))
    if glob and "*" in expanded:
        yield from sorted(iglob(expanded))
    elif path.isdir(expanded):
        yield from sorted(iglob(path.join(expanded, "*")))
    else:
        yield expanded


def scan_avro(
    sources: Sequence[str | Path] | Sequence[BinaryIO] | str | Path | BinaryIO,
    *,
    batch_size: int = 1024,
    glob: bool = True,
    strict: bool = False,
    utf8_view: bool = False,
) -> LazyFrame:
    """Scan Avro files.

    Parameters
    ----------
    sources : The source(s) to scan. Local file paths or readable binary
        buffers. Binary buffers must be seekable (support ``seek``/``tell``);
        the reader rewinds them to read headers and to rewind for projection.
    batch_size : How many rows to attempt to read at a time.
    glob : Whether to use globbing to find files.
    strict : Whether to use strict mode when parsing avro. Incurs a
        performance hit.
    utf8_view : Whether to read strings as views. When ``False`` (default),
        UUIDs are read as binary and nullable strings preserve nulls. When
        ``True``, UUIDs are read as formatted strings and nulls in nullable
        strings are replaced with ``""`` (lossy). Since polars tends to work
        with string views internally, ``True`` is likely faster.
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

    def_batch_size = batch_size

    src = AvroSource(strs, bins)

    def get_schema() -> pl.Schema:
        return _arrow_to_frame(src.schema().empty_table()).schema

    def source_generator(
        with_columns: list[str] | None,
        predicate: Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[DataFrame]:
        avro_iter = src.batch_iter(
            strict, utf8_view, batch_size or def_batch_size, with_columns
        )
        for arrow_batch in avro_iter:
            batch = _arrow_to_frame(arrow_batch)
            if predicate is not None:
                # importing the typed native module confuses pyright's view of
                # DataFrame.filter here; the call is correct at runtime
                batch = batch.filter(predicate)  # type: ignore[reportUnknownMemberType]
            if n_rows is None:
                yield batch
            else:
                batch = batch[:n_rows]
                n_rows -= len(batch)
                yield batch
                if n_rows == 0:
                    break

    # type errors with callable schema
    # https://github.com/pola-rs/polars/issues/22182
    return register_io_source(source_generator, schema=get_schema)  # type: ignore[reportArgumentType]


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
    strict: bool = False,
    utf8_view: bool = False,
) -> DataFrame:
    """Read an Avro file into a DataFrame.

    Parameters
    ----------
    sources : The source(s) to scan. Local file paths or readable binary
        buffers. Binary buffers must be seekable (support ``seek``/``tell``);
        the reader rewinds them to read headers and to rewind for projection.
    columns : The columns to select.
    n_rows : The number of rows to read.
    row_index_name : The name of the row index column, or None to not add one.
    row_index_offset : The offset to start the row index at.
    rechunk : Whether to rechunk the DataFrame after reading.
    batch_size : How many rows to attempt to read at a time.
    glob : Whether to use globbing to find files.
    strict : Whether to use strict mode when parsing avro. Incurs a
        performance hit.
    utf8_view : Whether to read strings as views. When ``False`` (default),
        UUIDs are read as binary and nullable strings preserve nulls. When
        ``True``, UUIDs are read as formatted strings and nulls in nullable
        strings are replaced with ``""`` (lossy). Since polars tends to work
        with string views internally, ``True`` is likely faster.
    """
    lazy = scan_avro(
        sources,
        batch_size=batch_size,
        glob=glob,
        strict=strict,
        utf8_view=utf8_view,
    )
    if columns is not None:
        # see the filter note in scan_avro: the native import perturbs pyright's
        # view of LazyFrame.select; correct at runtime
        lazy = lazy.select(  # type: ignore[reportUnknownMemberType]
            [pl.nth(c) if isinstance(c, int) else pl.col(c) for c in columns]
        )
    if row_index_name is not None:
        lazy = lazy.with_row_index(row_index_name, offset=row_index_offset)
    if n_rows is not None:
        lazy = lazy.limit(n_rows)
    res = lazy.collect()
    return res.rechunk() if rechunk else res
