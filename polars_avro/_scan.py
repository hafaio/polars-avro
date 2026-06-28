from collections.abc import Iterator, Mapping, Sequence
from glob import iglob
from os import path
from pathlib import Path
from typing import BinaryIO

import polars as pl
import pyarrow as pa
from polars import DataFrame, Expr, LazyFrame
from polars.io.plugins import register_io_source

from ._avro_rs import AvroSource
from ._source import SourceFactory, cloud_factory, is_url, seekable_factory


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


def scan_avro(  # noqa: PLR0913
    sources: Sequence[str | Path] | Sequence[BinaryIO] | str | Path | BinaryIO,
    *,
    batch_size: int = 1024,
    glob: bool = True,
    strict: bool = False,
    utf8_view: bool = False,
    storage_options: Mapping[str, str] | None = None,
) -> LazyFrame:
    """Scan Avro files.

    Parameters
    ----------
    sources : The source(s) to scan: local file paths, cloud URLs (``s3://``,
        ``gs://``, ``az://``, ``http(s)://``, ...), or readable binary buffers.
        Binary buffers must be seekable (support ``seek``/``tell``); the reader
        rewinds them to read headers and to rewind for projection. Cloud URLs
        require ``fsspec`` (plus the relevant backend, e.g. ``s3fs``).
    batch_size : How many rows to attempt to read at a time.
    glob : Whether to use globbing to find files (local paths only).
    strict : Whether to use strict mode when parsing avro. Incurs a
        performance hit.
    utf8_view : Whether to read strings as views. When ``False`` (default),
        UUIDs are read as binary and nullable strings preserve nulls. When
        ``True``, UUIDs are read as formatted strings and nulls in nullable
        strings are replaced with ``""`` (lossy). Since polars tends to work
        with string views internally, ``True`` is likely faster.
    storage_options : Extra options forwarded to ``fsspec.open`` for cloud URLs.
    """
    # normalize sources: local paths read natively, cloud/buffers open lazily
    opts = storage_options or {}
    strs: list[str] = []
    opened: list[SourceFactory] = []

    def classify(source: str | Path | BinaryIO) -> None:
        match source:
            case str() if is_url(source):
                opened.append(cloud_factory(source, opts))
            case str() | Path():
                strs.extend(expand_str(source, glob=glob))
            case _:
                opened.append(seekable_factory(source))

    match sources:
        case [*_]:
            for source in sources:
                classify(source)
        case _:
            classify(sources)

    def_batch_size = batch_size

    src = AvroSource(strs, opened)

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
    storage_options: Mapping[str, str] | None = None,
) -> DataFrame:
    """Read an Avro file into a DataFrame.

    Parameters
    ----------
    sources : The source(s) to scan: local file paths, cloud URLs (``s3://``,
        ``gs://``, ``az://``, ``http(s)://``, ...), or readable binary buffers.
        Binary buffers must be seekable (support ``seek``/``tell``); the reader
        rewinds them to read headers and to rewind for projection. Cloud URLs
        require ``fsspec`` (plus the relevant backend, e.g. ``s3fs``).
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
    storage_options : Extra options forwarded to ``fsspec.open`` for cloud URLs.
    """
    lazy = scan_avro(
        sources,
        batch_size=batch_size,
        glob=glob,
        strict=strict,
        utf8_view=utf8_view,
        storage_options=storage_options,
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
