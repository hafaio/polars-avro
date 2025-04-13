"""Bencharmk avro reading and writing compared to native polars implementation."""

from collections.abc import Callable
from io import BytesIO

import polars as pl
import polars_fastavro
import pytest
from polars import DataFrame
from pytest_benchmark.fixture import BenchmarkFixture

import polars_avro

from .utils import frames_equal


@pytest.mark.parametrize(
    "num",
    [
        pytest.param(1024, marks=pytest.mark.benchmark(group="read small")),
        pytest.param(128 * 1024, marks=pytest.mark.benchmark(group="read large")),
    ],
)
@pytest.mark.parametrize(
    "read_func",
    [
        pytest.param(pl.read_avro, id="polars"),
        pytest.param(polars_fastavro.read_avro, id="polars_fastavro"),
        pytest.param(polars_avro.read_avro, id="polars_avro"),
    ],
)
def test_read(
    read_func: Callable[[BytesIO], DataFrame], num: int, benchmark: BenchmarkFixture
) -> None:
    """Benchmark reading standard."""
    frame = pl.from_dict(
        {"ints": [*range(num)], "strings": [str(x) for x in range(num)]}
    )
    buff = BytesIO()
    polars_avro.write_avro(frame, buff)

    def func() -> None:
        buff.seek(0)
        read_func(buff)

    benchmark(func)


@pytest.mark.parametrize(
    "num",
    [
        pytest.param(1024, marks=pytest.mark.benchmark(group="write small")),
        pytest.param(128 * 1024, marks=pytest.mark.benchmark(group="write large")),
    ],
)
@pytest.mark.parametrize(
    "write_func",
    [
        pytest.param(DataFrame.write_avro, id="polars"),
        pytest.param(polars_fastavro.write_avro, id="polars_fastavro"),
        pytest.param(polars_avro.write_avro, id="polars_avro"),
    ],
)
def test_write(
    write_func: Callable[[DataFrame, BytesIO], None],
    num: int,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark reading standard."""
    frame = pl.from_dict(
        {"ints": [*range(num)], "strings": [str(x) for x in range(num)]}
    )

    def func() -> None:
        write_func(frame, BytesIO())

    benchmark(func)


@pytest.mark.parametrize(
    "write_func,read_func",
    [
        pytest.param(
            DataFrame.write_avro, pl.read_avro, id="polars", marks=pytest.mark.xfail
        ),
        pytest.param(polars_avro.write_avro, polars_avro.read_avro, id="polars_avro"),
    ],
)
def test_noncontiguous_chunks(
    write_func: Callable[[DataFrame, BytesIO], None],
    read_func: Callable[[BytesIO], pl.DataFrame],
) -> None:
    """Test that non contiguous arrays can be written and read."""
    frame = pl.concat(
        [
            pl.from_dict({"split": [*range(3)]}),
            pl.from_dict({"split": [*range(3, 6)]}),
        ],
        rechunk=False,
    ).with_columns(contig=pl.int_range(pl.len()))
    buff = BytesIO()
    write_func(frame, buff)
    buff.seek(0)
    dup = read_func(buff)
    assert frames_equal(frame, dup)


@pytest.mark.parametrize(
    "write_func,read_func",
    [
        pytest.param(
            DataFrame.write_avro, pl.read_avro, id="polars", marks=pytest.mark.xfail
        ),
        pytest.param(polars_avro.write_avro, polars_avro.read_avro, id="polars_avro"),
    ],
)
def test_noncontiguous_arrays(
    write_func: Callable[[DataFrame, BytesIO], None],
    read_func: Callable[[BytesIO], pl.DataFrame],
) -> None:
    """Test that non contiguous arrays can be written and read."""
    frame = pl.concat(
        [
            pl.from_dict({"split": [*range(3)]}),
            pl.from_dict({"split": [*range(3, 6)]}),
        ],
        rechunk=False,
    )
    buff = BytesIO()
    write_func(frame, buff)
    buff.seek(0)
    dup = read_func(buff)
    assert frames_equal(frame, dup)
