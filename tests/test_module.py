"""Bencharmk avro reading and writing compared to native polars implementation."""

from collections.abc import Callable
from datetime import date, datetime
from io import BytesIO

import polars as pl
import polars_fastavro
import pytest
from polars import DataFrame
from pytest_benchmark.fixture import BenchmarkFixture

import polars_avro

from .utils import frames_equal


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(pl.from_dict({}, schema={}), id="empty"),
        pytest.param(
            pl.from_dict({"col": [None, None]}, schema={"col": pl.Null}),
            id="nulls",
            marks=pytest.mark.xfail(
                reason="https://github.com/apache/avro-rs/issues/181"
            ),
        ),
        pytest.param(
            pl.from_dict({"col": [True, False, None]}, schema={"col": pl.Boolean}),
            id="bools",
        ),
        pytest.param(
            pl.from_dict({"col": [-1, 0, 6, None]}, schema={"col": pl.Int32}),
            id="ints",
        ),
        pytest.param(
            pl.from_dict({"col": [-1, 0, 6, None]}, schema={"col": pl.Int64}),
            id="longs",
        ),
        pytest.param(
            pl.from_dict({"col": [date.today(), None]}, schema={"col": pl.Date}),
            id="dates",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [datetime.now(), None]},
                schema={"col": pl.Datetime("ms", "UTC")},
            ),
            id="datetime-ms",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [datetime.now(), None]},
                schema={"col": pl.Datetime("us", "UTC")},
            ),
            id="datetime-us",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [datetime.now(), None]},
                schema={"col": pl.Datetime("ms")},
            ),
            id="datetime-ms-local",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [datetime.now(), None]},
                schema={"col": pl.Datetime("us")},
            ),
            id="datetime-us-local",
        ),
        pytest.param(
            pl.from_dict({"col": [-1.0, 0.0, 6.0, None]}, schema={"col": pl.Float32}),
            id="floats",
        ),
        pytest.param(
            pl.from_dict({"col": [-1.0, 0.0, 6.0, None]}, schema={"col": pl.Float64}),
            id="doubles",
        ),
        pytest.param(
            pl.from_dict(
                {"col": ["a", "b", None]}, schema={"col": pl.Enum(["a", "b"])}
            ),
            id="enum",
            marks=pytest.mark.xfail(
                reason="https://github.com/pola-rs/polars/issues/22273"
            ),
        ),
        pytest.param(
            pl.from_dict({"col": [b"a", b"b", None]}, schema={"col": pl.Binary}),
            id="binary",
        ),
        pytest.param(
            pl.from_dict({"col": ["a", "b", None]}, schema={"col": pl.String}),
            id="string",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [["a", None], ["b", "c"], ["d"], None]},
                schema={"col": pl.List(pl.String)},
            ),
            id="list-string",
        ),
        pytest.param(
            pl.from_dict(
                {
                    "col": [
                        {"s": "a", "i": 5},
                        {"s": "b", "i": None},
                        {"s": None, "i": 6},
                        {"s": None, "i": None},
                        None,
                    ]
                },
                schema={"col": pl.Struct({"s": pl.String, "i": pl.Int32})},
            ),
            id="struct",
        ),
        pytest.param(
            pl.from_dict(
                {
                    "col": [
                        [[{"s": "a", "i": 5}, None], [], None],
                        [[{"s": "b", "i": None}]],
                        [[{"s": None, "i": 6}], [{"s": None, "i": None}]],
                        None,
                    ]
                },
                schema={
                    "col": pl.List(pl.List(pl.Struct({"s": pl.String, "i": pl.Int32})))
                },
            ),
            id="nested",
        ),
        pytest.param(
            pl.from_dict(
                {
                    "struct": [[1, "a"], [None, "b"], [3, None]],
                    "single": [[1.0], [2.0], [3.0]],
                },
                schema={
                    "struct": pl.Struct({"a": pl.Int32, "b": pl.String}),
                    "single": pl.Struct({"x": pl.Float64}),
                },
            ),
            id="double struct",
        ),
        pytest.param(
            pl.from_dict(
                {
                    "one": ["a", "b", None],
                    "two": ["c", None, "d"],
                },
                schema={
                    "one": pl.Enum(["a", "b", "c"]),
                    "two": pl.Enum(["c", "d"]),
                },
            ),
            id="double enum",
            marks=pytest.mark.xfail(
                reason="https://github.com/pola-rs/polars/issues/22273"
            ),
        ),
    ],
)
def test_transitive(frame: pl.DataFrame):
    """Test that frames can be serialized and deserialized."""
    buff = BytesIO()
    polars_avro.write_avro(frame, buff)
    buff.seek(0)
    dup = polars_avro.read_avro(buff)

    assert frames_equal(frame, dup)


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
    ).with_columns(contig=pl.int_range(pl.len()))  # type: ignore
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
