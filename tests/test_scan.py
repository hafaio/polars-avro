"""Test scan functionality."""

from io import BytesIO
from typing import NoReturn
from unittest.mock import patch

import fastavro
import polars as pl
import pytest
from polars.io.cloud.credential_provider._builder import (
    CredentialProviderBuilder,  # type: ignore[reportPrivateImportUsage]
)

from polars_avro import read_avro, scan_avro, write_avro

from .utils import frames_equal


def test_scan_avro() -> None:
    """Test generic scan of files."""
    frame = scan_avro("resources/food.avro").with_row_index("row_index").collect()
    assert frame["row_index"].to_list() == [*range(27)]

    frame = (
        scan_avro("resources/food.avro")
        .with_row_index("row_index")
        .filter(pl.col("category") == pl.lit("vegetables"))  # type: ignore
        .collect()
    )
    assert frame["row_index"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    frame = (
        scan_avro("resources/food.avro")
        .with_row_index("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))  # type: ignore
        .collect()
    )
    assert frame["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


def test_projection_pushdown_avro() -> None:
    """Test that projection is pushed down to scan."""
    file_path = "resources/food.avro"
    lazy = scan_avro(file_path).select(pl.col.calories)

    explain = lazy.explain()

    assert "simple π" not in explain
    assert "PROJECT 1/4 COLUMNS" in explain

    normal = lazy.collect()
    unoptimized = lazy.collect(no_optimization=True)
    assert frames_equal(normal, unoptimized)


def test_predicate_pushdown_avro() -> None:
    """Test that predicate is pushed down to scan."""
    file_path = "resources/food.avro"
    thresh = 80
    lazy = scan_avro(file_path).filter(pl.col("calories") > thresh)  # type: ignore

    explain = lazy.explain()

    assert "FILTER" not in explain
    assert """SELECTION: [(col("calories")) > (80)]""" in explain

    normal = lazy.collect()
    unoptimized = lazy.collect(no_optimization=True)
    assert frames_equal(normal, unoptimized)


def test_glob_n_rows() -> None:
    """Test that globbing and n_rows work."""
    file_path = "resources/*.avro"
    frame = scan_avro(file_path).limit(28).collect()

    # 27 rows from food.avro and 1 from grains.avro
    assert frame.shape == (28, 4)

    # take first and last rows
    assert frame[[0, 27]].to_dict(as_series=False) == {
        "category": ["vegetables", "rice"],
        "calories": [45, 9],
        "fats_g": [0.5, 0.0],
        "sugars_g": [2, 0.3],
    }


def test_many_files() -> None:
    """Test that scan works with many files."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [5, 12, 14]})
    write_avro(frame, buff)

    buffs = [BytesIO(buff.getvalue()) for _ in range(1023)]
    res = scan_avro(buffs).collect()
    reference = pl.from_dict({"x": [5, 12, 14] * 1023})
    assert frames_equal(res, reference)


def test_scan_nrows_empty() -> None:
    """Test that scan doesn't panic with n_rows set to 0."""
    file_path = "resources/food.avro"
    frame = scan_avro(file_path).head(0).collect()
    reference = read_avro(file_path).head(0)
    assert frames_equal(frame, reference)


def test_scan_filter_empty() -> None:
    """Test that scan doesn't panic when filter removes all rows."""
    file_path = "resources/food.avro"
    frame = scan_avro(file_path).filter(pl.col("category") == "empty").collect()  # type: ignore
    reference = read_avro(file_path).filter(pl.col("category") == "empty")  # type: ignore
    assert frames_equal(frame, reference)


def test_directory() -> None:
    """Test scan on directory."""
    frame = scan_avro("resources").collect()
    assert frame.shape == (30, 4)


def test_avro_list_arg() -> None:
    """Test that scan works when passing a list."""
    first = "resources/food.avro"
    second = "resources/grains.avro"

    frame = scan_avro([first, second]).collect()
    assert frame.shape == (30, 4)
    assert frame.row(-1) == ("corn", 99, 0.1, 10.4)
    assert frame.row(0) == ("vegetables", 45, 0.5, 2)


def test_glob_single_scan() -> None:
    """Test that globbing works with a single file."""
    file_path = "resources/food*.avro"
    frame = scan_avro(file_path)

    explain = frame.explain()

    assert explain.count("SCAN") == 1
    assert "UNION" not in explain


def test_scan_in_memory() -> None:
    """Test that scan works for in memory buffers."""
    frame = pl.from_dict({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    buff = BytesIO()
    write_avro(frame, buff)

    buff.seek(0)
    scanned = scan_avro(buff).collect()
    assert frames_equal(frame, scanned)

    buff.seek(0)
    scanned = scan_avro(buff).slice(1, 2).collect()
    assert frames_equal(frame.slice(1, 2), scanned)

    buff.seek(0)
    scanned = scan_avro(buff).slice(-1, 1).collect()
    assert frames_equal(frame.slice(-1, 1), scanned)

    other = BytesIO(buff.getvalue())

    buff.seek(0)
    scanned = scan_avro([buff, other]).collect()
    assert frames_equal(pl.concat([frame, frame]), scanned)

    buff.seek(0)
    other.seek(0)
    scanned = scan_avro([buff, other]).slice(1, 3).collect()
    assert frames_equal(pl.concat([frame, frame]).slice(1, 3), scanned)

    buff.seek(0)
    other.seek(0)
    scanned = scan_avro([buff, other]).slice(-4, 3).collect()
    assert frames_equal(pl.concat([frame, frame]).slice(-4, 3), scanned)


def test_read_map_type() -> None:
    """Test that we can read a map type."""
    buff = BytesIO()
    values = [{"map": {"a": 5}}, {"map": None}, {"map": {"c": 8, "f": -10}}]
    fastavro.writer(  # type: ignore
        buff,
        {
            "type": "record",
            "name": "map_test",
            "fields": [
                {"name": "map", "type": ["null", {"type": "map", "values": "int"}]}
            ],
        },
        values,
    )
    buff.seek(0)
    # we need to sort the list to guarantee order for comparison
    res = scan_avro(buff).select(pl.col("map").list.sort()).collect()
    expected = pl.from_dict(
        {"map": [[["a", 5]], None, [["c", 8], ["f", -10]]]},
        schema={"map": pl.List(pl.Struct({"key": pl.String, "value": pl.Int32}))},
    )
    assert frames_equal(res, expected)


def test_read_options() -> None:
    """Test read works with options."""
    frame = read_avro(
        "resources/food.avro", row_index_name="row_index", columns=[1], n_rows=11
    )
    assert frame.shape == (11, 2)
    assert frame["row_index"].to_list() == [*range(11)]


def test_filename_in_err() -> None:
    """Test that invalid filename is reported in error."""
    lazy = scan_avro("does not exist")
    with pytest.raises(Exception, match="does not exist"):
        lazy.collect()


def test_empty_sources() -> None:
    """Test that empty sources raises an error."""
    lazy = scan_avro([])
    with pytest.raises(Exception, match="must scan at least one source"):
        lazy.collect()


class SentinelError(AssertionError):
    """A sentinel error for raising."""

    pass


def raises(*_: object, **__: object) -> NoReturn:
    """Raise a sentinel error."""
    raise SentinelError("sentinel")


def test_credential_provider_scan() -> None:
    """Test that credential provider is passed to scan."""
    with patch.object(CredentialProviderBuilder, "__init__", raises):
        with pytest.raises(SentinelError):
            scan_avro("s3://bucket/path", credential_provider="auto")

        with pytest.raises(AssertionError):
            scan_avro(
                "s3://bucket/path",
                credential_provider="auto",
                storage_options={"aws_region": "eu-west-1"},
            )

    with patch.object(CredentialProviderBuilder, "__init__", raises):
        # Passing `None` should disable the automatic instantiation of
        # `CredentialProviderAWS`
        scan_avro(
            "s3://bucket/path",
            credential_provider=None,
        )
        # Passing `storage_options` should disable the automatic instantiation of
        # `CredentialProviderAWS`
        scan_avro(
            "s3://bucket/path",
            credential_provider="auto",
            storage_options={
                "aws_secret_access_key_id": "polars",
            },
        )

    with pytest.raises(Exception, match="is not callable"):
        scan_avro(
            "s3://bucket/path",
            credential_provider=raises,
            storage_options={"aws_region": "eu-west-1"},
        ).collect()
