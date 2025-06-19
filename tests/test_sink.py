"""Test write functionality."""

from io import BytesIO
from pathlib import Path

import polars as pl
import pytest

from polars_avro import AvroSpecError, read_avro, write_avro

from .utils import frames_equal


def test_binary_write() -> None:
    """Test writing to a buffer."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [1]})
    write_avro(frame, buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frames_equal(frame, duplicate)


def test_chunked_binary_write() -> None:
    """Test writing to a buffer."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [1, 2]})
    write_avro(frame, buff, batch_size=1)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frames_equal(frame, duplicate)


def test_empty_write() -> None:
    """Test writing an empty frame."""
    buff = BytesIO()
    frame = pl.from_dict({"x": []}, schema={"x": pl.Int32})
    write_avro(frame, buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frames_equal(frame, duplicate)


def test_struct_write() -> None:
    """Test writing a struct."""
    buff = BytesIO()
    frame = pl.from_dict(
        {"x": [[1, "a"]]},
        schema={
            "x": pl.Struct(
                {
                    "a": pl.Int32,
                    "b": pl.String,
                }
            )
        },
    )
    write_avro(frame, buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frames_equal(frame, duplicate)


def test_complex_write() -> None:
    """Test writing a complex data type."""
    buff = BytesIO()
    frame = pl.from_dict(
        {"x": [[[[1, "a"]], [None], []], []]},
        schema={
            "x": pl.List(
                pl.List(
                    pl.Struct(
                        {
                            "a": pl.Int32,
                            "b": pl.String,
                        }
                    )
                )
            )
        },
    )
    write_avro(frame, buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frames_equal(frame, duplicate)


def test_file_write(tmp_path: Path) -> None:
    """Test writing to a file."""
    path = tmp_path / "test.avro"

    frame = pl.from_dict({"x": [1]})
    write_avro(frame, path)
    duplicate = read_avro(path)
    assert frames_equal(frame, duplicate)


def test_no_int_promotion() -> None:
    """Test exception when writing ints without promotion."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [1]}, schema={"x": pl.Int8})
    with pytest.raises(AvroSpecError, match="unsupported type in write conversion: i8"):
        write_avro(frame, buff, promote_ints=False)


def test_no_array_promotion() -> None:
    """Test exception when writing arrays without promotion."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [[1]]}, schema={"x": pl.Array(pl.Int32, 1)})
    with pytest.raises(
        AvroSpecError, match="unsupported type in write conversion: array"
    ):
        write_avro(frame, buff, promote_array=False)
