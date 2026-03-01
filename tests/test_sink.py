"""Test write functionality."""

from io import BytesIO
from pathlib import Path

import polars as pl

from polars_avro import read_avro, write_avro


def test_binary_write() -> None:
    """Test writing to a buffer."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [1]})
    write_avro([frame], buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frame.equals(duplicate)


def test_chunked_binary_write() -> None:
    """Test writing to a buffer."""
    buff = BytesIO()
    one = pl.from_dict({"x": [1, 2]})
    two = pl.from_dict({"x": [3, 4]})
    write_avro([one, two], buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert pl.concat([one, two]).equals(duplicate)


def test_empty_write() -> None:
    """Test writing an empty frame."""
    buff = BytesIO()
    frame = pl.from_dict({"x": []}, schema={"x": pl.Int32})
    write_avro([frame], buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frame.equals(duplicate)


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
    write_avro([frame], buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frame.equals(duplicate)


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
    write_avro([frame], buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frame.equals(duplicate)


def test_file_write(tmp_path: Path) -> None:
    """Test writing to a file."""
    path = tmp_path / "test.avro"

    frame = pl.from_dict({"x": [1]})
    write_avro([frame], path)
    duplicate = read_avro(path)
    assert frame.equals(duplicate)


def test_write_array() -> None:
    """Test writing array columns (FixedSizeList is natively supported)."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [[1]]}, schema={"x": pl.Array(pl.Int32, 1)})
    write_avro([frame], buff)
    buff.seek(0)
    duplicate = read_avro(buff)
    assert frame.equals(duplicate)
