"""Testing utilities."""

import polars as pl
from polars import DataFrame


def frames_equal(left: DataFrame, right: DataFrame) -> bool:
    """Test if frames are equal."""
    return (left == right).select(equal=pl.all_horizontal("*"))["equal"].all()
