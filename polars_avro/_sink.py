from __future__ import annotations

from collections.abc import Mapping
from os import path
from pathlib import Path
from typing import BinaryIO, Literal, cast

from polars import CredentialProviderFunction, DataFrame
from polars.io.cloud.credential_provider._builder import (
    _init_credential_provider_builder,  # type: ignore[reportPrivateImportUsage]
)

from ._avro_rs import Codec
from ._avro_rs import write_avro as write_avro_rs


def write_avro(  # noqa: PLR0913
    frame: DataFrame,
    dest: str | Path | BinaryIO,
    *,
    batch_size: int | None = None,
    codec: Codec = Codec.Null,
    promote_ints: bool = True,
    promote_array: bool = True,
    truncate_time: bool = False,
    retries: int = 2,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
    storage_options: Mapping[str, str] | None = None,
) -> None:
    """Write a DataFrame to an Avro file.

    Parameters
    ----------
    frame : The DataFrame to write.
    dest : Where to write the dataframe
    codec : The codec to use for compression, or Null for uncompressed.
    promote_ints : Whether to promote integer columns to a larger type in order
        to support writing them.
    promote_array : Whether to promote array columns to lists in order to
        support writing them to avro.
    truncate_time : Whether to truncate time columns so they can be written to
        avro as avro doesn't support time with nanosecond precision.
    retries : The number of times to retry cloud operations.
    credential_provider : The credential provider to use for cloud operations.
        Defaults to "auto" which uses the default credential provider.
    storage_options : Additional options for cloud operations.
    """
    # normalize dest
    match dest:
        case str() | Path():
            normed = path.expanduser(dest)
        case _:
            normed = dest

    # normalize cloud options
    cloud_options = None if storage_options is None else [*storage_options.items()]
    credential_provider_builder = _init_credential_provider_builder(
        credential_provider,
        normed,
        cast(dict[str, str], storage_options),  # incorrect signature
        "scan_avro",
    )

    # chunk array if batchsize specified
    if batch_size is None:
        frames = [frame]
    else:
        frames = [
            frame[i : i + batch_size].rechunk()
            for i in range(0, len(frame), batch_size)
        ]

    write_avro_rs(
        frames,
        normed,
        codec,
        promote_ints,
        promote_array,
        truncate_time,
        cloud_options,
        credential_provider_builder,
        retries,
    )
