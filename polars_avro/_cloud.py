from collections.abc import Mapping
from typing import Literal, TypeAlias
from urllib.parse import urlparse

from polars.io.cloud import (
    CredentialProvider,
    CredentialProviderAWS,
    CredentialProviderAzure,
    CredentialProviderFunction,
    CredentialProviderGCP,
)

_CLOUD_SCHEMES: Mapping[str, type[CredentialProvider]] = {
    "s3": CredentialProviderAWS,
    "s3a": CredentialProviderAWS,
    "gs": CredentialProviderGCP,
    "gcp": CredentialProviderGCP,
    "gcs": CredentialProviderGCP,
    "az": CredentialProviderAzure,
    "azure": CredentialProviderAzure,
    "adl": CredentialProviderAzure,
    "abfs": CredentialProviderAzure,
    "abfss": CredentialProviderAzure,
}


CredentialProviderInput: TypeAlias = CredentialProviderFunction | Literal["auto"] | None


def resolve_credentials(
    credential_provider: CredentialProviderInput,
    paths: list[str],
    storage_options: Mapping[str, str] | None,
) -> list[tuple[str, str]]:
    """Resolve cloud credentials into a flat list of (key, value) pairs."""
    opts: dict[str, str]
    if credential_provider is None:
        opts = {}
    elif credential_provider == "auto":
        providers: set[type[CredentialProvider]] = set()
        for path in paths:
            scheme = urlparse(path).scheme.lower()
            provider = _CLOUD_SCHEMES.get(scheme)
            if provider is not None:
                providers.add(provider)
        match [*providers]:
            case []:
                opts = {}
            case [provider]:
                opts, _ = provider()()
            case _:
                raise ValueError(
                    "credential provider set to auto, but multiple cloud schemes identified"
                )
    else:
        opts, _ = credential_provider()
    if storage_options is not None:
        opts.update(storage_options)
    return [*opts.items()]
