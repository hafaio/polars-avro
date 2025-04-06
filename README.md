# polars-avro

A polars io plugin for reading and writing avro files.

Polars is likely deprecating support for reading and writing avro files, and
this plugin fills in support. Currently it's about 7x slower at reading avro
files and up to 20x slower at writing files.

However, in exchange for speed you get:

1. future proof - this won't get deprecated
2. robust support - the current polars avro implementation has bugs with non-contiguous data frames
3. scan support - this can scan and push down predicates by chunk

## Python Usage

```py
from polars_avro import scan_avro, read_avro, write_avro

lazy = scan_avro(path)
frame = read_avro(path)
write_avro(frame, path)
```

## Rust Usage

There are two main objects exported in rust: `AvroScanner` for creating an
iterator of `DataFrames` from polars `ScanSources`, and `sink_avro` for writing
an iterator of `DataFrame`s to a `Write`able.

```rs
use polars_avro::{AvroScanner, sink_avro, WriteOptions};

let scanner = AvroScanner::new_from_sources(
    &ScanSources::Paths(...),
    1024,  //  batch size
    false, // expand globs
    None,  // cloud options
).unwrap()

sink_avro(
    scanner.map(Result::unwrap),
    ..., // impl Write
    WriteOptions::default(),
).unwrap();
```

> ℹ️ Avro supports writing with a fire compression schemes. In
> rust these features need to be enabled manually, e.g. `apache-avro/bzip` to
> enable bzip2 compression. Decompression is handled automatically.

## Development

### Rust

Standard `cargo` commands will build and test the rust library.

### Python

The python library is built with uv and maturin. Run the following to compile
rust for use by python:

For local rust development, run

```sh
uv run maturin develop -m Cargo.toml
```

to build a local copy of the rust interface.

### Testing

```sh
cargo clippy --all-features
cargo test
uv run ruff format --check
uv run ruff check
uv run pyright
uv run pytest
```
