# polars-avro

[![build](https://github.com/hafaio/polars-avro/actions/workflows/build.yml/badge.svg)](https://github.com/hafaio/polars-avro/actions/workflows/build.yml)
[![pypi](https://img.shields.io/pypi/v/polars-avro)](https://pypi.org/project/polars-avro/)
[![docs](https://img.shields.io/badge/api-docs-blue)](https://hafaio.github.io/polars-avro)

A polars io plugin for reading and writing avro files.

Polars is deprecating support for reading and writing avro files, and this
plugin fills in support. Currently it's about 7x slower at reading avro files
and up to 20x slower at writing files.

The reason it's slower is beause this uses the apache rust library, which is
fully complaint, but does a lot of unnecessary memory allocation and object
creation that the polars implementaiton avoids. However, this is likely not a
bottlneck, so the benefits of the standard implementation seem to outweight the
added computation.

In exchange for speed you get:

1. future proof - this won't get deprecated
2. robust support - the current polars avro implementation has bugs with non-contiguous data frames
3. better coverage - this supports reading map types as lists
4. scan support - this can scan and push down predicates by chunk

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
    false, // expand globs
    None,  // cloud options
    None,  // name for single column avros
).unwrap()

sink_avro(
    scanner.into_iter(
        1024, // batch size
        None, // columns to select
    ).map(Result::unwrap),
    ..., // impl Write
    WriteOptions::default(),
).unwrap();
```

> ℹ️ Avro supports writing with a file compression schemes. In
> rust these features need to be enabled manually, e.g. `apache-avro/bzip` to
> enable bzip2 compression. Decompression is handled automatically.

## Idiosyncorcies

Avro and Arrow don't align fully, and polars only supports a subset of arrow.
This library tries to allow you to serialize tow avro and deserialize from avro.
Trying to do both means that many types will change at each pass due to the way
serde works.

1. Avro only supports time with at most microsecond resolution, polars only
   supports time with nanosecond resolution, so writing times values truncates
   them. You must explicitely allow this behavior.
2. Avro fixed types don't support storing null values in the individual bytes so
   while a fixed type can be read into a u8 array, it must be serialized back as
   a list of i32s. This may be addressed with polars support for arrow
   fixedlengthbinary, but that seems unlikely.

## Benchmarks

| Library           |         Read Python |        Write Python |            Read Rust |           Write Rust |
| ----------------- | ------------------: | ------------------: | -------------------: | -------------------: |
| `polars`          |   6.0319 ms ( 1.00) |   3.0663 ms ( 1.00) |  41,393.76 ns (1.00) |  42,046.06 ns (1.00) |
| `polars-avro`     |  39.9563 ms ( 6.62) |  67.9542 ms (22.16) | 248,206.20 ns (6.00) | 395,947.65 ns (9.42) |
| `polars-fastavro` | 179.0461 ms (29.68) | 246.3771 ms (80.35) |                    - |                    - |

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

to build a local copy of the rust interface. Add `-r` if you want to trust the
benchmark results.

### Testing

```sh
cargo fmt --check
cargo clippy --all-features
cargo test
uv run ruff format --check
uv run ruff check
uv run pyright
uv run pytest
```

### Benchmarking

```sh
cargo +nightly bench
uv run pytest
```

> ℹ️ For python benchmarks, make sure you've compiled in release mode: `uv run maturin develop -m Cargo.toml -r`

### Releasing

```sh
rm -rf dist
uv build --sdist
uv run maturin build -r -o dist --target aarch64-apple-darwin
uv run maturin build -r -o dist --target aarch64-unknown-linux-gnu --zig
uv publish --username __token__
```
