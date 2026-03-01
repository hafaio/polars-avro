//! Polars plugin for reading and writing [Apache Avro](https://avro.apache.org/) files.
//!
//! Not all polars types can be written to avro. See [`Writer`] for types that
//! require casting before writing. When reading, some avro types map to
//! different polars types than you might expect — see [`ReadOptions`] for
//! details on `utf8_view` and type mapping behavior.

#![warn(clippy::pedantic)]
#![warn(missing_docs)]

mod error;
mod ffi;
#[cfg(feature = "pyo3")]
mod py;
mod scan;
mod schema;
mod sink;
#[cfg(test)]
mod tests;

/// Avro compression codecs for writing.
pub use arrow_avro::compression::CompressionCodec;
/// Error type for avro operations.
pub use error::Error;
/// Configuration for reading avro files.
pub use scan::{FullReadOptions, ReadOptions, Reader};
/// Schema projection and retrieval utilities.
pub use schema::{IndProj, NameProj, Projection, get_schema};
/// Incremental avro file writer.
pub use sink::Writer;
