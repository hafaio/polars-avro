//! Polars pluging for reading and writing Apache avro files

#![warn(clippy::pedantic)]
#![warn(missing_docs)]

mod des;
mod error;
mod io;
#[cfg(feature = "pyo3")]
mod py;
mod scan;
mod ser;
mod sink;
#[cfg(test)]
mod tests;

pub use apache_avro::Codec;
pub use error::Error;
pub use scan::{AvroIter, AvroScanner};
pub use sink::{WriteOptions, sink_avro};
