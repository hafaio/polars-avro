//! Internal error types

use apache_avro::Error as AvroError;
use arrow::datatypes::Schema;
use arrow::error::ArrowError;
use polars::error::PolarsError;
use polars::prelude::DataType;
use std::collections::HashMap;
use std::convert::Infallible;
use std::error::Error as StdError;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::io;
use std::sync::Arc;

/// Any error raised by this crate
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// An error from polars
    Polars(PolarsError),
    /// An error from the arrow library
    Arrow(ArrowError),
    /// An error from parsing the avro header
    Avro(AvroError),
    /// Cannot scan empty sources
    EmptySources,
    /// Top level avro schema must be a record
    NonRecordSchema,
    /// Avro and arrow don't share the same types and this type can't be converted
    ///
    /// There are options for sink that allow promotion or truncation that alter
    /// what types can be serialized
    UnsupportedPolarsType(DataType),
    /// Polars allows unspecified enums, but avro does not
    NullEnum,
    /// Happens when an avro header doesn't fit in an i64
    LargeHeader,
    /// If not all schemas in a batch were identical
    NonMatchingSchemas {
        /// The schema we expected (from the first source)
        expected: Schema,
        /// The schema we actually got
        actual: Arc<Schema>,
    },
    /// If a column wasn't found in the schema
    ColumnNotFound(String),
    /// Column index is out of bounds
    ColumnIndexOutOfBounds(usize),
    /// I/O related errors
    IO(io::Error, String),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Error::Polars(e) => write!(f, "Error from polars: {e}"),
            Error::Arrow(e) => write!(f, "Error from arrow: {e}"),
            Error::Avro(e) => write!(f, "Error from avro: {e}"),
            Error::EmptySources => write!(f, "Cannot scan empty sources"),
            Error::NonRecordSchema => write!(f, "Top level avro schema must be a record"),
            Error::UnsupportedPolarsType(dtype) => write!(
                f,
                "Avro and arrow don't share the same types, this polars type can't be converted: {dtype}",
            ),
            Error::NullEnum => write!(f, "Polars allows unspecified enums, but avro does not"),
            Error::LargeHeader => write!(f, "Avro header is too large"),
            Error::NonMatchingSchemas { expected, actual } => {
                write!(f, "schemas differ:")?;
                let mut act_by_name: HashMap<_, _> = actual
                    .fields()
                    .iter()
                    .map(|f| (f.name().as_str(), f.data_type()))
                    .collect();
                for field in expected.fields() {
                    match act_by_name.remove(field.name().as_str()) {
                        None => write!(
                            f,
                            " removed \"{}\" ({:?}).",
                            field.name(),
                            field.data_type()
                        )?,
                        Some(act_dt) if act_dt != field.data_type() => write!(
                            f,
                            " \"{}\": expected {:?}, got {:?}.",
                            field.name(),
                            field.data_type(),
                            act_dt
                        )?,
                        _ => {}
                    }
                }
                for (name, dt) in &act_by_name {
                    write!(f, " added \"{name}\" ({dt:?}).")?;
                }
                Ok(())
            }
            Error::ColumnNotFound(col) => write!(f, "Column \"{col}\" wasn't found in the schema"),
            Error::ColumnIndexOutOfBounds(ind) => {
                write!(f, "Column index {ind} is out of bounds")
            }
            Error::IO(err, path) => write!(f, "Problem with {path}: {err}"),
        }
    }
}

impl StdError for Error {}

impl From<ArrowError> for Error {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}

impl From<AvroError> for Error {
    fn from(value: AvroError) -> Self {
        Self::Avro(value)
    }
}

impl From<PolarsError> for Error {
    fn from(value: PolarsError) -> Self {
        Self::Polars(value)
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::IO(value, "io".into())
    }
}

impl From<Infallible> for Error {
    fn from(value: Infallible) -> Self {
        match value {}
    }
}

#[cfg(test)]
mod tests {
    use super::Error;
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
    use arrow::error::ArrowError;
    use polars::error::PolarsError;
    use polars::prelude::DataType;
    use std::sync::Arc;

    #[test]
    fn test_display() {
        let expected = Schema::new(vec![
            Field::new("kept", ArrowDataType::Int32, false),
            Field::new("changed", ArrowDataType::Int32, false),
            Field::new("removed", ArrowDataType::Float64, false),
        ]);
        let actual = Arc::new(Schema::new(vec![
            Field::new("kept", ArrowDataType::Int32, false),
            Field::new("changed", ArrowDataType::Utf8, false),
            Field::new("added", ArrowDataType::Boolean, false),
        ]));
        for err in [
            Error::Polars(PolarsError::NoData("test".into())),
            Error::Arrow(ArrowError::NotYetImplemented("test".into())),
            Error::EmptySources,
            Error::NonRecordSchema,
            Error::UnsupportedPolarsType(DataType::Null),
            Error::NullEnum,
            Error::NonMatchingSchemas { expected, actual },
        ] {
            assert!(!format!("{err}").is_empty());
        }
    }

    #[test]
    fn test_non_matching_display() {
        let expected = Schema::new(vec![
            Field::new("kept", ArrowDataType::Int32, false),
            Field::new("changed", ArrowDataType::Int32, false),
            Field::new("removed", ArrowDataType::Float64, false),
        ]);
        let actual = Arc::new(Schema::new(vec![
            Field::new("kept", ArrowDataType::Int32, false),
            Field::new("changed", ArrowDataType::Utf8, false),
            Field::new("added", ArrowDataType::Boolean, false),
        ]));
        let msg: String = format!("{}", Error::NonMatchingSchemas { expected, actual });
        assert!(msg.contains("removed \"removed\""), "{msg}");
        assert!(msg.contains("added \"added\""), "{msg}");
        assert!(msg.contains("\"changed\": expected"), "{msg}");
        assert!(!msg.contains("\"kept\""), "{msg}");
    }
}
