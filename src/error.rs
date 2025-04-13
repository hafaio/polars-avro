//! Internal error types

use std::error::Error as StdError;
use std::fmt::{Display, Formatter, Result as FmtResult};

use apache_avro::types::Value;
use apache_avro::{Error as AvroError, Schema};
use polars::error::PolarsError;
use polars::prelude::{ArrowDataType, DataType};

/// Any error raised by this crate
#[derive(Debug)]
pub enum Error {
    /// An error from polars
    Polars(PolarsError),
    /// An error propogated from the underlying avro library
    Avro(Box<AvroError>), // NOTE AvroErrors are big!
    /// Cannot scan empty sources
    EmptySources,
    /// Top level avro schema must be a record
    NonRecordSchema(Box<Schema>), // NOTE AvroSchema's are big!
    /// Avro and arrow don't share the same types
    UnsupportedAvroType(Box<Schema>), // NOTE AvroSchema's are big!
    /// Avro and arrow don't share the same types and this type can't be cnverted
    ///
    /// There are options for sink that allow promotion or truncation that alter
    /// what types can be serialized
    UnsupportedPolarsType(DataType),
    /// Polars allows unspecified enums, but avro does not
    NullEnum,
    /// If not all schemas in a batch were identical
    NonMatchingSchemas,
    /// If arrow type doesn't match expected schema type during serialization
    InvalidArrowType(DataType, ArrowDataType),
    /// If an avro value didn't match the expected deserializer
    InvalidAvroValue(Value),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Error::Polars(e) => write!(f, "Error from polars: {e}"),
            Error::Avro(e) => write!(f, "Error from avro: {e}"),
            Error::EmptySources => write!(f, "Cannot scan empty sources"),
            Error::NonRecordSchema(schema) => {
                write!(
                    f,
                    "Top level avro schema must be a record, but got: {schema:?}"
                )
            }
            Error::UnsupportedAvroType(schema) => {
                write!(
                    f,
                    "Avro and arrow don't share the same types, the specific sub-schema cannot be read: {schema:?}"
                )
            }
            Error::UnsupportedPolarsType(dtype) => {
                write!(
                    f,
                    "Avro and arrow don't share the same types, this polars type can't be cnverted: {dtype}",
                )
            }
            Error::NullEnum => write!(f, "Polars allows unspecified enums, but avro does not"),
            Error::NonMatchingSchemas => {
                write!(f, "All batches must share the same schema")
            }
            Error::InvalidArrowType(dtype, arrow_dtype) => {
                write!(
                    f,
                    "Column dtypes must match their arrow types during serialization, but {dtype} != {arrow_dtype:?}",
                )
            }
            Error::InvalidAvroValue(value) => {
                write!(
                    f,
                    "Avro value didn't match the expected deserializer: {value:?}",
                )
            }
        }
    }
}

impl StdError for Error {}

impl From<AvroError> for Error {
    fn from(value: AvroError) -> Self {
        Self::Avro(Box::new(value))
    }
}

impl From<PolarsError> for Error {
    fn from(value: PolarsError) -> Self {
        Self::Polars(value)
    }
}

#[cfg(test)]
mod tests {
    use apache_avro::{Error as AvroError, Schema, types::Value};
    use polars::error::PolarsError;
    use polars::prelude::{ArrowDataType, DataType};

    use super::Error;

    #[test]
    fn test_display() {
        for err in [
            Error::Polars(PolarsError::NoData("test".into())),
            Error::Avro(Box::new(AvroError::EmptyUnion)),
            Error::EmptySources,
            Error::NonRecordSchema(Box::new(Schema::Null)),
            Error::UnsupportedAvroType(Box::new(Schema::Null)),
            Error::UnsupportedPolarsType(DataType::Null),
            Error::NullEnum,
            Error::NonMatchingSchemas,
            Error::InvalidArrowType(DataType::Null, ArrowDataType::Null),
            Error::InvalidAvroValue(Value::Null),
        ] {
            assert!(!format!("{err}").is_empty());
        }
    }
}
