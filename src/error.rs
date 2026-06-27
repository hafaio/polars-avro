//! Internal error types

use apache_avro::Error as AvroError;
use arrow::datatypes::Schema;
use arrow::error::ArrowError;
use arrow_avro::errors::AvroError as ArrowAvroError;
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
    /// An error from the arrow library
    Arrow(ArrowError),
    /// An error from the arrow-avro library
    ArrowAvro(ArrowAvroError),
    /// An error from parsing the avro header
    Avro(AvroError),
    /// Cannot scan empty sources
    EmptySources,
    /// Top level avro schema must be a record
    NonRecordSchema,
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
            Error::Arrow(e) => write!(f, "Error from arrow: {e}"),
            Error::ArrowAvro(e) => write!(f, "Error from arrow-avro: {e}"),
            Error::Avro(e) => write!(f, "Error from avro: {e}"),
            Error::EmptySources => write!(f, "Cannot scan empty sources"),
            Error::NonRecordSchema => write!(f, "Top level avro schema must be a record"),
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

impl From<ArrowAvroError> for Error {
    fn from(value: ArrowAvroError) -> Self {
        Self::ArrowAvro(value)
    }
}

impl From<AvroError> for Error {
    fn from(value: AvroError) -> Self {
        Self::Avro(value)
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
    use arrow_avro::errors::AvroError as ArrowAvroError;
    use std::io;
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
        let avro_err = apache_avro::Schema::parse_str("not a schema").unwrap_err();
        for err in [
            Error::Arrow(ArrowError::NotYetImplemented("test".into())),
            Error::ArrowAvro(ArrowAvroError::General("test".into())),
            Error::Avro(avro_err),
            Error::EmptySources,
            Error::NonRecordSchema,
            Error::LargeHeader,
            Error::NonMatchingSchemas { expected, actual },
            Error::ColumnNotFound("missing".into()),
            Error::ColumnIndexOutOfBounds(7),
            Error::IO(io::Error::other("boom"), "path".into()),
        ] {
            assert!(!format!("{err}").is_empty());
        }
    }

    #[test]
    fn test_from_conversions() {
        assert!(matches!(
            Error::from(ArrowError::NotYetImplemented("test".into())),
            Error::Arrow(_)
        ));
        assert!(matches!(
            Error::from(ArrowAvroError::General("test".into())),
            Error::ArrowAvro(_)
        ));
        assert!(matches!(
            Error::from(apache_avro::Schema::parse_str("not a schema").unwrap_err()),
            Error::Avro(_)
        ));
        assert!(matches!(
            Error::from(io::Error::other("boom")),
            Error::IO(_, _)
        ));
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

    /// A `fmt::Write` that fails as soon as it sees a marker substring, used to
    /// drive the writer-error paths inside `Display`.
    struct FailOn(&'static str);

    impl std::fmt::Write for FailOn {
        fn write_str(&mut self, segment: &str) -> std::fmt::Result {
            if segment.contains(self.0) {
                Err(std::fmt::Error)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn test_non_matching_write_errors() {
        use std::fmt::Write as _;

        // failing while writing a removed field propagates the error
        let expected = Schema::new(vec![Field::new("gone", ArrowDataType::Int32, false)]);
        let actual = Arc::new(Schema::new(Vec::<Field>::new()));
        let removed = Error::NonMatchingSchemas { expected, actual };
        assert!(write!(FailOn("removed"), "{removed}").is_err());

        // failing while writing a changed field propagates the error
        let expected = Schema::new(vec![Field::new("col", ArrowDataType::Int32, false)]);
        let actual = Arc::new(Schema::new(vec![Field::new(
            "col",
            ArrowDataType::Utf8,
            false,
        )]));
        let changed = Error::NonMatchingSchemas { expected, actual };
        assert!(write!(FailOn("expected"), "{changed}").is_err());

        // failing on the leading "schemas differ:" write propagates the error
        let expected = Schema::new(vec![Field::new("col", ArrowDataType::Int32, false)]);
        let actual = Arc::new(Schema::new(vec![Field::new(
            "col",
            ArrowDataType::Int32,
            false,
        )]));
        let same = Error::NonMatchingSchemas { expected, actual };
        assert!(write!(FailOn("schemas"), "{same}").is_err());

        // failing while writing an added field propagates the error
        let expected = Schema::new(vec![Field::new("col", ArrowDataType::Int32, false)]);
        let actual = Arc::new(Schema::new(vec![
            Field::new("col", ArrowDataType::Int32, false),
            Field::new("extra", ArrowDataType::Int32, false),
        ]));
        let added = Error::NonMatchingSchemas { expected, actual };
        assert!(write!(FailOn("added"), "{added}").is_err());
    }
}
