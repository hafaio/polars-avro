//! Rust sink implementation
use super::Error;
use super::ffi;
use arrow::datatypes::Schema as ArrowSchema;
use arrow_avro::compression::CompressionCodec;
use arrow_avro::writer::{AvroWriter, WriterBuilder};
use polars::prelude::{DataFrame, Schema};
use std::io::Write;

/// Incrementally write avro files.
///
/// Some polars types (`Int8`, `Int16`, `UInt8`, `UInt16`, `UInt32`, `UInt64`,
/// `Time`, `Categorical`, `Enum`) can't be written directly and must be cast
/// first — see the README for workarounds.
pub struct Writer<W: Write> {
    base: AvroWriter<W>,
    schema: ArrowSchema,
}

impl<W: Write> Writer<W> {
    /// Create a writer with a schema and options
    ///
    /// # Errors
    /// If the schema can't be converted, or the writer can't be created
    pub fn try_new(
        writer: W,
        schema: &Schema,
        codec: Option<CompressionCodec>,
    ) -> Result<Self, Error> {
        let schema = ffi::polars_schema_to_arrow(schema)?;
        let base: AvroWriter<_> = WriterBuilder::new(schema.clone())
            .with_compression(codec)
            .build(writer)?;
        Ok(Writer { base, schema })
    }

    /// Finish writing and return the underlying writer.
    ///
    /// # Errors
    /// If there were problems flushing the writer
    pub fn into_inner(mut self) -> Result<W, Error> {
        self.base.finish()?;
        Ok(self.base.into_inner())
    }

    /// Write a single dataframe chunk
    ///
    /// # Errors
    /// If there were problems writing the batch, or the frame doesn't match the schema
    pub fn write(&mut self, batch: &DataFrame) -> Result<(), Error> {
        let batch = ffi::dataframe_to_recordbatch(batch)?;
        if *batch.schema() == self.schema {
            Ok(self.base.write(&batch)?)
        } else {
            Err(Error::NonMatchingSchemas {
                expected: self.schema.clone(),
                actual: batch.schema(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Error;
    use super::Writer;
    use apache_avro::Reader;
    use apache_avro::types::Value;
    use arrow_avro::errors::AvroError;
    use chrono::NaiveTime;
    use polars::df;
    use polars::prelude::{
        self as pl, Categories, DataType, FrozenCategories, IntoLazy, LazyFrame, Series,
    };
    use std::mem;

    fn bad_frame() -> LazyFrame {
        df! {
            "cat" => ["a", "b", "a"],
        }
        .unwrap()
        .lazy()
        .with_columns([
            pl::col("cat").strict_cast(DataType::from_categories(Categories::global())),
            pl::col("cat")
                .strict_cast(DataType::from_frozen_categories(
                    FrozenCategories::new(["a", "b"]).unwrap(),
                ))
                .alias("enum"),
        ])
    }

    #[test]
    fn test_empty() {
        let ex = df! {
            "col" => [1, 2, 3],
        }
        .unwrap();
        let mut dest = Vec::new();
        mem::drop(Writer::try_new(&mut dest, ex.schema(), None).unwrap());
        let written = dest.len();
        assert_ne!(written, 0);
    }

    #[test]
    fn test_write_array() {
        let batch = df! {
                "array" => [Series::from_iter([1, 2, 3]), Series::from_iter([4, 5, 6]), Series::from_iter([7, 8, 9]), Series::from_iter([10, 11, 12])],
            }
            .unwrap()
            .lazy().with_column(pl::col("array").strict_cast(DataType::Array(Box::new(DataType::Int32), 3))).collect().unwrap();
        Writer::try_new(Vec::new(), batch.schema(), None)
            .unwrap()
            .write(&batch)
            .unwrap();
    }

    #[test]
    fn test_write_record() {
        let batch = df! {
            "int" => [1, 2, 3],
            "str" => ["a", "b", "c"],
        }
        .unwrap()
        .lazy()
        .select([pl::as_struct(vec![pl::col("int"), pl::col("str")])])
        .collect()
        .unwrap();
        Writer::try_new(Vec::new(), batch.schema(), None)
            .unwrap()
            .write(&batch)
            .unwrap();
    }

    #[test]
    fn test_bad_frame_mitigations() {
        let batch = bad_frame()
            .with_columns([
                pl::col("cat").strict_cast(DataType::Int32),
                pl::col("enum").strict_cast(DataType::Int32),
            ])
            .collect()
            .unwrap();
        Writer::try_new(Vec::new(), batch.schema(), None)
            .unwrap()
            .write(&batch)
            .unwrap();
    }

    #[test]
    fn test_write_time() {
        let batch = df! {
            "time" => [
                NaiveTime::from_hms_opt(1, 2, 3).unwrap(),
                NaiveTime::from_hms_opt(4, 5, 6).unwrap(),
                NaiveTime::from_hms_opt(7, 8, 9).unwrap(),
                NaiveTime::from_hms_opt(10, 11, 12).unwrap(),
            ],
        }
        .unwrap();
        Writer::try_new(Vec::new(), batch.schema(), None)
            .unwrap()
            .write(&batch)
            .unwrap();
    }

    #[test]
    fn test_time_truncation() {
        let batch = df! {
            "time" => [
                NaiveTime::from_hms_nano_opt(1, 2, 3, 1_002_003).unwrap(),
            ],
        }
        .unwrap();
        let mut buff = Vec::new();
        Writer::try_new(&mut buff, batch.schema(), None)
            .unwrap()
            .write(&batch)
            .unwrap();
        let row = Reader::new(&*buff).unwrap().next().unwrap().unwrap();
        let Value::Record(fields) = row else {
            panic!("expected record");
        };
        let [(_, Value::Union(1, time))] = &*fields else {
            panic!("expected singleton");
        };
        let Value::TimeMicros(val) = **time else {
            panic!("expected time");
        };
        // truncated to remove nanos
        assert_eq!(val, 3723001002);
    }

    #[test]
    fn test_write_ints() {
        let batch = df! {
            "int8" => [1_i8, 2_i8, 3_i8],
            "int16" => [4_i16, 5_i16, 6_i16],
            "int32" => [7_i32, 8_i32, 9_i32],
            "int64" => [10_i64, 11_i64, 12_i64],
            "uint8" => [13_u8, 14_u8, 15_u8],
            "uint16" => [16_u16, 17_u16, 18_u16],
            "uint32" => [19_u32, 20_u32, 21_u32],
            "uint64" => [22_u64, 23_u64, 24_u64],
        }
        .unwrap();
        Writer::try_new(Vec::new(), batch.schema(), None)
            .unwrap()
            .write(&batch)
            .unwrap();
    }

    #[test]
    fn test_diff_schemas_error() {
        let first = df! {
            "a" => [1, 2, 3],
        }
        .unwrap();
        let mut writer = Writer::try_new(Vec::new(), first.schema(), None).unwrap();
        writer.write(&first).unwrap();
        let second = df! {
            "b" => ["a", "b", "c"],
        }
        .unwrap();
        let err = writer.write(&second).unwrap_err();
        assert!(matches!(err, Error::NonMatchingSchemas { .. }));
    }

    #[test]
    fn test_bad_frame_error() {
        let batch = bad_frame().collect().unwrap();
        let err = Writer::try_new(Vec::new(), batch.schema(), None)
            .unwrap()
            .write(&batch)
            .unwrap_err();
        assert!(matches!(err, Error::ArrowAvro(_)));
    }

    #[test]
    fn test_large_uint64_error() {
        let batch = df! {
            "uint64" => [u64::MAX],
        }
        .unwrap();
        let err = Writer::try_new(Vec::new(), batch.schema(), None)
            .unwrap()
            .write(&batch)
            .unwrap_err();
        assert!(matches!(
            err,
            Error::ArrowAvro(AvroError::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_categorical_error() {
        let batch = df! {
            "cat" => ["a", "b"],
        }
        .unwrap()
        .lazy()
        .with_column(pl::col("cat").strict_cast(DataType::from_categories(Categories::global())))
        .collect()
        .unwrap();
        let err = Writer::try_new(Vec::new(), batch.schema(), None)
            .unwrap()
            .write(&batch)
            .unwrap_err();
        assert!(matches!(err, Error::ArrowAvro(_)));
    }

    #[test]
    fn test_enum_error() {
        let batch = df! {
            "enum" => ["a", "b"],
        }
        .unwrap()
        .lazy()
        .with_column(
            pl::col("enum").strict_cast(DataType::from_frozen_categories(
                FrozenCategories::new(["a", "b"]).unwrap(),
            )),
        )
        .collect()
        .unwrap();
        let err = Writer::try_new(Vec::new(), batch.schema(), None)
            .unwrap()
            .write(&batch)
            .unwrap_err();
        assert!(matches!(err, Error::ArrowAvro(_)));
    }
}
