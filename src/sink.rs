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
    use chrono::NaiveTime;
    use polars::df;
    use polars::prelude::{
        self as pl, Categories, DataType, FrozenCategories, IntoLazy, LazyFrame, Series,
    };
    use std::mem;

    fn bad_frame() -> LazyFrame {
        df! {
            "int" => [4_i8, 5_i8, 6_i8],
            "time" => [
                NaiveTime::from_hms_opt(1, 2, 3).unwrap(),
                NaiveTime::from_hms_opt(4, 5, 6).unwrap(),
                NaiveTime::from_hms_opt(7, 8, 9).unwrap(),
            ],
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
                pl::col("int").strict_cast(DataType::Int32),
                pl::col("time").strict_cast(DataType::Int64),
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
        assert!(matches!(err, Error::Arrow(_)));
    }

    macro_rules! test_int_write_success {
        ($name:ident, $type:ty) => {
            #[test]
            fn $name() {
                let batch = df! { "int" => [4 as $type, 5 as $type, 6 as $type] }.unwrap();
                Writer::try_new(Vec::new(), batch.schema(), None)
                    .unwrap()
                    .write(&batch)
                    .unwrap();
            }
        };
    }

    macro_rules! test_int_write_error {
        ($name:ident, $type:ty) => {
            #[test]
            fn $name() {
                let batch = df! { "int" => [4 as $type, 5 as $type, 6 as $type] }.unwrap();
                let err = Writer::try_new(Vec::new(), batch.schema(), None)
                    .unwrap()
                    .write(&batch)
                    .unwrap_err();
                assert!(matches!(err, Error::Arrow(_)));
            }
        };
    }

    test_int_write_success!(test_int32_success, i32);
    test_int_write_success!(test_int64_success, i64);

    test_int_write_error!(test_int8_error, i8);
    test_int_write_error!(test_int16_error, i16);
    test_int_write_error!(test_uint8_error, u8);
    test_int_write_error!(test_uint16_error, u16);
    test_int_write_error!(test_uint32_error, u32);
    test_int_write_error!(test_uint64_error, u64);

    #[test]
    fn test_time_error() {
        let batch = df! {
            "time" => [
                NaiveTime::from_hms_opt(1, 2, 3).unwrap(),
                NaiveTime::from_hms_opt(4, 5, 6).unwrap(),
                NaiveTime::from_hms_opt(7, 8, 9).unwrap(),
                NaiveTime::from_hms_opt(10, 11, 12).unwrap(),
            ],
        }
        .unwrap();
        let err = Writer::try_new(Vec::new(), batch.schema(), None)
            .unwrap()
            .write(&batch)
            .unwrap_err();
        assert!(matches!(err, Error::Arrow(_)));
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
        assert!(matches!(err, Error::Arrow(_)));
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
        assert!(matches!(err, Error::Arrow(_)));
    }
}
