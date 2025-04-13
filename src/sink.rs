//! Rust sink implementation
use std::io::Write;

use super::Error;
use super::ser;
use apache_avro::Writer;
use polars::frame::DataFrame;
use polars::frame::row::Row;
use polars::prelude::AnyValue;

use apache_avro::Codec;

/// Options for writing an avro file
#[derive(Debug, Clone, Copy)]
pub struct WriteOptions {
    /// Code to write with
    pub codec: Codec,
    /// If true, promote ints to make them writable to avro
    pub promote_ints: bool,
    /// If true, promote array to a list to make it writable to avro
    pub promote_array: bool,
    /// If true, truncate time so that it can be written to avro
    pub truncate_time: bool,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            codec: Codec::Null,
            promote_ints: true,
            promote_array: true,
            truncate_time: false,
        }
    }
}

/// Sink `DataFrame` chunks into something `Write`-able
///
/// # Errors
///
/// If the schema can't be written as an avro file, or there are other io errors
/// directly from avro
pub fn sink_avro(
    chunks: impl IntoIterator<Item = DataFrame>,
    dest: impl Write,
    options: WriteOptions,
) -> Result<(), Error> {
    let mut chunks_iter = chunks.into_iter().peekable();
    if let Some(first) = chunks_iter.peek() {
        // polars schema
        let schema = first.schema().clone();
        // get avro schema
        let avro_schema = ser::try_as_schema(
            &schema,
            options.promote_ints,
            options.promote_array,
            options.truncate_time,
        )?;

        let mut writer = Writer::with_codec(&avro_schema, dest, options.codec);
        for part in chunks_iter {
            if part.schema() == &schema {
                let mut row = Row(vec![AnyValue::Null; part.width()]);
                for idx in 0..part.height() {
                    part.get_row_amortized(idx, &mut row)?;
                    let Row(record) = &row;
                    writer.append(ser::try_as_value(&schema, record)?)?;
                }
            } else {
                return Err(Error::NonMatchingSchemas);
            }
        }
        // flush does not write header, but into_inner does
        writer.into_inner()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use chrono::NaiveTime;
    use polars::df;
    use polars::prelude::{self as pl, DataType, IntoLazy, Series};
    use std::io::Cursor;

    use super::WriteOptions;
    use crate::Error;

    #[test]
    fn test_empty() {
        let dest = Cursor::new(Vec::new());
        super::sink_avro([], dest, WriteOptions::default()).unwrap();
    }

    #[test]
    fn test_diff_schemas() {
        let dest = Cursor::new(Vec::new());
        let res = super::sink_avro(
            [
                df! {
                    "a" => [1, 2, 3],
                }
                .unwrap(),
                df! {
                    "b" => ["a", "b", "c"],
                }
                .unwrap(),
            ],
            dest,
            WriteOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(res, Error::NonMatchingSchemas));
    }

    /// since types are upcase to i32s, these are still compatable
    #[test]
    fn test_promote_error() {
        let dest = Cursor::new(Vec::new());
        let res = super::sink_avro(
            [df! {
                "a" => [4_i8, 5_i8, 6_i8],
            }
            .unwrap()],
            dest,
            WriteOptions {
                promote_ints: false,
                ..WriteOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(res, Error::UnsupportedPolarsType(DataType::Int8)));
    }

    #[test]
    fn test_promote_ints() {
        let dest = Cursor::new(Vec::new());
        super::sink_avro(
            [df! {
                "a" => [4_i8, 5_i8, 6_i8],
                "b" => [4_i16, 5_i16, 6_i16],
                "c" => [4_u8, 5_u8, 6_u8],
                "d" => [4_u16, 5_u16, 6_u16],
                "e" => [4_u32, 5_u32, 6_u32],
            }
            .unwrap()],
            dest,
            WriteOptions {
                promote_ints: true,
                ..WriteOptions::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn test_promote_array() {
        let dest = Cursor::new(Vec::new());
        super::sink_avro(
            [df! {
                "a" => [Series::from_iter([1, 2, 3]), Series::from_iter([4, 5, 6]), Series::from_iter([7, 8, 9]), Series::from_iter([10, 11, 12])],
            }
            .unwrap()
            .lazy().with_column(pl::col("a").strict_cast(DataType::Array(Box::new(DataType::Int32), 3))).collect().unwrap()],
            dest,
            WriteOptions {
                promote_array: true,
                ..WriteOptions::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn test_truncate_time() {
        let dest = Cursor::new(Vec::new());
        super::sink_avro(
            [df! {
                "a" => [
                    NaiveTime::from_hms_opt(1, 2, 3).unwrap(),
                    NaiveTime::from_hms_opt(4, 5, 6).unwrap(),
                    NaiveTime::from_hms_opt(7, 8, 9).unwrap(),
                    NaiveTime::from_hms_opt(10, 11, 12).unwrap(),
                ],
            }
            .unwrap()],
            dest,
            WriteOptions {
                truncate_time: true,
                ..WriteOptions::default()
            },
        )
        .unwrap();
    }
}
