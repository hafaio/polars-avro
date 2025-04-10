use std::io::Cursor;

use crate::{AvroScanner, Codec, WriteOptions, sink_avro};
use chrono::{NaiveDate, NaiveTime};
use polars::prelude::null::MutableNullArray;
use polars::prelude::{
    DataFrame, DataType, IntoLazy, Series, TimeUnit, UnionArgs, as_struct, col, concat,
    create_enum_dtype, df,
};
use polars_arrow::array::{MutableArray, Utf8ViewArray};
use polars_plan::plans::ScanSources;
use polars_utils::mmap::MemSlice;

fn serialize(frame: DataFrame, opts: WriteOptions) -> Vec<u8> {
    let mut buff = Cursor::new(Vec::new());
    sink_avro([frame], &mut buff, opts).unwrap();
    buff.into_inner()
}

fn deserialize(buff: Vec<u8>) -> DataFrame {
    let sources = ScanSources::Buffers(vec![MemSlice::from_vec(buff)].into_boxed_slice().into());
    let scanner = AvroScanner::new_from_sources(&sources, false, None).unwrap();
    let schema = scanner.schema();
    let iter = scanner.into_iter(2, None);
    let parts: Vec<_> = iter.map(|part| part.unwrap().lazy()).collect();
    if parts.is_empty() {
        DataFrame::empty_with_schema(&schema)
    } else {
        concat(parts, UnionArgs::default())
            .unwrap()
            .collect()
            .unwrap()
    }
}

#[test]
fn test_transitivity() {
    // create data
    let frame: DataFrame = df!(
        "name" => [Some("Alice Archer"), Some("Ben Brown"), Some("Chloe Cooper"), None],
        "weight" => [None, Some(72.5), Some(53.6), Some(83.1)],
        "height" => [Some(1.56_f32), None, Some(1.65_f32), Some(1.75_f32)],
        "birthtime" => [
            Some(NaiveDate::from_ymd_opt(1997, 1, 10).unwrap().and_hms_nano_opt(1, 2, 3, 1_002_003).unwrap()),
            Some(NaiveDate::from_ymd_opt(1985, 2, 15).unwrap().and_hms_nano_opt(4, 5, 6, 4_005_006).unwrap()),
            None,
            Some(NaiveDate::from_ymd_opt(1981, 4, 30).unwrap().and_hms_nano_opt(10, 11, 12, 10_011_012).unwrap()),
        ],
        "items" => [None, Some(Series::from_iter([Some("spoon"), None, Some("coin")])), Some(Series::from_iter([""; 0])), Some(Series::from_iter(["hat"]))],
        "good" => [Some(true), Some(false), None, Some(true)],
        "age" => [Some(10), None, Some(32), Some(97)],
        "income" => [Some(10_000_i64), None, Some(0_i64), Some(-42_i64)],
        "null" => Series::from_arrow("null".into(), MutableNullArray::new(4).as_box()).unwrap(),
        "codename" => [Some(&b"al1c3"[..]), Some(&b"b3n"[..]), Some(&b"chl03"[..]), None],
        "rating" => [None, Some("mid"), Some("slay"), Some("slay")],
    )
    .unwrap().lazy().with_columns([
        as_struct(vec![col("name"), col("age")]).alias("combined"),
        col("birthtime").strict_cast(DataType::Date).alias("birthdate"),
        col("birthtime").strict_cast(DataType::Datetime(TimeUnit::Milliseconds, None)).alias("birthtime_milli"),
        col("birthtime").strict_cast(DataType::Datetime(TimeUnit::Microseconds, None)).alias("birthtime_micro"),
        col("birthtime").strict_cast(DataType::Datetime(TimeUnit::Nanoseconds, None)).alias("birthtime_nano"),
        col("rating").strict_cast(create_enum_dtype(Utf8ViewArray::from_slice_values(["mid", "slay"]))),
    ]).collect().unwrap();

    // write / read
    let reconstruction = deserialize(serialize(
        frame.clone(),
        WriteOptions {
            codec: Codec::Null,
            promote_ints: false,
            promote_array: false,
            truncate_time: false,
        },
    ));

    assert_eq!(frame, reconstruction);
}

#[test]
fn test_promotion_truncation() {
    let frame: DataFrame = df!(
        "ints" => [10_u8, 54_u8, 32_u8, 97_u8],
        "arr" => [Series::from_iter([1, 2, 3]), Series::from_iter([4, 5, 6]), Series::from_iter([7, 8, 9]), Series::from_iter([10, 11, 12])],
        "time" => [
            NaiveTime::from_hms_nano_opt(1, 2, 3, 1_001).unwrap(),
            NaiveTime::from_hms_nano_opt(4, 5, 6, 2_002).unwrap(),
            NaiveTime::from_hms_nano_opt(7, 8, 9, 3_003).unwrap(),
            NaiveTime::from_hms_nano_opt(10, 11, 12, 4_004).unwrap(),
        ],
    )
    .unwrap().lazy().with_column(col("arr").strict_cast(DataType::Array(Box::new(DataType::Int32), 3))).collect().unwrap();

    let reconstruction = deserialize(serialize(
        frame.clone(),
        WriteOptions {
            codec: Codec::Null,
            promote_ints: true,
            promote_array: true,
            truncate_time: true,
        },
    ));

    let promoted = frame
        .lazy()
        .select([
            col("ints").strict_cast(DataType::Int32),
            col("arr").strict_cast(DataType::List(Box::new(DataType::Int32))),
            (col("time").cast(DataType::Int64) / 1000.into() * 1000.into()).cast(DataType::Time), // truncate to micro seconds
        ])
        .collect()
        .unwrap();

    assert_eq!(promoted, reconstruction);
}

#[test]
fn test_empty() {
    // create empty frame
    let frame: DataFrame = df!(
        "weight" => [0.0; 0],
    )
    .unwrap();

    // write / read
    let reconstruction = deserialize(serialize(
        frame.clone(),
        WriteOptions {
            codec: Codec::Null,
            promote_ints: false,
            promote_array: false,
            truncate_time: false,
        },
    ));

    assert_eq!(frame, reconstruction);
}
