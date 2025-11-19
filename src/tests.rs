use std::io::Cursor;

use crate::{AvroScanner, Codec, Error, WriteOptions, sink_avro};
use chrono::{NaiveDate, NaiveTime};
use polars::df;
use polars::prelude::null::MutableNullArray;
use polars::prelude::{
    self as pl, DataFrame, DataType, FrozenCategories, IntoLazy, Null, Series, TimeUnit, TimeZone,
    UnionArgs,
};
use polars_arrow::array::MutableArray;

fn serialize(frame: DataFrame, opts: WriteOptions) -> Vec<u8> {
    let mut buff = Cursor::new(Vec::new());
    sink_avro([frame], &mut buff, opts).unwrap();
    buff.into_inner()
}

fn deserialize(buff: Vec<u8>) -> DataFrame {
    let scanner = AvroScanner::new([Cursor::new(buff)], None).unwrap();
    let schema = scanner.schema();
    let iter = scanner.into_iter(2, None);
    let parts: Vec<_> = iter.map(|part| part.unwrap().lazy()).collect();
    if parts.is_empty() {
        DataFrame::empty_with_schema(&schema)
    } else {
        pl::concat(parts, UnionArgs::default())
            .unwrap()
            .collect()
            .unwrap()
    }
}

macro_rules! test_transitivity {
    ($name:ident: $frame:expr) => {
        #[test]
        fn $name() {
            // create data
            let frame: DataFrame = $frame;

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
    };
}

test_transitivity!(test_transitivity_complex: df!(
        "name" => [Some("Alice Archer"), Some("Ben Brown"), Some("Chloe Cooper"), None],
        "weight" => [None, Some(72.5), Some(53.6), None],
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
        pl::when(pl::col("name") == "Alice Archer".into()).then(pl::lit(Null {})).otherwise(pl::as_struct(vec![pl::col("name"), pl::col("age")])).alias("combined"),
        pl::col("birthtime").strict_cast(DataType::Date).alias("birthdate"),
        pl::col("birthtime").strict_cast(DataType::Datetime(TimeUnit::Milliseconds, Some(TimeZone::UTC))).alias("birthtime_milli"),
        pl::col("birthtime").strict_cast(DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC))).alias("birthtime_micro"),
        pl::col("birthtime").strict_cast(DataType::Datetime(TimeUnit::Nanoseconds, Some(TimeZone::UTC))).alias("birthtime_nano"),
        pl::col("birthtime").strict_cast(DataType::Datetime(TimeUnit::Milliseconds, None)).alias("birthtime_milli_local"),
        pl::col("birthtime").strict_cast(DataType::Datetime(TimeUnit::Microseconds, None)).alias("birthtime_micro_local"),
        pl::col("birthtime").strict_cast(DataType::Datetime(TimeUnit::Nanoseconds, None)).alias("birthtime_nano_local"),
        pl::col("rating").strict_cast(DataType::from_frozen_categories(FrozenCategories::new(["mid", "slay"]).unwrap())),
        pl::col("height").strict_cast(DataType::Decimal(15, 2)).alias("decimal"),
    ]).collect().unwrap()
);

test_transitivity!(test_transitivity_enum: df! {
        "col" => [Some("a"), Some("b"), None],
    }
    .unwrap()
    .lazy()
    .select([
        pl::col("col").strict_cast(DataType::from_frozen_categories(FrozenCategories::new([
            "a", "b",
        ]).unwrap())),
    ])
    .collect()
    .unwrap()
);

test_transitivity!(test_transitivity_double_enum: df! {
        "one" => [Some("a"), Some("b"), None],
        "two" => [Some("c"), Some("d"), None],
    }
    .unwrap()
    .lazy()
    .select([
        pl::col("one").strict_cast(DataType::from_frozen_categories(FrozenCategories::new([
            "a", "b",
        ]).unwrap())),
        pl::col("two").strict_cast(DataType::from_frozen_categories(FrozenCategories::new([
            "d", "c",
        ]).unwrap())),
    ])
    .collect()
    .unwrap()
);

test_transitivity!(test_transitivity_double_decimal: df! {
        "one" => [Some("1.0"), Some("2.0"), None],
        "two" => [Some("234242342.1231"), Some("2342.12"), None],
    }
    .unwrap()
    .lazy()
    .select([
        pl::col("one").strict_cast(DataType::Decimal(3, 1)),
        pl::col("two").strict_cast(DataType::Decimal(16, 6)),
    ])
    .collect()
    .unwrap()
);

test_transitivity!(test_transitivity_double_struct: df! {
        "one" => [Some(1.0), Some(2.0), None],
        "two" => [Some("234242342.1231"), None, Some("2342.12")],
    }
    .unwrap()
    .lazy()
    .select([
        pl::as_struct(vec![pl::col("one"), pl::col("two")]).alias("first"),
        pl::as_struct(vec![pl::col("two"), pl::col("one")]).alias("second"),
    ])
    .collect()
    .unwrap()
);

test_transitivity!(test_transitivity_enum_struct: df! {
        "one" => ["a", "a", "b"],
        "two" => [1, 4, 3],
    }
    .unwrap()
    .lazy()
    .select([
        pl::as_struct(vec![pl::col("one").strict_cast(DataType::from_frozen_categories(FrozenCategories::new([
                "a", "b",
            ]).unwrap()
        )), pl::col("two")]).alias("col"),
    ])
    .collect()
    .unwrap()
);

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
    .unwrap().lazy().with_column(pl::col("arr").strict_cast(DataType::Array(Box::new(DataType::Int32), 3))).collect().unwrap();

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
            pl::col("ints").strict_cast(DataType::Int32),
            pl::col("arr").strict_cast(DataType::List(Box::new(DataType::Int32))),
            (pl::col("time").cast(DataType::Int64) / 1000.into() * 1000.into())
                .cast(DataType::Time), // truncate to micro seconds
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

#[test]
fn test_different_schemas() {
    let one = serialize(
        df! {
            "x" => [1, 2, 3]
        }
        .unwrap(),
        WriteOptions::default(),
    );
    let two = serialize(
        df! {
            "y" => ["a", "b", "c"]
        }
        .unwrap(),
        WriteOptions::default(),
    );

    let scanner = AvroScanner::new([Cursor::new(one), Cursor::new(two)], None).unwrap();
    let iter = scanner.into_iter(2, None);
    let parts: Result<Vec<_>, _> = iter.map(|part| part).collect();
    assert!(matches!(parts, Err(Error::NonMatchingSchemas)));
}
