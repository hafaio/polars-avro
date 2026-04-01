use super::{FullReadOptions, ReadOptions, Reader, Writer, get_schema};
use chrono::{NaiveDate, NaiveTime};
use polars::df;
use polars::prelude::null::MutableNullArray;
use polars::prelude::{
    self as pl, DataFrame, DataType, IntoLazy, IntoSeries, Null, Series, TimeUnit, TimeZone,
    UnionArgs,
};
use polars_arrow::array::MutableArray;
use std::convert::Infallible;
use std::io::Cursor;

#[allow(clippy::unnecessary_wraps)]
fn ok<T>(val: T) -> Result<T, Infallible> {
    Ok(val)
}

fn serialize(frame: &DataFrame) -> Vec<u8> {
    let mut buff = Cursor::new(Vec::new());
    Writer::try_new(&mut buff, frame.schema(), None)
        .unwrap()
        .write(frame)
        .unwrap();
    buff.into_inner()
}

fn deserialize(buff: Vec<u8>) -> DataFrame {
    let mut buff = Cursor::new(buff);
    let schema = get_schema(&mut buff).unwrap();
    buff.set_position(0);
    let iter = Reader::try_new(
        [ok(buff)],
        ReadOptions {
            batch_size: 2,
            ..FullReadOptions::default()
        },
    )
    .unwrap();
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

/// Compare two `DataFrame`s, treating enum and categorical as equivalent.
fn assert_frames_equal(left: &DataFrame, right: &DataFrame) {
    assert_eq!(left.shape(), right.shape(), "shape mismatch");
    for (l, r) in left.columns().iter().zip(right.columns()) {
        assert_eq!(l.name(), r.name(), "column name mismatch");
        // Cast enum/cat to string for comparison, leave others as-is
        let l_norm = normalize_column(l.as_materialized_series());
        let r_norm = normalize_column(r.as_materialized_series());
        assert_eq!(l_norm, r_norm, "column {:?} values differ", l.name());
    }
}

fn normalize_column(s: &Series) -> Series {
    match s.dtype() {
        DataType::Enum(_, _) | DataType::Categorical(_, _) => s.cast(&DataType::String).unwrap(),
        DataType::Struct(_) => {
            let ca = s.struct_().unwrap();
            let fields: Vec<Series> = ca.fields_as_series().iter().map(normalize_column).collect();
            pl::StructChunked::from_series(s.name().clone(), ca.len(), fields.iter())
                .unwrap()
                .into_series()
        }
        _ => s.clone(),
    }
}

macro_rules! test_transitivity {
    ($name:ident: $frame:expr) => {
        #[test]
        fn $name() {
            // create data
            let frame: DataFrame = $frame;

            // write / read
            let reconstruction = deserialize(serialize(&frame));

            // Avro enums deserialize as categorical, so compare by value
            assert_frames_equal(&frame, &reconstruction);
        }
    };
}

test_transitivity!(test_transitivity_null_string: df!(
        "name" => [Some("Alice Archer"), Some("Ben Brown"), Some("Chloe Cooper"), None],
    ).unwrap()
);

test_transitivity!(test_transitivity_complex: df!(
        "name" => [Some("Alice Archer"), Some("Ben Brown"), Some("Chloe Cooper"), None],
        "weight" => [None, Some(72.5), Some(53.6), None],
        "height" => [Some(1.56_f32), None, Some(1.65_f32), Some(1.75_f32)],
        "smol" => [Some(1_i8), None, Some(0_i8), Some(3_i8)],
        "daytime" => [
            Some(NaiveTime::from_hms_micro_opt(1, 1, 1, 1002).unwrap()),
            None,
            Some(NaiveTime::from_hms_micro_opt(2, 2, 2, 4005).unwrap()),
            Some(NaiveTime::from_hms_micro_opt(3, 3, 3, 7008).unwrap()),
        ],
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
        pl::col("height").strict_cast(DataType::Decimal(15, 2)).alias("decimal"),
    ]).collect().unwrap()
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

#[test]
fn test_empty() {
    let frame: DataFrame = df!(
        "weight" => [0.0; 0],
    )
    .unwrap();
    let reconstruction = deserialize(serialize(&frame));
    assert_eq!(frame, reconstruction);
}
