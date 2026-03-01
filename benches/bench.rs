#![feature(test)]

extern crate test;

use polars::prelude::{self as pl, Column, DataFrame, IntoLazy, Series, df};
use polars_avro::{FullReadOptions, ReadOptions, Reader, Writer};
use polars_io::avro::{AvroReader, AvroWriter};
use polars_io::{SerReader, SerWriter};
use std::convert::Infallible;
use std::io::Cursor;
use test::Bencher;

fn create_narrow_frame(n: i32) -> DataFrame {
    df!(
        "idx" => Vec::from_iter(0..n),
        "name" => Vec::from_iter((0..n).map(|v| v.to_string())),
    )
    .unwrap()
}

fn create_wide_frame(n: i32) -> DataFrame {
    df!(
        "i32_col" => Vec::from_iter(0..n),
        "i64_col" => Vec::from_iter((0..n).map(|v| v as i64 * 1000)),
        "f64_col" => Vec::from_iter((0..n).map(|v| v as f64 * 0.01)),
        "f32_col" => Vec::from_iter((0..n).map(|v| v as f32 * 0.1)),
        "bool_col" => Vec::from_iter((0..n).map(|v| v % 2 == 0)),
        "str_col" => Vec::from_iter((0..n).map(|v| format!("val_{v}"))),
        "binary_col" => Vec::from_iter((0..n).map(|v| format!("bin_{v}").into_bytes())),
        "str2_col" => Vec::from_iter((0..n).map(|v| format!("extra_{v}"))),
    )
    .unwrap()
}

fn create_nested_frame(n: i32) -> DataFrame {
    df!(
        "idx" => Vec::from_iter(0..n),
        "name" => Vec::from_iter((0..n).map(|v| format!("name_{v}"))),
        "score" => Vec::from_iter((0..n).map(|v| v as f64 * 1.5)),
        "active" => Vec::from_iter((0..n).map(|v| v % 2 == 0)),
        "tags" => Vec::from_iter((0..n).map(|v| {
            Series::from_iter((0..v % 4).map(|t| format!("tag_{t}")))
        })),
        "rating" => Vec::from_iter((0..n).map(|v| if v % 3 == 0 { "good" } else if v % 3 == 1 { "mid" } else { "bad" })),
        "notes" => Vec::from_iter((0..n).map(|v| format!("note_{v}"))),
    )
    .unwrap()
    .lazy()
    .with_column(
        pl::as_struct(vec![
            pl::col("name"),
            pl::as_struct(vec![
                pl::col("tags"),
                pl::col("score")
            ])
        ]).alias("info"),
    )
    .collect()
    .unwrap()
}

fn create_mega_wide_frame(n: i32) -> DataFrame {
    let base = create_wide_frame(n);
    let mut columns: Vec<Column> = Vec::with_capacity(base.width() * 16);
    for g in 0..16 {
        for series in base.materialized_column_iter() {
            let mut s = series.clone();
            s.rename(format!("{}_{g}", series.name()).into());
            columns.push(s.into());
        }
    }
    DataFrame::new(n as usize, columns).unwrap()
}

fn create_complex_frame(n: i32) -> DataFrame {
    df!(
        "idx" => Vec::from_iter(0..n),
        "name" => Vec::from_iter((0..n).map(|v| format!("name_{v}"))),
        "score" => Vec::from_iter((0..n).map(|v| v as f64 * 1.5)),
        "active" => Vec::from_iter((0..n).map(|v| v % 2 == 0)),
        "tags" => Vec::from_iter((0..n).map(|v| {
            Series::from_iter((0..v % 5).map(|t| format!("tag_{t}")))
        })),
        "key" => Vec::from_iter((0..n).map(|v| format!("key_{v}"))),
        "value" => Vec::from_iter(0..n),
        "created_i32" => Vec::from_iter(0..n),
        "notes" => Vec::from_iter((0..n).map(|v| format!("note_{v}"))),
    )
    .unwrap()
    .lazy()
    .with_columns([
        pl::as_struct(vec![pl::col("key"), pl::col("value")]).alias("metadata"),
        pl::col("created_i32")
            .cast(pl::DataType::Date)
            .alias("created"),
    ])
    .select([
        pl::col("idx"),
        pl::col("name"),
        pl::col("score"),
        pl::col("active"),
        pl::col("tags"),
        pl::col("metadata"),
        pl::col("created"),
        pl::col("notes"),
    ])
    .collect()
    .unwrap()
}

macro_rules! bench_shape {
    ($name:ident, $frame_expr:expr) => {
        mod $name {
            use super::*;

            #[bench]
            fn write_polars_avro(b: &mut Bencher) {
                let frame: DataFrame = $frame_expr;
                b.iter(|| {
                    test::black_box(
                        Writer::try_new(Vec::new(), frame.schema(), None)
                            .unwrap()
                            .write(&frame)
                            .unwrap(),
                    )
                });
            }

            #[bench]
            fn write_polars_native(b: &mut Bencher) {
                let frame: DataFrame = $frame_expr;
                b.iter(|| {
                    test::black_box(
                        AvroWriter::new(&mut Vec::new())
                            .with_name("".to_owned())
                            .finish(&mut frame.clone()),
                    )
                });
            }

            #[bench]
            fn read_polars_avro(b: &mut Bencher) {
                let frame: DataFrame = $frame_expr;
                let mut buff: Vec<u8> = Vec::new();
                Writer::try_new(&mut buff, frame.schema(), None)
                    .unwrap()
                    .write(&frame)
                    .unwrap();
                b.iter(|| {
                    test::black_box(
                        Reader::try_new(
                            [Ok::<_, Infallible>(Cursor::new(buff.clone()))],
                            FullReadOptions::default(),
                        )
                        .unwrap()
                        .collect::<Vec<_>>(),
                    )
                });
            }

            #[bench]
            fn read_polars_native(b: &mut Bencher) {
                let frame: DataFrame = $frame_expr;
                let mut buff: Vec<u8> = Vec::new();
                Writer::try_new(&mut buff, frame.schema(), None)
                    .unwrap()
                    .write(&frame)
                    .unwrap();
                b.iter(|| test::black_box(AvroReader::new(Cursor::new(buff.clone())).finish()));
            }
        }
    };
}

macro_rules! bench_shape_avro_only {
    ($name:ident, $frame_expr:expr) => {
        mod $name {
            use super::*;

            #[bench]
            fn write_polars_avro(b: &mut Bencher) {
                let frame: DataFrame = $frame_expr;
                b.iter(|| {
                    test::black_box(
                        Writer::try_new(Vec::new(), frame.schema(), None)
                            .unwrap()
                            .write(&frame)
                            .unwrap(),
                    )
                });
            }

            #[bench]
            fn read_polars_avro(b: &mut Bencher) {
                let frame: DataFrame = $frame_expr;
                let mut buff: Vec<u8> = Vec::new();
                Writer::try_new(&mut buff, frame.schema(), None)
                    .unwrap()
                    .write(&frame)
                    .unwrap();
                b.iter(|| {
                    test::black_box(
                        Reader::try_new(
                            [Ok::<_, Infallible>(Cursor::new(buff.clone()))],
                            FullReadOptions::default(),
                        )
                        .unwrap()
                        .collect::<Vec<_>>(),
                    )
                });
            }
        }
    };
}

macro_rules! bench_projection {
    ($name:ident, $frame_expr:expr, [$($col:literal),+ $(,)?]) => {
        mod $name {
            use super::*;

            #[bench]
            fn read_polars_avro(b: &mut Bencher) {
                let frame: DataFrame = $frame_expr;
                let mut buff: Vec<u8> = Vec::new();
                Writer::try_new(&mut buff, frame.schema(), None)
                    .unwrap()
                    .write(&frame)
                    .unwrap();
                let columns: Vec<&str> = vec![$($col),+];
                b.iter(|| {
                    test::black_box(
                        Reader::try_new(
                            [Ok::<_, Infallible>(Cursor::new(buff.clone()))],
                            ReadOptions {
                                projection: Some(&columns[..]),
                                ..ReadOptions::default()
                            },
                        )
                        .unwrap()
                        .collect::<Vec<_>>(),
                    )
                });
            }

            #[bench]
            fn read_polars_native(b: &mut Bencher) {
                let frame: DataFrame = $frame_expr;
                let mut buff: Vec<u8> = Vec::new();
                Writer::try_new(&mut buff, frame.schema(), None)
                    .unwrap()
                    .write(&frame)
                    .unwrap();
                let columns: Vec<String> = vec![$($col.to_owned()),+];
                b.iter(|| {
                    test::black_box(
                        AvroReader::new(Cursor::new(buff.clone()))
                            .with_columns(Some(columns.clone()))
                            .finish(),
                    )
                });
            }
        }
    };
}

macro_rules! bench_read_options {
    ($name:ident, $frame_expr:expr, $($field:ident : $value:expr),+ $(,)?) => {
        mod $name {
            use super::*;

            #[bench]
            fn read_polars_avro(b: &mut Bencher) {
                let frame: DataFrame = $frame_expr;
                let mut buff: Vec<u8> = Vec::new();
                Writer::try_new(&mut buff, frame.schema(), None)
                    .unwrap()
                    .write(&frame)
                    .unwrap();
                b.iter(|| {
                    test::black_box(
                        Reader::try_new(
                            [Ok::<_, Infallible>(Cursor::new(buff.clone()))],
                            ReadOptions {
                                $($field: $value,)*
                                ..FullReadOptions::default()
                            },
                        )
                        .unwrap()
                        .collect::<Vec<_>>(),
                    )
                });
            }
        }
    };
}

bench_shape!(narrow_small, create_narrow_frame(16));
bench_shape!(narrow_medium, create_narrow_frame(1024));
bench_shape!(narrow_long, create_narrow_frame(65536));
bench_shape!(wide, create_wide_frame(1024));
bench_shape!(large, create_wide_frame(1_048_576));
bench_shape!(mega_wide, create_mega_wide_frame(1_048_576));
bench_shape_avro_only!(nested, create_nested_frame(1024));
bench_shape_avro_only!(complex, create_complex_frame(1_048_576));

bench_read_options!(narrow_medium_strict, create_narrow_frame(1024), strict: true);
bench_read_options!(narrow_medium_utf8_view, create_narrow_frame(1024), utf8_view: true);

bench_projection!(
    wide_project_two,
    create_wide_frame(1024),
    ["i32_col", "str_col"]
);
bench_projection!(
    wide_project_five,
    create_wide_frame(1024),
    ["i32_col", "f64_col", "bool_col", "str_col", "binary_col"]
);
bench_projection!(
    mega_wide_project_eight,
    create_mega_wide_frame(1_048_576),
    [
        "i32_col_0",
        "i64_col_0",
        "f64_col_0",
        "f32_col_0",
        "bool_col_0",
        "str_col_0",
        "binary_col_0",
        "str2_col_0"
    ]
);
