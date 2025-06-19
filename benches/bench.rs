#![feature(test)]

extern crate test;

use std::io::Cursor;

use polars::prelude::{DataFrame, df};
use polars_avro::{AvroScanner, WriteOptions, sink_avro};
use polars_io::avro::{AvroReader, AvroWriter};
use polars_io::{SerReader, SerWriter};
use test::Bencher;

fn create_frame(num: i32) -> DataFrame {
    df!(
        "idx" => Vec::from_iter(0..num),
        "name" => Vec::from_iter((0..num).map(|v| v.to_string())),
    )
    .unwrap()
}

#[bench]
fn bench_write_polars_avro(b: &mut Bencher) {
    let frame: DataFrame = create_frame(1024);
    b.iter(|| {
        test::black_box(sink_avro(
            [frame.clone()],
            Vec::new(),
            WriteOptions::default(),
        ))
    });
}

#[bench]
fn bench_write_polars(b: &mut Bencher) {
    let frame: DataFrame = create_frame(1024);
    b.iter(|| {
        test::black_box(
            AvroWriter::new(&mut Vec::new())
                .with_name("".to_owned())
                .finish(&mut frame.clone()),
        )
    });
}

#[bench]
fn bench_read_polars_avro(b: &mut Bencher) {
    let frame: DataFrame = create_frame(1024);
    let mut buff = Vec::new();
    sink_avro([frame], &mut buff, WriteOptions::default()).unwrap();
    b.iter(|| {
        test::black_box(
            AvroScanner::new([Cursor::new(buff.clone())], None)
                .unwrap()
                .into_iter(1024, None)
                .collect::<Vec<_>>(),
        )
    });
}

#[bench]
fn bench_read_polars(b: &mut Bencher) {
    let frame: DataFrame = create_frame(1024);
    let mut buff = Vec::new();
    sink_avro([frame], &mut buff, WriteOptions::default()).unwrap();
    b.iter(|| test::black_box(AvroReader::new(Cursor::new(buff.clone())).finish()));
}
