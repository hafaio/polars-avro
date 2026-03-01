//! Rust scan implementation

use super::Error;
use super::Projection;
use super::ffi;
use apache_avro::Reader as AvroReader;
use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use arrow_avro::reader::{Reader as ArrowAvroReader, ReaderBuilder};
use arrow_avro::schema::AvroSchema;
use polars::frame::DataFrame;
use std::convert::Infallible;
use std::io::{BufReader, Read, Seek};
use std::iter::FusedIterator;
use std::sync::Arc;

/// Configuration options for the avro reader
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadOptions<P = Infallible> {
    /// If we should use strict parsing. Incurs a performance hit.
    pub strict: bool,
    /// If strings should be read in as views instead of character arrays.
    ///
    /// This affects UUID and nullable string handling — see the README for details.
    /// Since polars tends to work with string views internally, enabling this is
    /// likely faster if you don't mind losing null string distinctions.
    pub utf8_view: bool,
    /// The batch size for reading
    pub batch_size: usize,
    /// The columns to select
    pub projection: Option<P>,
}

/// Read options without projection
pub type FullReadOptions = ReadOptions<Infallible>;

impl<P> Default for ReadOptions<P> {
    fn default() -> Self {
        Self {
            strict: false,
            utf8_view: false,
            batch_size: 1024,
            projection: None,
        }
    }
}

impl<P: Projection> ReadOptions<P> {
    fn create_reader<R: Read + Seek>(
        &self,
        reader: R,
    ) -> Result<ArrowAvroReader<BufReader<R>>, Error> {
        let mut builder = ReaderBuilder::new()
            .with_utf8_view(self.utf8_view)
            .with_strict_mode(self.strict)
            .with_batch_size(self.batch_size);
        let mut buf_reader = BufReader::new(reader);
        if let Some(proj) = &self.projection {
            // To do a projection we need to supply a schema, but arrow-avro
            // doesn't keep the metadata necessary for ensuing a match
            let orig = buf_reader.stream_position()?;
            let projected = {
                let reader = AvroReader::new(&mut buf_reader)?;
                proj.project(reader.writer_schema())?
            };
            // we use seek_relative to avoid flishing the buffer
            let cur = buf_reader.stream_position()?;
            let seek = orig.checked_signed_diff(cur).ok_or(Error::LargeHeader)?;
            buf_reader.seek_relative(seek)?;
            builder = builder.with_reader_schema(AvroSchema::new(projected.canonical_form()));
        }
        Ok(builder.build(buf_reader)?)
    }
}

/// An iterator that yields [`DataFrame`]s from one or more avro sources.
///
/// All sources must share the same schema; a [`Error::NonMatchingSchemas`]
/// error is returned if they differ.
#[derive(Debug)]
pub struct Reader<R: Read, I, C> {
    sources: I,
    source: ArrowAvroReader<BufReader<R>>,
    options: ReadOptions<C>,
    schema: Arc<Schema>,
}

impl<R, E, I, P> Reader<R, I, P>
where
    R: Read + Seek,
    Error: From<E>,
    I: Iterator<Item = Result<R, E>>,
    P: Projection,
{
    /// Create a new iterator from sources and a config
    ///
    /// # Errors
    /// If sources is empty, or is a problem creating a reader from the first
    /// source
    pub fn try_new(
        sources: impl IntoIterator<IntoIter = I>,
        config: ReadOptions<P>,
    ) -> Result<Self, Error> {
        let mut sources = sources.into_iter();
        let first = sources.next().ok_or(Error::EmptySources)??;
        let source = config.create_reader(first)?;
        let schema = source.schema();
        Ok(Self {
            sources,
            source,
            options: config,
            schema,
        })
    }

    fn matched_schema(&self, batch: &RecordBatch) -> bool {
        self.options.projection.is_some() || batch.schema() == self.schema
    }
}

impl<R, E, I, P> Iterator for Reader<R, I, P>
where
    R: Read + Seek,
    Error: From<E>,
    I: Iterator<Item = Result<R, E>>,
    P: Projection,
{
    type Item = Result<DataFrame, Error>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.source.next() {
                Some(Ok(batch)) if self.matched_schema(&batch) => {
                    return Some(ffi::recordbatch_to_dataframe(&batch));
                }
                Some(Ok(batch)) => {
                    return Some(Err(Error::NonMatchingSchemas {
                        expected: (*self.schema).clone(),
                        actual: batch.schema(),
                    }));
                }
                Some(Err(e)) => return Some(Err(e.into())),
                None => match self.sources.next() {
                    Some(Ok(source)) => {
                        self.source = match self.options.create_reader(source) {
                            Ok(reader) => reader,
                            Err(e) => return Some(Err(e)),
                        };
                    }
                    Some(Err(e)) => return Some(Err(e.into())),
                    None => return None,
                },
            }
        }
    }
}

impl<R, E, I, P> FusedIterator for Reader<R, I, P>
where
    R: Read + Seek,
    Error: From<E>,
    I: Iterator<Item = Result<R, E>> + FusedIterator,
    P: Projection,
{
}

#[cfg(test)]
mod tests {
    use super::{Error, FullReadOptions, Projection, ReadOptions, Reader};
    use apache_avro::schema::ArraySchema;
    use apache_avro::schema::EnumSchema;
    use apache_avro::schema::MapSchema;
    use apache_avro::schema::UnionSchema;
    use apache_avro::schema::{DecimalSchema, FixedSchema, RecordField, RecordSchema, Schema};
    use apache_avro::types::{Record, Value};
    use apache_avro::{
        AvroResult, Days, Decimal as AvroDecimal, Duration as AvroDuration, Millis, Months, Writer,
    };
    use bigdecimal::BigDecimal;
    use polars::df;
    use polars::frame::DataFrame;
    use polars::prelude::PlSmallStr;
    use polars::prelude::SortOptions;
    use polars::prelude::StructChunked;
    use polars::prelude::{
        self as pl, Categories, DataType, IntoLazy, NamedFrom, Series, TimeUnit, TimeZone,
        UnionArgs,
    };
    use std::collections::BTreeMap;
    use std::collections::HashMap;
    use std::convert::Infallible;
    use std::fs::File;
    use std::io::Cursor;
    use std::io::Read;
    use std::io::Seek;
    use std::mem;
    use uuid::Uuid;

    #[allow(clippy::unnecessary_wraps)]
    fn ok<T>(val: T) -> Result<T, Infallible> {
        Ok(val)
    }

    fn read_scan<R: Read + Seek, E>(
        scanner: Reader<R, impl Iterator<Item = Result<R, E>>, impl Projection>,
    ) -> DataFrame
    where
        Error: From<E>,
    {
        let frames: Vec<_> = scanner.map(|part| part.unwrap().lazy()).collect();
        pl::concat(frames, UnionArgs::default())
            .unwrap()
            .collect()
            .unwrap()
    }

    fn map_struct<'a, V>(
        name: impl Into<PlSmallStr>,
        keys: impl IntoIterator<Item = &'a str>,
        values: impl IntoIterator<Item = V>,
    ) -> Series
    where
        Series: NamedFrom<Vec<&'a str>, [&'a str]> + NamedFrom<Vec<V>, [V]>,
    {
        let keys: Vec<&str> = keys.into_iter().collect();
        let len = keys.len();
        let values: Vec<V> = values.into_iter().collect();
        let key_series = Series::new("key".into(), keys);
        let value_series = Series::new("value".into(), values);
        Series::from(
            StructChunked::from_series(name.into(), len, [key_series, value_series].iter())
                .unwrap(),
        )
    }

    fn write_avro(
        name: &str,
        dtype: Schema,
        vals: impl IntoIterator<Item = impl Into<Value>>,
    ) -> AvroResult<Cursor<Vec<u8>>> {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::Record(
            RecordSchema::builder()
                .name("base".into())
                .fields(vec![
                    RecordField::builder()
                        .name(name.into())
                        .schema(dtype)
                        .build(),
                ])
                .lookup([(name.into(), 0)].into())
                .build(),
        );
        let mut writer = Writer::new(&schema, &mut buff);
        for val in vals {
            let mut first = Record::new(&schema).unwrap();
            first.put(name, val);
            writer.append(first)?;
        }
        writer.flush()?;
        mem::drop(writer);
        buff.set_position(0);
        Ok(buff)
    }

    /// Test scan on a simple file
    #[test]
    fn test_scan() {
        let batches = Reader::try_new(
            [File::open("./resources/food.avro")],
            FullReadOptions::default(),
        )
        .unwrap();
        let frame = read_scan(batches);
        assert_eq!(frame.height(), 27);
        assert_eq!(frame.schema().len(), 4);
    }

    /// Test scan on a simple file
    #[test]
    fn test_reorder() {
        let columns = ["sugars_g", "calories"];
        let batches = Reader::try_new(
            [File::open("./resources/food.avro")],
            ReadOptions {
                projection: Some(&columns[..]),
                ..ReadOptions::default()
            },
        )
        .unwrap();
        let frame = read_scan(batches);
        assert_eq!(frame.get_column_names(), columns);
    }

    macro_rules! test_read_type {
        ($(#[$attr:meta])* $name:ident: $col:expr, $schema:expr, $val:expr, $expected:expr $(, $field:ident : $value:expr)* $(,)?) => {
            $(#[$attr])*
            #[test]
            fn $name() {
                let buff = write_avro($col, $schema, $val).unwrap();
                let res = Reader::try_new([ok(buff)], ReadOptions { $($field: $value,)* ..FullReadOptions::default() }).unwrap();
                let frame = read_scan(res);
                let expected: Series = $expected;
                let actual = frame.column($col).unwrap().as_series().unwrap();
                assert_eq!(actual, &expected);
            }
        };
    }

    test_read_type!(test_bytes: "bytes", Schema::Bytes,
        [&b"test"[..], &b"another"[..]],
        Series::new("bytes".into(), [&b"test"[..], &b"another"[..]])
    );

    test_read_type!(test_enum: "enum",
        Schema::Enum(
            EnumSchema::builder()
                .name("enum".into())
                .symbols(vec!["a".into(), "b".into()])
                .build(),
        ),
        ["a", "b", "a"],
        Series::new("enum".into(), ["a", "b", "a"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap()
    );

    test_read_type!(test_fixed: "fixed",
        Schema::Fixed(FixedSchema::builder().name("fixed".into()).size(4).build()),
        [Value::Fixed(4, vec![1, 2, 3, 4]), Value::Fixed(4, vec![5, 6, 7, 8])],
        Series::new("fixed".into(), [&[1u8, 2, 3, 4][..], &[5, 6, 7, 8][..]])
    );

    test_read_type!(test_decimal: "decimal",
        Schema::Decimal(DecimalSchema { precision: 10, scale: 2, inner: Box::new(Schema::Bytes) }),
        [
            Value::Decimal(AvroDecimal::from(vec![0x64u8])),
            Value::Decimal(AvroDecimal::from(vec![0x00, 0xFA])),
        ],
        {
            let frame = df!("decimal" => ["1.00", "2.50"]).unwrap()
                .lazy()
                .select([pl::col("decimal").strict_cast(DataType::Decimal(10, 2))])
                .collect().unwrap();
            frame.column("decimal").unwrap().as_materialized_series().clone()
        }
    );

    // big decimal are deserialized as binary
    test_read_type!(test_big_decimal: "big_decimal", Schema::BigDecimal,
        [
            Value::BigDecimal(BigDecimal::from(12345u32)),
            Value::BigDecimal(BigDecimal::from(67890u32)),
        ],
        Series::new("big_decimal".into(), [
            &b"\x0409\x00"[..],
            &b"\x06\x01\x092\x00"[..],
        ])
    );

    test_read_type!(test_uuid_arr: "uuid", Schema::Uuid,
        [
            Value::Uuid(Uuid::parse_str("936da01f-9abd-4d9d-80c7-02af85c822a8").unwrap()),
            Value::Uuid(Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap()),
        ],
        Series::new("uuid".into(), [
            &b"\x93m\xa0\x1f\x9a\xbdM\x9d\x80\xc7\x02\xaf\x85\xc8\"\xa8"[..],
            &b"U\x0e\x84\x00\xe2\x9bA\xd4\xa7\x16DfUD\x00\x00"[..],
        ]),
        utf8_view: false
    );

    test_read_type!(test_uuid_view: "uuid", Schema::Uuid,
        [
            Value::Uuid(Uuid::parse_str("936da01f-9abd-4d9d-80c7-02af85c822a8").unwrap()),
            Value::Uuid(Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap()),
        ],
        Series::new("uuid".into(), [
            "936da01f-9abd-4d9d-80c7-02af85c822a8",
            "550e8400-e29b-41d4-a716-446655440000",
        ]),
        utf8_view: true
    );

    test_read_type!(test_null_string_arr: "uuid", Schema::Union(UnionSchema::new(vec![Schema::Null, Schema::String]).unwrap()),
        [
            Some("string"),
            None,
        ],
        Series::new("uuid".into(), [
            Some("string"),
            None,
       ]),
        utf8_view: false
    );

    test_read_type!(test_null_string_view: "uuid", Schema::Union(UnionSchema::new(vec![Schema::Null, Schema::String]).unwrap()),
        [
            Some("string"),
            None,
        ],
        Series::new("uuid".into(), [
            "string",
            "",
        ]),
        utf8_view: true
    );

    /// This needs to be separate so we can sort the keys of the result, since
    /// we can't guarantee order
    #[test]
    fn test_map() {
        let buff = write_avro(
            "map",
            Schema::Map(MapSchema {
                types: Box::new(Schema::Int),
                attributes: BTreeMap::new(),
            }),
            [
                HashMap::from([("a", 1), ("b", 2)]),
                HashMap::from([("c", 3)]),
            ],
        )
        .unwrap();
        let res = Reader::try_new([ok(buff)], FullReadOptions::default()).unwrap();
        let base = read_scan(res);
        // need to sort all of the lists to account for unknown serialization
        // order
        let frame = base
            .lazy()
            .with_column(pl::col("map").list().sort(SortOptions::default()))
            .collect()
            .unwrap();
        let expected = Series::new(
            "map".into(),
            [
                map_struct("1", ["a", "b"], [1, 2]),
                map_struct("2", ["c"], [3]),
            ],
        );
        let actual = frame.column("map").unwrap().as_series().unwrap();
        assert_eq!(actual, &expected);
    }

    test_read_type!(test_date: "date", Schema::Date,
        [Value::Date(19000), Value::Date(19500)],
        Series::new("date".into(), [19000i32, 19500])
            .cast(&DataType::Date).unwrap()
    );

    test_read_type!(test_time_millis: "time_ms", Schema::TimeMillis,
        [Value::TimeMillis(1000), Value::TimeMillis(2000)],
        Series::new("time_ms".into(), [1_000_000_000i64, 2_000_000_000])
            .cast(&DataType::Time).unwrap()
    );

    test_read_type!(test_time_micros: "time_us", Schema::TimeMicros,
        [Value::TimeMicros(1_000_000), Value::TimeMicros(2_000_000)],
        Series::new("time_us".into(), [1_000_000_000i64, 2_000_000_000])
            .cast(&DataType::Time).unwrap()
    );

    test_read_type!(test_timestamp_millis: "ts_ms", Schema::TimestampMillis,
        [Value::TimestampMillis(1_000_000), Value::TimestampMillis(2_000_000)],
        Series::new("ts_ms".into(), [1_000_000i64, 2_000_000])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, Some(TimeZone::UTC))).unwrap()
    );

    test_read_type!(test_timestamp_micros: "ts_us", Schema::TimestampMicros,
        [Value::TimestampMicros(1_000_000), Value::TimestampMicros(2_000_000)],
        Series::new("ts_us".into(), [1_000_000i64, 2_000_000])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC))).unwrap()
    );

    test_read_type!(test_timestamp_nanos: "ts_ns", Schema::TimestampNanos,
        [Value::TimestampNanos(1_000_000), Value::TimestampNanos(2_000_000)],
        Series::new("ts_ns".into(), [1_000_000i64, 2_000_000])
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, Some(TimeZone::UTC))).unwrap()
    );

    test_read_type!(test_local_timestamp_millis: "local_ts_ms", Schema::LocalTimestampMillis,
        [Value::LocalTimestampMillis(1_000_000), Value::LocalTimestampMillis(2_000_000)],
        Series::new("local_ts_ms".into(), [1_000_000i64, 2_000_000])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None)).unwrap()
    );

    test_read_type!(test_local_timestamp_micros: "local_ts_us", Schema::LocalTimestampMicros,
        [Value::LocalTimestampMicros(1_000_000), Value::LocalTimestampMicros(2_000_000)],
        Series::new("local_ts_us".into(), [1_000_000i64, 2_000_000])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None)).unwrap()
    );

    test_read_type!(test_local_timestamp_nanos: "local_ts_ns", Schema::LocalTimestampNanos,
        [Value::LocalTimestampNanos(1_000_000), Value::LocalTimestampNanos(2_000_000)],
        Series::new("local_ts_ns".into(), [1_000_000i64, 2_000_000])
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None)).unwrap()
    );

    /// arrow-avro will not deserialize durations
    #[test]
    fn test_duration_error() {
        let buff = write_avro(
            "duration",
            Schema::Duration,
            [Value::Duration(AvroDuration::new(
                Months::new(1),
                Days::new(2),
                Millis::new(3),
            ))],
        )
        .unwrap();
        let err = Reader::try_new([ok(buff)], FullReadOptions::default()).unwrap_err();
        assert!(matches!(err, Error::Arrow(_)));
    }

    /// Root Avro schema must be a Record
    #[test]
    fn test_single_column_error() {
        let mut buff = Cursor::new(Vec::new());
        let mut writer = Writer::new(&Schema::Int, &mut buff);
        writer.append(1).unwrap();
        writer.append(2).unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.set_position(0);
        let err = Reader::try_new([ok(buff)], FullReadOptions::default()).unwrap_err();
        assert!(matches!(err, Error::Arrow(_)));
    }

    /// Test failure on `with_columns` when column isn't present
    #[test]
    fn test_missing_columns_error() {
        let res = Reader::try_new(
            [File::open("./resources/food.avro")],
            ReadOptions {
                projection: Some(&["missing"][..]),
                ..ReadOptions::default()
            },
        );
        assert!(matches!(res, Err(Error::ColumnNotFound(_))));
    }

    #[test]
    fn test_different_schemas() {
        let one = write_avro("x", Schema::Int, [1, 2, 3]).unwrap();
        let two = write_avro("y", Schema::String, ["a", "b", "c"]).unwrap();

        let iter = Reader::try_new(
            [ok(one), ok(two)],
            ReadOptions {
                batch_size: 2,
                ..FullReadOptions::default()
            },
        )
        .unwrap();
        let err = iter.collect::<Result<Vec<_>, _>>().unwrap_err();
        assert!(matches!(err, Error::NonMatchingSchemas { .. }));
    }

    // arrow-avro reads null map values as empty lists instead of null
    test_read_type!(test_null_map: "map",
        Schema::Union(UnionSchema::new(vec![
            Schema::Null,
            Schema::Map(MapSchema {
                types: Box::new(Schema::Int),
                attributes: BTreeMap::new(),
            }),
        ]).unwrap()),
        [
            Value::Union(1, Box::new(Value::Map(HashMap::from([("a".into(), Value::Int(1))])))),
            Value::Union(0, Box::new(Value::Null)),
            Value::Union(1, Box::new(Value::Map(HashMap::from([("b".into(), Value::Int(2))])))),
        ],
        Series::new("map".into(), [
            map_struct("1", ["a"], [1]),
            map_struct("2", [], [] as [i32; 0]),
            map_struct("3", ["b"], [2]),
        ])
    );

    // arrow-avro reads null list values correctly
    test_read_type!(test_null_list: "list",
        Schema::Union(UnionSchema::new(vec![
            Schema::Null,
            Schema::Array(ArraySchema {
                items: Box::new(Schema::Int),
                attributes: BTreeMap::new(),
            }),
        ]).unwrap()),
        [
            Value::Union(1, Box::new(Value::Array(vec![Value::Int(1), Value::Int(2)]))),
            Value::Union(0, Box::new(Value::Null)),
            Value::Union(1, Box::new(Value::Array(vec![Value::Int(3)]))),
        ],
        Series::new("list".into(), &[
            Some(Series::from_iter([1_i32, 2_i32])),
            None,
            Some(Series::from_iter([3_i32])),
        ])
    );

    #[test]
    fn test_different_schemas_projection() {
        let one = write_avro("x", Schema::Int, [1, 2, 3]).unwrap();
        let two = write_avro("y", Schema::String, ["a", "b", "c"]).unwrap();

        let iter = Reader::try_new(
            [ok(one), ok(two)],
            ReadOptions {
                batch_size: 2,
                projection: Some(&["x"][..]),
                ..ReadOptions::default()
            },
        )
        .unwrap();
        let err = iter.collect::<Result<Vec<_>, _>>().unwrap_err();
        assert!(matches!(err, Error::ColumnNotFound(_)));
    }
}
