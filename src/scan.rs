//! Rust scan implementation

use super::Error;
use super::Projection;
use apache_avro::Reader as AvroReader;
use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use arrow_avro::reader::{Reader as ArrowAvroReader, ReaderBuilder};
use arrow_avro::schema::AvroSchema;
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
    /// String views avoid copying, so enabling this is likely faster if you
    /// don't mind losing null string distinctions.
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

/// An iterator that yields [`RecordBatch`]es from one or more avro sources.
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
    type Item = Result<RecordBatch, Error>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.source.next() {
                Some(Ok(batch)) if self.matched_schema(&batch) => {
                    return Some(Ok(batch));
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
    use apache_avro::schema::{FixedSchema, RecordField, RecordSchema, Schema, UnionSchema};
    use apache_avro::types::{Record, Value};
    use apache_avro::{
        AvroResult, Days, Decimal as AvroDecimal, Duration as AvroDuration, Millis, Months, Writer,
    };
    use arrow::array::{Array, RecordBatch};
    use arrow::compute::concat_batches;
    use arrow::datatypes::DataType;
    use std::convert::Infallible;
    use std::fs::File;
    use std::io::{Cursor, Read, Seek};
    use std::mem;
    use uuid::Uuid;

    #[allow(clippy::unnecessary_wraps)]
    fn ok<T>(val: T) -> Result<T, Infallible> {
        Ok(val)
    }

    /// Drain a reader and concatenate all of its batches into one.
    fn collect_one<R, E, I, P>(reader: Reader<R, I, P>) -> RecordBatch
    where
        R: Read + Seek,
        Error: From<E>,
        I: Iterator<Item = Result<R, E>>,
        P: Projection,
    {
        let batches: Vec<RecordBatch> = reader.map(|batch| batch.unwrap()).collect();
        let schema = batches
            .first()
            .expect("expected at least one batch")
            .schema();
        concat_batches(&schema, &batches).unwrap()
    }

    fn is_string_type(dtype: &DataType) -> bool {
        matches!(
            dtype,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        )
    }

    fn is_binary_type(dtype: &DataType) -> bool {
        matches!(
            dtype,
            DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_)
        )
    }

    /// Write a single-field avro record file to an in-memory buffer.
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
        let frame = collect_one(batches);
        assert_eq!(frame.num_rows(), 27);
        assert_eq!(frame.num_columns(), 4);
    }

    /// Projection reorders and subsets the columns
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
        let frame = collect_one(batches);
        let schema = frame.schema();
        let names: Vec<&str> = schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect();
        assert_eq!(names, columns);
    }

    /// Avro bytes are read as an arrow binary type
    #[test]
    fn test_bytes() {
        let buff = write_avro("bytes", Schema::Bytes, [&b"test"[..], &b"another"[..]]).unwrap();
        let frame = collect_one(Reader::try_new([ok(buff)], FullReadOptions::default()).unwrap());
        assert_eq!(frame.num_rows(), 2);
        assert!(is_binary_type(frame.column(0).data_type()));
    }

    /// Avro fixed is read as a fixed size binary
    #[test]
    fn test_fixed() {
        let buff = write_avro(
            "fixed",
            Schema::Fixed(FixedSchema::builder().name("fixed".into()).size(4).build()),
            [
                Value::Fixed(4, vec![1, 2, 3, 4]),
                Value::Fixed(4, vec![5, 6, 7, 8]),
            ],
        )
        .unwrap();
        let frame = collect_one(Reader::try_new([ok(buff)], FullReadOptions::default()).unwrap());
        assert_eq!(frame.column(0).data_type(), &DataType::FixedSizeBinary(4));
    }

    /// Avro decimal is read as a 128-bit decimal
    #[test]
    fn test_decimal() {
        let buff = write_avro(
            "decimal",
            Schema::Decimal(apache_avro::schema::DecimalSchema {
                precision: 10,
                scale: 2,
                inner: Box::new(Schema::Bytes),
            }),
            [
                Value::Decimal(AvroDecimal::from(vec![0x64u8])),
                Value::Decimal(AvroDecimal::from(vec![0x00, 0xFA])),
            ],
        )
        .unwrap();
        let frame = collect_one(Reader::try_new([ok(buff)], FullReadOptions::default()).unwrap());
        assert_eq!(frame.column(0).data_type(), &DataType::Decimal128(10, 2));
    }

    /// With `utf8_view` off, avro UUIDs decode to a binary type
    #[test]
    fn test_uuid_binary() {
        let buff = write_avro(
            "uuid",
            Schema::Uuid,
            [Value::Uuid(
                Uuid::parse_str("936da01f-9abd-4d9d-80c7-02af85c822a8").unwrap(),
            )],
        )
        .unwrap();
        let frame = collect_one(
            Reader::try_new(
                [ok(buff)],
                ReadOptions {
                    utf8_view: false,
                    ..FullReadOptions::default()
                },
            )
            .unwrap(),
        );
        assert!(is_binary_type(frame.column(0).data_type()));
    }

    /// With `utf8_view` on, avro UUIDs decode to a string type
    #[test]
    fn test_uuid_view() {
        let buff = write_avro(
            "uuid",
            Schema::Uuid,
            [Value::Uuid(
                Uuid::parse_str("936da01f-9abd-4d9d-80c7-02af85c822a8").unwrap(),
            )],
        )
        .unwrap();
        let frame = collect_one(
            Reader::try_new(
                [ok(buff)],
                ReadOptions {
                    utf8_view: true,
                    ..FullReadOptions::default()
                },
            )
            .unwrap(),
        );
        assert!(is_string_type(frame.column(0).data_type()));
    }

    /// With `utf8_view` off, nulls in a nullable string are preserved
    #[test]
    fn test_null_string_preserved() {
        let buff = write_avro(
            "val",
            Schema::Union(UnionSchema::new(vec![Schema::Null, Schema::String]).unwrap()),
            [Some("string"), None],
        )
        .unwrap();
        let frame = collect_one(
            Reader::try_new(
                [ok(buff)],
                ReadOptions {
                    utf8_view: false,
                    ..FullReadOptions::default()
                },
            )
            .unwrap(),
        );
        assert!(is_string_type(frame.column(0).data_type()));
        assert_eq!(frame.column(0).null_count(), 1);
    }

    /// With `utf8_view` on, nulls in a nullable string become empty strings
    #[test]
    fn test_null_string_lossy() {
        let buff = write_avro(
            "val",
            Schema::Union(UnionSchema::new(vec![Schema::Null, Schema::String]).unwrap()),
            [Some("string"), None],
        )
        .unwrap();
        let frame = collect_one(
            Reader::try_new(
                [ok(buff)],
                ReadOptions {
                    utf8_view: true,
                    ..FullReadOptions::default()
                },
            )
            .unwrap(),
        );
        assert!(is_string_type(frame.column(0).data_type()));
        assert_eq!(frame.column(0).null_count(), 0);
    }

    /// A non-first source that fails surfaces as an error item during iteration.
    #[test]
    fn test_source_error_propagates() {
        let valid = write_avro("col", Schema::Int, [1, 2, 3]).unwrap();
        let sources: Vec<Result<Cursor<Vec<u8>>, std::io::Error>> =
            vec![Ok(valid), Err(std::io::Error::other("boom"))];
        let mut reader = Reader::try_new(sources, FullReadOptions::default()).unwrap();
        let last = reader.by_ref().last().unwrap();
        assert!(matches!(last, Err(Error::IO(_, _))));
    }

    /// Serves a valid avro stream until `fail_at`, then returns a single I/O
    /// error followed by EOF, so the reader errors once and then stops.
    struct FailOnceReader {
        data: Cursor<Vec<u8>>,
        fail_at: u64,
        failed: bool,
    }

    impl Read for FailOnceReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let pos = self.data.position();
            if pos >= self.fail_at {
                if self.failed {
                    return Ok(0);
                }
                self.failed = true;
                return Err(std::io::Error::other("boom"));
            }
            let remaining = usize::try_from(self.fail_at - pos).unwrap_or(usize::MAX);
            let limit = remaining.min(buf.len());
            self.data.read(&mut buf[..limit])
        }
    }

    impl Seek for FailOnceReader {
        fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
            self.data.seek(pos)
        }
    }

    /// An I/O error raised by the underlying reader mid-stream is surfaced as an
    /// error item. The file is larger than the `BufReader` buffer so the header
    /// is read up front (letting construction succeed) while the failure lands
    /// on a later block read.
    #[test]
    fn test_reader_error_propagates() {
        let bytes = write_avro("col", Schema::Int, 0..20_000)
            .unwrap()
            .into_inner();
        assert!(bytes.len() > 8192, "need a multi-buffer file");
        let source = FailOnceReader {
            fail_at: u64::try_from(bytes.len()).unwrap() / 2,
            data: Cursor::new(bytes),
            failed: false,
        };
        let mut reader = Reader::try_new(
            [ok(source)],
            ReadOptions {
                batch_size: 2,
                ..FullReadOptions::default()
            },
        )
        .unwrap();
        // read up to (and including) the first error, then stop
        let err = reader
            .find(Result::is_err)
            .expect("expected an error item")
            .unwrap_err();
        assert!(matches!(err, Error::Arrow(_)), "{err:?}");
    }

    #[test]
    fn test_empty_sources_error() {
        let sources: Vec<Result<Cursor<Vec<u8>>, std::io::Error>> = Vec::new();
        let err = Reader::try_new(sources, FullReadOptions::default()).unwrap_err();
        assert!(matches!(err, Error::EmptySources));
    }

    #[test]
    fn test_first_source_error() {
        let sources: Vec<Result<Cursor<Vec<u8>>, std::io::Error>> =
            vec![Err(std::io::Error::other("boom"))];
        let err = Reader::try_new(sources, FullReadOptions::default()).unwrap_err();
        assert!(matches!(err, Error::IO(_, _)));
    }

    #[test]
    fn test_projection_bad_header_error() {
        let err = Reader::try_new(
            [ok(Cursor::new(b"not an avro file".to_vec()))],
            ReadOptions {
                projection: Some(&["x"][..]),
                ..ReadOptions::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::Avro(_) | Error::Arrow(_) | Error::ArrowAvro(_)),
            "{err:?}"
        );
    }

    /// Serves valid data but fails after `ok_seeks` successful seeks, to exercise
    /// the seek error paths in the projection branch of `create_reader`.
    #[derive(Debug)]
    struct FailSeekAfter {
        data: Cursor<Vec<u8>>,
        ok_seeks: usize,
    }

    impl Read for FailSeekAfter {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.data.read(buf)
        }
    }

    impl Seek for FailSeekAfter {
        fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
            if self.ok_seeks == 0 {
                return Err(std::io::Error::other("no seek"));
            }
            self.ok_seeks -= 1;
            self.data.seek(pos)
        }
    }

    fn projection_with_seeks(ok_seeks: usize) -> Error {
        let valid = write_avro("col", Schema::Int, [1, 2, 3])
            .unwrap()
            .into_inner();
        Reader::try_new(
            [ok(FailSeekAfter {
                data: Cursor::new(valid),
                ok_seeks,
            })],
            ReadOptions {
                projection: Some(&["col"][..]),
                ..ReadOptions::default()
            },
        )
        .unwrap_err()
    }

    #[test]
    fn test_projection_initial_seek_error() {
        // fails on the first `stream_position`, before reading the schema
        assert!(matches!(projection_with_seeks(0), Error::IO(_, _)));
    }

    #[test]
    fn test_projection_rewind_seek_error() {
        // fails on the second `stream_position`, after reading the schema
        assert!(matches!(projection_with_seeks(1), Error::IO(_, _)));
    }

    /// Reports position 0 on the first `stream_position` and `u64::MAX` on the
    /// second, so the header offset can't fit in an `i64`.
    #[derive(Debug)]
    struct HugePosReader {
        data: Cursor<Vec<u8>>,
        seeks: usize,
    }

    impl Read for HugePosReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.data.read(buf)
        }
    }

    impl Seek for HugePosReader {
        fn seek(&mut self, _: std::io::SeekFrom) -> std::io::Result<u64> {
            self.seeks += 1;
            if self.seeks == 1 { Ok(0) } else { Ok(u64::MAX) }
        }
    }

    #[test]
    fn test_large_header_error() {
        let valid = write_avro("col", Schema::Int, [1, 2, 3])
            .unwrap()
            .into_inner();
        let err = Reader::try_new(
            [ok(HugePosReader {
                data: Cursor::new(valid),
                seeks: 0,
            })],
            ReadOptions {
                projection: Some(&["col"][..]),
                ..ReadOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::LargeHeader), "{err:?}");
    }

    /// Allows `stream_position` (a `Current(0)` seek) but fails any real seek, so
    /// the post-schema rewind fails once the header is too big to stay buffered.
    #[derive(Debug)]
    struct NoRelativeSeek(Cursor<Vec<u8>>);

    impl Read for NoRelativeSeek {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.0.read(buf)
        }
    }

    impl Seek for NoRelativeSeek {
        fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
            match pos {
                std::io::SeekFrom::Current(0) => self.0.seek(pos),
                _ => Err(std::io::Error::other("no relative seek")),
            }
        }
    }

    #[test]
    fn test_projection_large_header_rewind_error() {
        // a long field name makes the header exceed the 8 KiB BufReader buffer
        let name = "f".repeat(9000);
        let valid = write_avro(&name, Schema::Int, [1, 2, 3])
            .unwrap()
            .into_inner();
        assert!(valid.len() > 8192, "header should exceed the buffer");
        let err = Reader::try_new(
            [ok(NoRelativeSeek(Cursor::new(valid)))],
            ReadOptions {
                projection: Some(&[name.as_str()][..]),
                ..ReadOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::IO(_, _)), "{err:?}");
    }

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

    /// A projected column that isn't present errors out
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
