//! Rust scan implementation

use std::io::Read;
use std::iter::{Fuse, FusedIterator};
use std::sync::Arc;

use crate::des::new_value_builder;

use super::{Error, des};
use apache_avro::Reader;
use apache_avro::types::Value;
use polars::error::PolarsError;
use polars::frame::DataFrame;
use polars::prelude::{Column, PlSmallStr, Schema as PlSchema};
use polars::series::Series;

/// An abstract scanner that can be converted into an iterator over `DataFrame`s
pub struct AvroScanner<R, I> {
    reader: Reader<'static, R>,
    sources: I,
    schema: Arc<PlSchema>,
    single_column_name: Option<PlSmallStr>,
}

// FIXME move into module since we need to expose it
pub enum Infallable {}

impl From<Infallable> for Error {
    fn from(_: Infallable) -> Self {
        unreachable!()
    }
}

pub struct InfallableIter<I>(I);

impl<I: Iterator> Iterator for InfallableIter<I> {
    type Item = Result<I::Item, Infallable>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(Result::Ok)
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for InfallableIter<I> {}
impl<I: FusedIterator> FusedIterator for InfallableIter<I> {}

impl<R, I> AvroScanner<R, InfallableIter<I>>
where
    R: Read,
    I: Iterator<Item = R>,
{
    /// Create a new scanner from `ScanSources`
    ///
    /// # Errors
    ///
    /// If the schema can't be converted into a polars schema.
    pub fn new(
        sources: impl IntoIterator<IntoIter = I>,
        single_column_name: Option<PlSmallStr>,
    ) -> Result<Self, Error> {
        Self::try_new(InfallableIter(sources.into_iter()), single_column_name)
    }
}

impl<R, E, I> AvroScanner<R, I>
where
    R: Read,
    Error: From<E>,
    I: Iterator<Item = Result<R, E>>,
{
    /// Create a new scanner from `ScanSources`
    ///
    /// # Errors
    ///
    /// If the schema can't be converted into a polars schema, or any errors from the readers.
    pub fn try_new(
        sources: impl IntoIterator<IntoIter = I>,
        single_column_name: Option<PlSmallStr>,
    ) -> Result<Self, Error> {
        let mut sources = sources.into_iter();
        let source = sources.next().ok_or(Error::EmptySources)??;
        let reader = Reader::new(source)?;
        let schema = Arc::new(des::try_from_schema(
            reader.writer_schema(),
            single_column_name.as_ref(),
        )?);

        Ok(Self {
            reader,
            sources,
            schema,
            single_column_name,
        })
    }

    /// Get the schema
    pub fn schema(&self) -> Arc<PlSchema> {
        self.schema.clone()
    }

    /// Convert the scanner into an actual iterator
    pub fn into_iter(
        self,
        batch_size: usize,
        with_columns: Option<Arc<[usize]>>,
    ) -> Fuse<AvroIter<R, I>> {
        AvroIter {
            reader: self.reader,
            sources: self.sources,
            schema: self.schema,
            single_column_name: self.single_column_name,
            batch_size,
            with_columns,
        }
        .fuse()
    }

    /// Convert the scanner into an actual iterator
    ///
    /// This uses string columns instead of indices
    ///
    /// # Errors
    ///
    /// If columns don't exist in the schema.
    pub fn try_into_iter(
        self,
        batch_size: usize,
        columns: Option<&[impl AsRef<str>]>,
    ) -> Result<Fuse<AvroIter<R, I>>, Error> {
        let with_columns = if let Some(columns) = columns {
            let indexes = columns
                .iter()
                .map(|name| {
                    self.schema
                        .index_of(name.as_ref())
                        .ok_or_else(|| PolarsError::ColumnNotFound(name.as_ref().to_owned().into()))
                })
                .collect::<Result<_, _>>()?;
            Some(indexes)
        } else {
            None
        };
        Ok(AvroIter {
            reader: self.reader,
            sources: self.sources,
            schema: self.schema,
            single_column_name: self.single_column_name,
            batch_size,
            with_columns,
        }
        .fuse())
    }
}

/// An `Iterator` of `DataFrame` batches scanned from various sources
pub struct AvroIter<R, I> {
    reader: Reader<'static, R>,
    sources: I,
    schema: Arc<PlSchema>,
    single_column_name: Option<PlSmallStr>,
    batch_size: usize,
    with_columns: Option<Arc<[usize]>>,
}

impl<R, E, I> AvroIter<R, I>
where
    R: Read,
    Error: From<E>,
    I: Iterator<Item = Result<R, E>>,
{
    fn read_columns(
        &mut self,
        with_columns: impl IntoIterator<Item = usize> + Clone,
    ) -> Result<Vec<Column>, Error> {
        // abstracts this where we also pass in inds, which is a cloneable usize iterator and can eeither be with_columns or 0..width()
        let mut arrow_columns: Box<[_]> = with_columns
            .clone()
            .into_iter()
            .map(|idx| {
                // already checked that idx valid for schema
                let (_, dtype) = self.schema.get_at_index(idx).unwrap();
                new_value_builder(dtype, self.batch_size)
            })
            .collect();

        for _ in 0..self.batch_size {
            if let Some(rec) = self.reader.next() {
                let val = rec?;
                if let Value::Record(rec_val) = val {
                    for (idx, col) in with_columns.clone().into_iter().zip(&mut arrow_columns) {
                        let (_, val) = &rec_val[idx];
                        col.try_push_value(val).map_err(Error::Polars)?;
                    }
                } else {
                    // mapped to a single column
                    let col = arrow_columns.first_mut().unwrap();
                    col.try_push_value(&val).map_err(Error::Polars)?;
                }
            } else if let Some(source) = self.sources.next() {
                self.reader = Reader::new(source?)?;
                // NOTE we could be lazy and just check compatability, but
                // we do want something like this equality, which will allow
                // scanning multiple avro files as long as they have the
                // same converted arrow schema, e.g. nullability or
                // different integers.
                let new_schema = des::try_from_schema(
                    self.reader.writer_schema(),
                    self.single_column_name.as_ref(),
                )?;
                if new_schema != *self.schema {
                    return Err(Error::NonMatchingSchemas);
                }
            } else {
                break;
            }
        }

        with_columns
            .into_iter()
            .zip(&mut arrow_columns)
            .map(|(idx, col)| {
                let (name, dtype) = self.schema.get_at_index(idx).unwrap();
                let ser = Series::from_arrow(name.clone(), col.as_box())?;
                // NOTE we intentionally want to avoid any actual casting here
                Ok(unsafe { ser.cast_unchecked(dtype) }?.into())
            })
            .collect()
    }

    fn read_frame(&mut self) -> Result<DataFrame, Error> {
        let columns = if let Some(with_columns) = &self.with_columns {
            let cols = with_columns.clone();
            self.read_columns(cols.iter().copied())?
        } else {
            self.read_columns(0..self.schema.len())?
        };

        Ok(DataFrame::new(columns)?)
    }
}

impl<R, E, I> Iterator for AvroIter<R, I>
where
    R: Read,
    Error: From<E>,
    I: Iterator<Item = Result<R, E>>,
{
    type Item = Result<DataFrame, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_frame() {
            Ok(frame) if frame.is_empty() => None,
            res => Some(res),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{Cursor, Read, Seek, SeekFrom};
    use std::mem;
    use std::sync::Arc;

    use apache_avro::types::Value;
    use apache_avro::{Schema, Writer};
    use chrono::NaiveTime;
    use polars::df;
    use polars::error::PolarsError;
    use polars::frame::DataFrame;
    use polars::prelude::{self as pl, DataType, IntoLazy, Schema as PlSchema, UnionArgs};

    use super::AvroScanner;
    use crate::Error;

    fn read_scan<R: Read, E>(
        scanner: AvroScanner<R, impl Iterator<Item = Result<R, E>>>,
    ) -> DataFrame
    where
        Error: From<E>,
    {
        let frames: Vec<_> = scanner
            .into_iter(1024, None)
            .map(|part| part.unwrap().lazy())
            .collect();
        pl::concat(frames, UnionArgs::default())
            .unwrap()
            .collect()
            .unwrap()
    }

    impl From<std::io::Error> for Error {
        fn from(err: std::io::Error) -> Self {
            Error::Polars(PolarsError::IO {
                error: Arc::new(err),
                msg: None,
            })
        }
    }

    /// Test scan on a simple fil
    #[test]
    fn test_scan() {
        let scanner = AvroScanner::try_new([File::open("./resources/food.avro")], None).unwrap();
        let frame = read_scan(scanner);
        assert_eq!(frame.height(), 27);
        assert_eq!(frame.schema().len(), 4);
    }

    /// Test reading uuid
    #[test]
    fn test_uuid() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::parse_str(
            r#"
        {
            "type": "record",
            "name": "base",
            "fields": [{"name": "field", "type": {"type": "string", "name": "uuid", "logicalType": "uuid"}}]
        }
        "#,
        )
        .unwrap();
        let mut writer = Writer::new(&schema, &mut buff);
        writer
            .append(Value::Record(vec![(
                "field".into(),
                Value::String("3738b99e-f4ae-40de-bb3f-fd4aa9d9e9f7".into()),
            )]))
            .unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.seek(SeekFrom::Start(0)).unwrap();
        let scanner = AvroScanner::new([buff], None).unwrap();
        read_scan(scanner);
    }

    /// Test reading fixed
    #[test]
    fn test_fixed() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::parse_str(
            r#"
        {
            "type": "record",
            "name": "base",
            "fields": [{"name": "field", "type": {"type": "fixed", "name": "fixed", "size": 4}}]
        }
        "#,
        )
        .unwrap();
        let mut writer = Writer::new(&schema, &mut buff);
        writer
            .append(Value::Record(vec![(
                "field".into(),
                Value::Fixed(4, b"0123".into()),
            )]))
            .unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.seek(SeekFrom::Start(0)).unwrap();
        let scanner = AvroScanner::new([buff], None).unwrap();
        let result = read_scan(scanner);
        let expected = df! {
            "field" => [&b"0123"[..]]
        }
        .unwrap();
        assert_eq!(result, expected);
    }

    /// Test time
    #[test]
    fn test_time() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::parse_str(
            r#"
        {
            "type": "record",
            "name": "base",
            "fields": [
                {"name": "millis", "type": {"type": "int", "logicalType": "time-millis"}},
                {"name": "micros", "type": {"type": "long", "logicalType": "time-micros"}}
            ]
        }
        "#,
        )
        .unwrap();
        let mut writer = Writer::new(&schema, &mut buff);
        writer
            .append(Value::Record(vec![
                ("millis".into(), Value::TimeMillis(1)),
                ("micros".into(), Value::TimeMicros(1)),
            ]))
            .unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.seek(SeekFrom::Start(0)).unwrap();
        let scanner = AvroScanner::new([buff], None).unwrap();
        let frame = read_scan(scanner);
        let expected = df! {
            "millis" => [
                NaiveTime::from_hms_milli_opt(0, 0, 0, 1).unwrap(),
            ],
            "micros" => [
                NaiveTime::from_hms_micro_opt(0, 0, 0, 1).unwrap(),
            ],
        }
        .unwrap();
        assert_eq!(frame, expected);
    }

    /// Test failure on avros that aren't a top level record
    #[test]
    fn test_non_record_avro() {
        let mut buff = Cursor::new(Vec::new());
        let mut writer = Writer::new(&Schema::Boolean, &mut buff);
        writer.append(true).unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.seek(SeekFrom::Start(0)).unwrap();
        let res = AvroScanner::new([buff], None);
        assert!(matches!(res, Err(Error::NonRecordSchema(_))));
    }

    /// Test failure on avro union type
    #[test]
    fn test_map() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::parse_str(
            r#"
        {
            "type": "record",
            "name": "base",
            "fields": [{"name": "field", "type": {"type": "map", "values": "int"}}]
        }
        "#,
        )
        .unwrap();
        let mut writer = Writer::new(&schema, &mut buff);
        writer
            .append(Value::Record(vec![(
                "field".into(),
                Value::Map([("key".into(), Value::Int(1))].into()),
            )]))
            .unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.seek(SeekFrom::Start(0)).unwrap();
        let scanner = AvroScanner::new([buff], None).unwrap();
        let frame = read_scan(scanner);
        let base = df! {
            "key" => ["key"],
            "value" => [1],
        }
        .unwrap();
        let expected = base
            .lazy()
            .select([
                pl::concat_list([pl::as_struct(vec![pl::col("key"), pl::col("value")])]).unwrap(),
            ])
            .collect()
            .unwrap();
        assert_eq!(frame, expected);
    }

    /// Test failure on avro union type
    #[test]
    fn test_union_avro() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::parse_str(
            r#"
        {
            "type": "record",
            "name": "base",
            "fields": [{"name": "field", "type": ["boolean", "int"]}]
        }
        "#,
        )
        .unwrap();
        let mut writer = Writer::new(&schema, &mut buff);
        writer
            .append(Value::Record(vec![(
                "field".into(),
                Value::Union(0, Box::new(Value::Boolean(true))),
            )]))
            .unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.seek(SeekFrom::Start(0)).unwrap();
        let res = AvroScanner::new([buff], None);
        assert!(matches!(res, Err(Error::UnsupportedAvroType(_))));
    }

    /// Test failure on union with only null member
    #[test]
    fn test_null_union_avro() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::parse_str(
            r#"
        {
            "type": "record",
            "name": "base",
            "fields": [{"name": "field", "type": ["null"]}]
        }
        "#,
        )
        .unwrap();
        let mut writer = Writer::new(&schema, &mut buff);
        writer
            .append(Value::Record(vec![(
                "field".into(),
                Value::Union(0, Box::new(Value::Null)),
            )]))
            .unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.seek(SeekFrom::Start(0)).unwrap();
        let scanner = AvroScanner::new([buff], None).unwrap();
        let frame = read_scan(scanner);
        assert_eq!(
            **frame.schema(),
            PlSchema::from_iter([("field".into(), DataType::Null)])
        );
    }

    /// Test failure on with_columns when column isn't present
    #[test]
    fn test_missing_columns() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::parse_str(
            r#"
        {
            "type": "record",
            "name": "base",
            "fields": [{"name": "field", "type": "int"}]
        }
        "#,
        )
        .unwrap();
        let mut writer = Writer::new(&schema, &mut buff);
        writer
            .append(Value::Record(vec![("field".into(), Value::Int(0))]))
            .unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.seek(SeekFrom::Start(0)).unwrap();
        let scanner = AvroScanner::new([buff], None).unwrap();
        let iter = scanner.try_into_iter(1024, Some(&["missing"]));
        assert!(matches!(
            iter,
            Err(Error::Polars(PolarsError::ColumnNotFound(_)))
        ));
    }

    /// Test non-record avro fails
    #[test]
    fn test_non_record() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::parse_str(r#""int""#).unwrap();
        {
            let mut writer = Writer::new(&schema, &mut buff);
            writer.append(Value::Int(0)).unwrap();
            writer.append(Value::Int(1)).unwrap();
            writer.append(Value::Int(4)).unwrap();
            writer.flush().unwrap();
        }
        buff.seek(SeekFrom::Start(0)).unwrap();
        let res = AvroScanner::new([buff], None);
        assert!(matches!(res, Err(Error::NonRecordSchema(_))));
    }

    /// Test non-record avro fails
    #[test]
    fn test_single_column_name() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::parse_str(r#""int""#).unwrap();
        let mut writer = Writer::new(&schema, &mut buff);
        writer.append(Value::Int(0)).unwrap();
        writer.append(Value::Int(1)).unwrap();
        writer.append(Value::Int(4)).unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.seek(SeekFrom::Start(0)).unwrap();
        let scanner = AvroScanner::new([buff], Some("col".into())).unwrap();
        let frame = read_scan(scanner);
        let expected = df! {
            "col" => [0, 1, 4]
        }
        .unwrap();
        assert_eq!(frame, expected);
    }

    /// Test that scanner can appropriately read schemas with references
    #[test]
    fn test_references() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::parse_str(
            r#"
        {
            "type": "record",
            "name": "base",
            "fields": [
                {"name": "first", "type": "enum", "symbols": ["a"]},
                {"name": "second", "type": { "type": "enum", "name": "explicit.second", "symbols": ["b"]}},
                {"name": "a", "type": { "name": "a", "type": "record", "namespace": "outer", "fields": [
                    {"name": "first", "type": "enum", "symbols": ["c"]},
                    {"name": "fourth", "namespace": "explicit", "type": "enum", "symbols": ["d"]},
                    {"name": "b", "type": {"name": "b", "type": "record", "fields": [
                        {"name": "fifth", "type": "enum", "symbols": ["e"]},
                        {"name": "sixth", "namespace": "explicit", "type": "enum", "symbols": ["f"]},
                        {"name": "c", "type": { "name": "inner.c", "type": "record", "fields": [
                            {"name": "first", "type": "enum", "symbols": ["g"]},
                            {"name": "eighth", "namespace": "explicit", "type": "enum", "symbols": ["h"]},
                            {"name": "ninth", "type": "first"},
                            {"name": "tenth", "type": "outer.first"},
                            {"name": "eleventh", "type": "explicit.second"}
                        ]}},
                        {"name": "twlevth", "type": "first"},
                        {"name": "thirteenth", "type": "inner.first"},
                        {"name": "fourteenth", "type": "explicit.eighth"}
                    ]}},
                    {"name": "fifteenth", "type": "first"},
                    {"name": "sixteenth", "type": "inner.first"},
                    {"name": "seventeenth", "type": "explicit.sixth"}
                ]}},
                {"name": "eighteenth", "type": "first"},
                {"name": "nineteenth", "type": "inner.first"},
                {"name": "twentyth", "type": "outer.first"}
            ]
        }
        "#,
        )
        .unwrap();
        let mut writer = Writer::new(&schema, &mut buff);
        writer
            .append(Value::Record(vec![
                ("first".into(), Value::Enum(0, "a".into())),
                ("second".into(), Value::Enum(0, "b".into())),
                (
                    "a".into(),
                    Value::Record(vec![
                        ("first".into(), Value::Enum(0, "c".into())),
                        ("fourth".into(), Value::Enum(0, "d".into())),
                        (
                            "b".into(),
                            Value::Record(vec![
                                ("fifth".into(), Value::Enum(0, "e".into())),
                                ("sixth".into(), Value::Enum(0, "f".into())),
                                (
                                    "c".into(),
                                    Value::Record(vec![
                                        ("first".into(), Value::Enum(0, "g".into())),
                                        ("eighth".into(), Value::Enum(0, "h".into())),
                                        ("ninth".into(), Value::Enum(0, "g".into())),
                                        ("tenth".into(), Value::Enum(0, "c".into())),
                                        ("eleventh".into(), Value::Enum(0, "b".into())),
                                    ]),
                                ),
                                ("twlevth".into(), Value::Enum(0, "c".into())),
                                ("thirteenth".into(), Value::Enum(0, "g".into())),
                                ("fourteenth".into(), Value::Enum(0, "h".into())),
                            ]),
                        ),
                        ("fifteenth".into(), Value::Enum(0, "c".into())),
                        ("sixteenth".into(), Value::Enum(0, "g".into())),
                        ("seventeenth".into(), Value::Enum(0, "f".into())),
                    ]),
                ),
                ("eighteenth".into(), Value::Enum(0, "a".into())),
                ("nineteenth".into(), Value::Enum(0, "g".into())),
                ("twentyth".into(), Value::Enum(0, "c".into())),
            ]))
            .unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.seek(SeekFrom::Start(0)).unwrap();
        let scanner = AvroScanner::new([buff], None).unwrap();
        let frame = read_scan(scanner);
        assert_eq!(frame.height(), 1);
        assert_eq!(frame.width(), 6);
    }
}
