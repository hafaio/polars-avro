//! Rust scan implementation

use std::io::Cursor;
use std::iter::Fuse;
use std::sync::Arc;

use crate::des::new_value_builder;

use super::io::SourceIter;
use super::{Error, des};
use apache_avro::Reader;
use apache_avro::types::Value;
use polars::error::PolarsError;
use polars::frame::DataFrame;
use polars::prelude::{Column, CompatLevel, Schema as PlSchema};
use polars::series::Series;
use polars_io::cloud::CloudOptions;
use polars_plan::prelude::ScanSources;
use polars_utils::mmap::MemSlice;

/// An abstract scanner that can be converted into an iterator over `DataFrame`s
pub struct AvroScanner {
    reader: Reader<'static, Cursor<MemSlice>>,
    source_iter: SourceIter,
    schema: Arc<PlSchema>,
}

impl AvroScanner {
    /// Create a new scanner from `ScanSources`
    ///
    /// # Errors
    ///
    /// If the schema can't be converted into a polars schema, or any other io errors.
    pub fn new_from_sources(
        sources: &ScanSources,
        glob: bool,
        cloud_options: Option<&CloudOptions>,
    ) -> Result<Self, Error> {
        let mut source_iter = SourceIter::try_from(sources, cloud_options, glob)?;
        let source = source_iter.next().ok_or(Error::EmptySources)??;
        let reader = Reader::new(source)?;
        let schema = Arc::new(des::try_from_schema(reader.writer_schema())?);

        Ok(Self {
            reader,
            source_iter,
            schema,
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
    ) -> Fuse<AvroIter> {
        AvroIter {
            reader: self.reader,
            source_iter: self.source_iter,
            schema: self.schema,
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
    ) -> Result<Fuse<AvroIter>, Error> {
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
            source_iter: self.source_iter,
            schema: self.schema,
            batch_size,
            with_columns,
        }
        .fuse())
    }
}

/// An `Iterator` of `DataFrame` batches scanned from various sources
pub struct AvroIter {
    reader: Reader<'static, Cursor<MemSlice>>,
    source_iter: SourceIter,
    schema: Arc<PlSchema>,
    batch_size: usize,
    with_columns: Option<Arc<[usize]>>,
}

impl AvroIter {
    fn read_columns(
        &mut self,
        with_columns: impl IntoIterator<Item = usize> + Clone,
    ) -> Result<Vec<Column>, Error> {
        let compat = CompatLevel::newest();
        // abstracts this where we also pass in inds, which is a cloneable usize iterator and can eeither be with_columns or 0..width()
        let mut arrow_columns: Box<[_]> = with_columns
            .clone()
            .into_iter()
            .map(|idx| {
                // already checked that idx valid for schema
                let (_, dtype) = self.schema.get_at_index(idx).unwrap();
                new_value_builder(&dtype.to_arrow(compat), self.batch_size)
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
                    unreachable!("top level schema validated as a record schema");
                }
            } else if let Some(source) = self.source_iter.next() {
                self.reader = Reader::new(source?)?;
                // NOTE we could be lazy and just check compatability, but
                // we do want something like this equality, which will allow
                // scanning multiple avro files as long as they have the
                // same converted arrow schema, e.g. nullability or
                // different integers.
                let new_schema = des::try_from_schema(self.reader.writer_schema())?;
                if new_schema != *self.schema {
                    return Err(Error::NonMatchingSchemas);
                }
            } else {
                break;
            }
        }

        Ok(with_columns
            .into_iter()
            .zip(&mut arrow_columns)
            .map(|(idx, col)| {
                let (name, dtype) = self.schema.get_at_index(idx).unwrap();
                // NOTE safety is checked inside from, only during debug, the
                // types won't align due to how enums are built from chunks, we
                // need to do it this way, so logical types are interpreted
                unsafe {
                    Series::from_chunks_and_dtype_unchecked(name.clone(), vec![col.as_box()], dtype)
                }
                .into()
            })
            .collect())
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

impl Iterator for AvroIter {
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
    use std::io::Cursor;
    use std::path::PathBuf;

    use apache_avro::types::Value;
    use apache_avro::{Schema, Writer};
    use chrono::NaiveTime;
    use polars::df;
    use polars::error::PolarsError;
    use polars::frame::DataFrame;
    use polars::prelude::{self as pl, DataType, IntoLazy, Schema as PlSchema, UnionArgs};
    use polars_plan::plans::ScanSources;
    use polars_utils::mmap::MemSlice;

    use super::AvroScanner;
    use crate::Error;

    fn from_paths(paths: impl IntoIterator<Item = impl Into<PathBuf>>) -> ScanSources {
        ScanSources::Paths(
            paths
                .into_iter()
                .map(std::convert::Into::into)
                .collect::<Box<[_]>>()
                .into(),
        )
    }

    fn read_scan(scanner: AvroScanner) -> DataFrame {
        let frames: Vec<_> = scanner
            .into_iter(1024, None)
            .map(|part| part.unwrap().lazy())
            .collect();
        pl::concat(frames, UnionArgs::default())
            .unwrap()
            .collect()
            .unwrap()
    }

    /// Test scan on a simple file
    #[test]
    fn test_scan() {
        let scanner =
            AvroScanner::new_from_sources(&from_paths(["./resources/food.avro"]), false, None)
                .unwrap();
        let frame = read_scan(scanner);
        assert_eq!(frame.height(), 27);
        assert_eq!(frame.schema().len(), 4);
    }

    /// Test support for globbing
    #[test]
    fn test_glob() {
        let scanner =
            AvroScanner::new_from_sources(&from_paths(["./resources/*.avro"]), true, None).unwrap();
        let frame = read_scan(scanner);
        assert_eq!(frame.height(), 30);
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
        let bytes = buff.into_inner();
        let scanner = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into()),
            false,
            None,
        )
        .unwrap();
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
        let bytes = buff.into_inner();
        let scanner = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into()),
            false,
            None,
        )
        .unwrap();
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
        let bytes = buff.into_inner();
        let scanner = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into()),
            false,
            None,
        )
        .unwrap();
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
        let bytes = buff.into_inner();
        let res = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into()),
            false,
            None,
        );
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
        let bytes = buff.into_inner();

        let scanner = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into()),
            false,
            None,
        )
        .unwrap();
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
        let bytes = buff.into_inner();

        let res = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into()),
            false,
            None,
        );
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
        let bytes = buff.into_inner();
        let scanner = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into()),
            false,
            None,
        )
        .unwrap();
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
        let bytes = buff.into_inner();
        let scanner = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into()),
            false,
            None,
        )
        .unwrap();
        let iter = scanner.try_into_iter(1024, Some(&["missing"]));
        assert!(matches!(
            iter,
            Err(Error::Polars(PolarsError::ColumnNotFound(_)))
        ));
    }
}
