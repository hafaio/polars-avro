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
                // we create col from dtype
                let (name, dtype) = self.schema.get_at_index(idx).unwrap();
                // NOTE safety is checked inside from, only during debug, the
                // types won't align due to how enums are built from chunks
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
    use std::collections::BTreeMap;
    use std::io::Cursor;
    use std::path::PathBuf;

    use apache_avro::schema::{
        FixedSchema, RecordField, RecordFieldOrder, RecordSchema, UnionSchema,
    };
    use apache_avro::types::Value;
    use apache_avro::{Schema, Writer};
    use polars::frame::DataFrame;
    use polars::prelude::{IntoLazy, UnionArgs, concat};
    use polars_plan::plans::ScanSources;
    use polars_utils::mmap::MemSlice;

    use crate::Error;

    use super::AvroScanner;

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
        concat(frames, UnionArgs::default())
            .unwrap()
            .collect()
            .unwrap()
    }

    #[test]
    fn test_scan() {
        let scanner =
            AvroScanner::new_from_sources(&from_paths(["./resources/food.avro"]), false, None)
                .unwrap();
        let frame = read_scan(scanner);
        assert_eq!(frame.height(), 27);
        assert_eq!(frame.schema().len(), 4);
    }

    #[test]
    fn test_glob() {
        let scanner =
            AvroScanner::new_from_sources(&from_paths(["./resources/*.avro"]), true, None).unwrap();
        let frame = read_scan(scanner);
        assert_eq!(frame.height(), 30);
        assert_eq!(frame.schema().len(), 4);
    }

    #[test]
    fn test_non_record_avro() {
        let mut buff = Cursor::new(Vec::new());
        let mut writer = Writer::new(&Schema::Boolean, &mut buff);
        writer.append(true).unwrap();
        writer.flush().unwrap();
        let bytes = buff.into_inner();
        let res = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into_boxed_slice().into()),
            false,
            None,
        );
        assert!(matches!(res, Err(Error::NonRecordSchema(_))));
    }

    #[test]
    fn test_union_avro() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::Record(RecordSchema {
            name: "base".into(),
            aliases: None,
            doc: None,
            fields: vec![RecordField {
                name: "a".into(),
                doc: None,
                aliases: None,
                default: None,
                schema: Schema::Union(
                    UnionSchema::new(vec![Schema::Boolean, Schema::Int]).unwrap(),
                ),
                order: RecordFieldOrder::Ignore,
                position: 0,
                custom_attributes: BTreeMap::new(),
            }],
            lookup: [("a".into(), 0)].into(),
            attributes: BTreeMap::new(),
        });
        let mut writer = Writer::new(&schema, &mut buff);
        writer
            .append(Value::Record(vec![(
                "a".into(),
                Value::Union(0, Box::new(Value::Boolean(true))),
            )]))
            .unwrap();
        writer.flush().unwrap();
        let bytes = buff.into_inner();
        let res = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into_boxed_slice().into()),
            false,
            None,
        );
        assert!(matches!(res, Err(Error::UnsupportedAvroType(_))));
    }

    #[test]
    fn test_null_union_avro() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::Record(RecordSchema {
            name: "base".into(),
            aliases: None,
            doc: None,
            fields: vec![RecordField {
                name: "a".into(),
                doc: None,
                aliases: None,
                default: None,
                schema: Schema::Union(UnionSchema::new(vec![Schema::Null]).unwrap()),
                order: RecordFieldOrder::Ignore,
                position: 0,
                custom_attributes: BTreeMap::new(),
            }],
            lookup: [("a".into(), 0)].into(),
            attributes: BTreeMap::new(),
        });
        let mut writer = Writer::new(&schema, &mut buff);
        writer
            .append(Value::Record(vec![(
                "a".into(),
                Value::Union(0, Box::new(Value::Null)),
            )]))
            .unwrap();
        writer.flush().unwrap();
        let bytes = buff.into_inner();
        let scanner = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into_boxed_slice().into()),
            false,
            None,
        )
        .unwrap();
        read_scan(scanner);
    }

    #[test]
    fn test_fixed_avro() {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::Record(RecordSchema {
            name: "base".into(),
            aliases: None,
            doc: None,
            fields: vec![RecordField {
                name: "a".into(),
                doc: None,
                aliases: None,
                default: None,
                schema: Schema::Fixed(FixedSchema {
                    name: "fixed".into(),
                    aliases: None,
                    doc: None,
                    size: 1,
                    default: None,
                    attributes: BTreeMap::new(),
                }),
                order: RecordFieldOrder::Ignore,
                position: 0,
                custom_attributes: BTreeMap::new(),
            }],
            lookup: [("a".into(), 0)].into(),
            attributes: BTreeMap::new(),
        });
        let mut writer = Writer::new(&schema, &mut buff);
        writer
            .append(Value::Record(vec![("a".into(), Value::Fixed(1, vec![0]))]))
            .unwrap();
        writer.flush().unwrap();
        let bytes = buff.into_inner();
        let res = AvroScanner::new_from_sources(
            &ScanSources::Buffers(vec![MemSlice::from_vec(bytes)].into_boxed_slice().into()),
            false,
            None,
        );
        assert!(matches!(res, Err(Error::UnsupportedAvroType(_))));
    }
}
