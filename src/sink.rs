//! Rust sink implementation
use super::Error;
use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow_avro::compression::CompressionCodec;
use arrow_avro::writer::{AvroWriter, WriterBuilder};
use std::io::Write;

/// Incrementally write avro files.
///
/// Some arrow types (`Int8`, `Int16`, `UInt8`, `UInt16`, `UInt32`, `UInt64`,
/// `Time`, dictionary-encoded types) can't be written directly and must be cast
/// first — see the README for workarounds.
pub struct Writer<W: Write> {
    base: AvroWriter<W>,
    schema: SchemaRef,
}

impl<W: Write> Writer<W> {
    /// Create a writer with a schema and options
    ///
    /// # Errors
    /// If the writer can't be created
    pub fn try_new(
        writer: W,
        schema: SchemaRef,
        codec: Option<CompressionCodec>,
    ) -> Result<Self, Error> {
        let base: AvroWriter<_> = WriterBuilder::new(schema.as_ref().clone())
            .with_compression(codec)
            .build(writer)?;
        Ok(Writer { base, schema })
    }

    /// Finish writing, flushing any buffered data through the underlying writer.
    ///
    /// Idempotent and non-consuming, so a sink can flush on close without
    /// giving up ownership of the writer.
    ///
    /// # Errors
    /// If there were problems flushing the writer
    pub fn finish(&mut self) -> Result<(), Error> {
        self.base.finish()?;
        Ok(())
    }

    /// Finish writing and return the underlying writer.
    ///
    /// # Errors
    /// If there were problems flushing the writer
    pub fn into_inner(mut self) -> Result<W, Error> {
        self.finish()?;
        Ok(self.base.into_inner())
    }

    /// Write a single record batch
    ///
    /// # Errors
    /// If there were problems writing the batch, or the batch doesn't match the schema
    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), Error> {
        if batch.schema() == self.schema {
            Ok(self.base.write(batch)?)
        } else {
            Err(Error::NonMatchingSchemas {
                expected: self.schema.as_ref().clone(),
                actual: batch.schema(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Error;
    use super::Writer;
    use apache_avro::Reader;
    use arrow::array::{ArrayRef, Int32Array, RecordBatch, StringArray, UInt64Array};
    use std::mem;
    use std::sync::Arc;

    /// Build a single-column record batch (the field is nullable).
    fn batch(name: &str, array: ArrayRef) -> RecordBatch {
        RecordBatch::try_from_iter([(name, array)]).unwrap()
    }

    #[test]
    fn test_empty() {
        // building a writer emits the avro header even with no rows written
        let batch = batch("col", Arc::new(Int32Array::from(vec![1, 2, 3])));
        let mut dest = Vec::new();
        mem::drop(Writer::try_new(&mut dest, batch.schema(), None).unwrap());
        assert_ne!(dest.len(), 0);
    }

    #[test]
    fn test_into_inner() {
        let batch = batch("col", Arc::new(Int32Array::from(vec![1, 2, 3])));
        let mut writer = Writer::try_new(Vec::new(), batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        let buff = writer.into_inner().unwrap();
        let rows = Reader::new(&*buff).unwrap().count();
        assert_eq!(rows, 3);
    }

    #[test]
    fn test_diff_schemas_error() {
        let first = batch("a", Arc::new(Int32Array::from(vec![1, 2, 3])));
        let mut writer = Writer::try_new(Vec::new(), first.schema(), None).unwrap();
        writer.write(&first).unwrap();
        let second = batch("b", Arc::new(StringArray::from(vec!["a", "b", "c"])));
        let err = writer.write(&second).unwrap_err();
        assert!(matches!(err, Error::NonMatchingSchemas { .. }));
    }

    /// A u64 that doesn't fit in an avro long fails when encoding the batch,
    /// exercising the write error path (the writer builds fine first).
    #[test]
    fn test_write_error() {
        let batch = batch("uint64", Arc::new(UInt64Array::from(vec![u64::MAX])));
        let mut writer = Writer::try_new(Vec::new(), batch.schema(), None).unwrap();
        let err = writer.write(&batch).unwrap_err();
        assert!(matches!(err, Error::ArrowAvro(_)), "{err:?}");
    }

    /// A writer that can fail on writes or on flush, to exercise the I/O error
    /// paths in `try_new` (header write) and `into_inner` (finish flushes).
    #[derive(Debug)]
    struct FailingWriter {
        fail_write: bool,
        fail_flush: bool,
    }

    impl std::io::Write for FailingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.fail_write {
                return Err(std::io::Error::other("disk full"));
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            if self.fail_flush {
                return Err(std::io::Error::other("disk full"));
            }
            Ok(())
        }
    }

    #[test]
    fn test_build_io_error() {
        let batch = batch("col", Arc::new(Int32Array::from(vec![1, 2, 3])));
        // the writer fails on the very first write (the avro header)
        let result = Writer::try_new(
            FailingWriter {
                fail_write: true,
                fail_flush: false,
            },
            batch.schema(),
            None,
        );
        let Err(err) = result else {
            panic!("expected build to fail");
        };
        assert!(matches!(err, Error::ArrowAvro(_)), "{err:?}");
    }

    #[test]
    fn test_finish_io_error() {
        let batch = batch("col", Arc::new(Int32Array::from(vec![1, 2, 3])));
        // header and data write fine; finishing flushes and fails
        let mut writer = Writer::try_new(
            FailingWriter {
                fail_write: false,
                fail_flush: true,
            },
            batch.schema(),
            None,
        )
        .unwrap();
        writer.write(&batch).unwrap();
        let err = writer.into_inner().unwrap_err();
        assert!(matches!(err, Error::ArrowAvro(_)), "{err:?}");
    }
}
