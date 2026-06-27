//! pyo3 bindings

use super::{Error, ReadOptions, Reader, Writer, get_schema};
use arrow::datatypes::SchemaRef;
use arrow_avro::compression::CompressionCodec;
use pyo3::exceptions::{PyException, PyIOError, PyIndexError, PyKeyError, PyValueError};
use pyo3::types::{PyAnyMethods, PyBytes, PyBytesMethods, PyModule, PyModuleMethods};
use pyo3::{
    Bound, Py, PyAny, PyErr, PyRef, PyResult, Python, create_exception, pyclass, pymethods,
    pymodule,
};
use pyo3_arrow::{PyRecordBatch, PySchema};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::iter::{Chain, Fuse};
use std::sync::Arc;

#[derive(Debug)]
enum ScanSource {
    File(File),
    Bytes(PyIO),
}

impl Read for ScanSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            ScanSource::File(reader) => reader.read(buf),
            ScanSource::Bytes(cursor) => cursor.read(buf),
        }
    }
}

impl Seek for ScanSource {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            ScanSource::File(reader) => reader.seek(pos),
            ScanSource::Bytes(cursor) => cursor.seek(pos),
        }
    }
}

#[derive(Debug)]
struct BytesIter {
    buffs: Arc<[(PyIO, u64)]>,
    idx: usize,
}

impl BytesIter {
    fn new(buffs: Arc<[(PyIO, u64)]>) -> Self {
        Self { buffs, idx: 0 }
    }
}

impl Iterator for BytesIter {
    type Item = Result<ScanSource, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffs.get(self.idx).map(|(buff, pos)| {
            self.idx += 1;
            // reset the buffer
            // NOTE there's a race condition here if two things are iterating
            // over this at the same time, but that _shouldn't_ happen
            buff.py_seek(SeekFrom::Start(*pos))?;
            Ok(ScanSource::Bytes(buff.clone()))
        })
    }
}

#[derive(Debug)]
struct PathIter {
    paths: Arc<[String]>,
    idx: usize,
}

impl PathIter {
    fn new(paths: Arc<[String]>) -> Self {
        Self { paths, idx: 0 }
    }
}

impl Iterator for PathIter {
    type Item = Result<ScanSource, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.paths.get(self.idx).map(|path| {
            self.idx += 1;
            match File::open(path) {
                Ok(file) => Ok(ScanSource::File(file)),
                Err(err) => Err(Error::IO(err, path.clone())),
            }
        })
    }
}

type SourceIter = Chain<PathIter, BytesIter>;

#[pyclass]
#[derive(Debug)]
pub struct PyAvroIter(Fuse<Reader<ScanSource, SourceIter, Vec<String>>>);

#[pymethods]
impl PyAvroIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    // Returning `Ok(None)` raises `StopIteration`.
    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.0.next().transpose()? {
            Some(batch) => Ok(Some(PyRecordBatch::from(batch).into_pyarrow(py)?)),
            None => Ok(None),
        }
    }
}

#[derive(Debug, Clone)]
struct PyIO(Arc<Py<PyAny>>);

impl PyIO {
    // readonly seek
    fn py_seek(&self, pos: SeekFrom) -> io::Result<u64> {
        match pos {
            SeekFrom::Start(pos) => Python::attach(|py| {
                let writer = self.0.bind(py);
                let res = writer.call_method1("seek", (pos,))?;
                res.extract()
            })
            .map_err(|err: PyErr| io::Error::other(err.to_string())),
            SeekFrom::Current(offset) => Python::attach(|py| {
                let writer = self.0.bind(py);
                let res = writer.call_method0("tell")?;
                let current: u64 = res.extract()?;
                let pos = if offset < 0 {
                    current.saturating_sub(offset.unsigned_abs())
                } else {
                    current.saturating_add(offset.unsigned_abs())
                };
                let res = writer.call_method1("seek", (pos,))?;
                res.extract()
            })
            .map_err(|err: PyErr| io::Error::other(err.to_string())),
            SeekFrom::End(_) => Err(io::Error::new(
                ErrorKind::Unsupported,
                "seeking from end is not supported",
            )),
        }
    }
}

impl Read for PyIO {
    fn read(&mut self, mut buf: &mut [u8]) -> io::Result<usize> {
        Python::attach(|py| {
            let res = self.0.bind(py).call_method1("read", (buf.len(),))?;
            let bytes = res.cast_into::<PyBytes>()?;
            let raw = bytes.as_bytes();
            buf.write_all(raw)?;
            Ok(raw.len())
        })
        .map_err(|err: PyErr| io::Error::other(err.to_string()))
    }
}

impl Write for PyIO {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Python::attach(|py| {
            let res = self.0.bind(py).call_method1("write", (buf,))?;
            res.extract()
        })
        .map_err(|err: PyErr| io::Error::other(err.to_string()))
    }

    fn flush(&mut self) -> io::Result<()> {
        Python::attach(|py| {
            self.0.bind(py).call_method0("flush")?;
            Ok(())
        })
        .map_err(|err: PyErr| io::Error::other(err.to_string()))
    }
}

impl Seek for PyIO {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.py_seek(pos)
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct AvroSource {
    paths: Arc<[String]>,
    buffs: Arc<[(PyIO, u64)]>,
    schema: Option<SchemaRef>,
}

impl AvroSource {
    fn get_sources(&self) -> SourceIter {
        PathIter::new(self.paths.clone()).chain(BytesIter::new(self.buffs.clone()))
    }

    fn get_schema(&mut self) -> Result<SchemaRef, Error> {
        if let Some(schema) = &self.schema {
            Ok(schema.clone())
        } else {
            let first = self.get_sources().next().ok_or(Error::EmptySources)??;
            let schema = get_schema(BufReader::new(first))?;
            self.schema = Some(schema.clone());
            Ok(schema)
        }
    }
}

#[pymethods]
impl AvroSource {
    #[new]
    #[pyo3(signature = (paths, buffs))]
    fn new(paths: Vec<String>, buffs: Vec<Py<PyAny>>) -> Result<Self, PyErr> {
        Ok(Self {
            paths: paths.into(),
            buffs: buffs
                .into_iter()
                .map(|obj| {
                    let mut buff = PyIO(Arc::new(obj));
                    buff.stream_position().map(move |pos| (buff, pos))
                })
                .collect::<Result<_, _>>()?,
            schema: None,
        })
    }

    /// Return the file schema as a pyarrow `Schema`.
    #[pyo3(signature = ())]
    fn schema<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PySchema::new(self.get_schema()?).into_pyarrow(py)
    }

    #[pyo3(signature = (strict, utf8_view, batch_size, with_columns))]
    #[allow(clippy::needless_pass_by_value)]
    fn batch_iter(
        &mut self,
        strict: bool,
        utf8_view: bool,
        batch_size: usize,
        with_columns: Option<Vec<String>>,
    ) -> PyResult<PyAvroIter> {
        let projection = if with_columns.is_none() && strict {
            let base_schema = self.get_schema()?;
            Some(
                base_schema
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect(),
            )
        } else {
            with_columns
        };
        Ok(PyAvroIter(
            Reader::try_new(
                self.get_sources(),
                ReadOptions {
                    strict,
                    utf8_view,
                    batch_size,
                    projection,
                },
            )?
            .fuse(),
        ))
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    Deflate,
    Snappy,
    Bzip2,
    Xz,
    Zstandard,
}

impl From<Codec> for CompressionCodec {
    fn from(obj: Codec) -> Self {
        match obj {
            Codec::Deflate => CompressionCodec::Deflate,
            Codec::Snappy => CompressionCodec::Snappy,
            Codec::Bzip2 => CompressionCodec::Bzip2,
            Codec::Xz => CompressionCodec::Xz,
            Codec::Zstandard => CompressionCodec::ZStandard,
        }
    }
}

#[pyclass]
pub struct AvroFileSink(Writer<BufWriter<File>>);

#[pymethods]
impl AvroFileSink {
    #[new]
    #[pyo3(signature = (path, schema, codec=None))]
    fn new(path: &str, schema: PySchema, codec: Option<Codec>) -> Result<Self, PyErr> {
        Ok(Self(Writer::try_new(
            BufWriter::new(File::create(path)?),
            schema.into_inner(),
            codec.map(CompressionCodec::from),
        )?))
    }

    #[pyo3(signature = (batch))]
    #[allow(clippy::needless_pass_by_value)]
    fn write(&mut self, batch: PyRecordBatch) -> Result<(), PyErr> {
        Ok(self.0.write(batch.as_ref())?)
    }

    #[pyo3(signature = ())]
    #[allow(clippy::unused_self)]
    fn close(&mut self) {}
}

#[pyclass]
pub struct AvroBuffSink(Writer<BufWriter<PyIO>>);

#[pymethods]
impl AvroBuffSink {
    #[new]
    #[pyo3(signature = (buff, schema, codec=None))]
    fn new(buff: Py<PyAny>, schema: PySchema, codec: Option<Codec>) -> Result<Self, PyErr> {
        Ok(Self(Writer::try_new(
            BufWriter::new(PyIO(Arc::new(buff))),
            schema.into_inner(),
            codec.map(CompressionCodec::from),
        )?))
    }

    #[pyo3(signature = (batch))]
    #[allow(clippy::needless_pass_by_value)]
    fn write(&mut self, batch: PyRecordBatch) -> Result<(), PyErr> {
        Ok(self.0.write(batch.as_ref())?)
    }

    #[pyo3(signature = ())]
    #[allow(clippy::unused_self)]
    fn close(&mut self) {}
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        match value {
            Error::Arrow(err) => AvroError::new_err(err.to_string()),
            Error::ArrowAvro(err) => AvroError::new_err(err.to_string()),
            Error::Avro(err) => AvroError::new_err(err.to_string()),
            Error::EmptySources => EmptySources::new_err("must scan at least one source"),
            Error::NonRecordSchema => {
                AvroSpecError::new_err("top level avro schema must be a record")
            }
            Error::LargeHeader => {
                AvroSpecError::new_err("header was too large to effectively parse")
            }
            e @ Error::NonMatchingSchemas { .. } => AvroSpecError::new_err(format!("{e}")),
            Error::ColumnNotFound(col) => {
                PyKeyError::new_err(format!("Column \"{col}\" not found in schema"))
            }
            Error::ColumnIndexOutOfBounds(ind) => {
                PyIndexError::new_err(format!("Column index {ind} is out of bounds"))
            }
            Error::IO(err, path) => PyIOError::new_err(format!("I/O error: {path}: {err}")),
        }
    }
}

create_exception!(exceptions, AvroError, PyException);
create_exception!(exceptions, EmptySources, PyValueError);
create_exception!(exceptions, AvroSpecError, PyValueError);

#[pymodule]
fn _avro_rs(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<AvroSource>()?;
    m.add_class::<AvroFileSink>()?;
    m.add_class::<AvroBuffSink>()?;
    m.add_class::<Codec>()?;
    m.add("AvroError", py.get_type::<AvroError>())?;
    m.add("EmptySources", py.get_type::<EmptySources>())?;
    m.add("AvroSpecError", py.get_type::<AvroSpecError>())?;
    Ok(())
}
