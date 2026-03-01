//! pyo3 bindings

use super::{Error, ReadOptions, Reader, Writer, get_schema};
use arrow_avro::compression::CompressionCodec;
use polars::prelude::{PlSmallStr, Schema};
use pyo3::exceptions::{PyException, PyIOError, PyIndexError, PyKeyError, PyValueError};
use pyo3::types::{PyAnyMethods, PyBytes, PyBytesMethods, PyModule, PyModuleMethods};
use pyo3::{
    Bound, FromPyObject, Py, PyAny, PyErr, PyResult, Python, create_exception, pyclass, pymethods,
    pymodule,
};
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PyDataType, PySchema};
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
    fn next(&mut self) -> PyResult<Option<PyDataFrame>> {
        Ok(self.0.next().transpose().map(|op| op.map(PyDataFrame))?)
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

#[pyclass]
#[derive(Debug, Clone)]
pub struct AvroSource {
    paths: Arc<[String]>,
    buffs: Arc<[(PyIO, u64)]>,
    schema: Option<Arc<Schema>>,
}

impl AvroSource {
    fn get_sources(&self) -> SourceIter {
        PathIter::new(self.paths.clone()).chain(BytesIter::new(self.buffs.clone()))
    }

    fn get_schema(&mut self) -> Result<Arc<Schema>, Error> {
        if let Some(schema) = &self.schema {
            Ok(schema.clone())
        } else {
            let first = self.get_sources().next().ok_or(Error::EmptySources)??;
            let schema = Arc::new(get_schema(BufReader::new(first))?);
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

    #[pyo3(signature = ())]
    fn schema(&mut self) -> PyResult<PySchema> {
        Ok(PySchema(self.get_schema()?.clone()))
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
                    .iter_names()
                    .map(PlSmallStr::to_string)
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

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    Null,
    Deflate,
    Snappy,
    Bzip2,
    Xz,
    Zstandard,
}

impl From<Codec> for Option<CompressionCodec> {
    fn from(obj: Codec) -> Self {
        match obj {
            Codec::Null => None,
            Codec::Deflate => Some(CompressionCodec::Deflate),
            Codec::Snappy => Some(CompressionCodec::Snappy),
            Codec::Bzip2 => Some(CompressionCodec::Bzip2),
            Codec::Xz => Some(CompressionCodec::Xz),
            Codec::Zstandard => Some(CompressionCodec::ZStandard),
        }
    }
}

struct PySchemaRef(Vec<(String, PyDataType)>);

impl<'a, 'py> FromPyObject<'a, 'py> for PySchemaRef {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self(Vec::<(String, PyDataType)>::extract(obj)?))
    }
}

impl From<PySchemaRef> for Schema {
    fn from(obj: PySchemaRef) -> Self {
        let mut schema = Schema::with_capacity(obj.0.len());
        for (name, PyDataType(dtype)) in obj.0 {
            schema.insert(name.into(), dtype);
        }
        schema
    }
}

// TODO Add credentials when stabilized
#[pyclass]
pub struct AvroFileSink(Writer<BufWriter<File>>);

#[pymethods]
impl AvroFileSink {
    #[new]
    #[pyo3(signature = (path, fields, codec))]
    fn new(path: &str, fields: PySchemaRef, codec: Codec) -> Result<Self, PyErr> {
        Ok(Self(Writer::try_new(
            BufWriter::new(File::create(path)?),
            &fields.into(),
            codec.into(),
        )?))
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (batch))]
    fn write(&mut self, batch: PyDataFrame) -> Result<(), PyErr> {
        Ok(self.0.write(&batch.0)?)
    }
}

#[pyclass]
pub struct AvroBuffSink(Writer<BufWriter<PyIO>>);

#[pymethods]
impl AvroBuffSink {
    #[new]
    #[pyo3(signature = (buff, fields, codec))]
    fn new(buff: Py<PyAny>, fields: PySchemaRef, codec: Codec) -> Result<Self, PyErr> {
        Ok(Self(Writer::try_new(
            BufWriter::new(PyIO(Arc::new(buff))),
            &fields.into(),
            codec.into(),
        )?))
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (batch))]
    fn write(&mut self, batch: PyDataFrame) -> Result<(), PyErr> {
        Ok(self.0.write(&batch.0)?)
    }
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        match value {
            Error::Polars(err) => PyPolarsErr::from(err).into(),
            Error::Arrow(err) => AvroError::new_err(err.to_string()),
            Error::Avro(err) => AvroError::new_err(err.to_string()),
            Error::EmptySources => EmptySources::new_err("must scan at least one source"),
            Error::NonRecordSchema => {
                AvroSpecError::new_err("top level avro schema must be a record")
            }
            Error::UnsupportedPolarsType(data_type) => {
                AvroSpecError::new_err(format!("unsupported type in write conversion: {data_type}"))
            }
            Error::NullEnum => AvroSpecError::new_err("enum schema contained null fields"),
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
#[pyo3(name = "_avro_rs")]
fn polars_avro(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<AvroSource>()?;
    m.add_class::<AvroFileSink>()?;
    m.add_class::<AvroBuffSink>()?;
    m.add_class::<Codec>()?;
    m.add("AvroError", py.get_type::<AvroError>())?;
    m.add("EmptySources", py.get_type::<EmptySources>())?;
    m.add("AvroSpecError", py.get_type::<AvroSpecError>())?;
    Ok(())
}
