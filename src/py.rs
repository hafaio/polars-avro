//! pyo3 bindings

use super::{Error, ReadOptions, Reader, Writer, get_schema};
use arrow_avro::compression::CompressionCodec;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStoreExt, PutPayload};
use polars::prelude::{PlSmallStr, Schema};
use polars_io::cloud::{CloudOptions, PolarsObjectStore, build_object_store};
use polars_io::pl_async::get_runtime;
use polars_utils::pl_path::CloudScheme;
use pyo3::exceptions::{PyException, PyIOError, PyIndexError, PyKeyError, PyValueError};
use pyo3::types::{PyAnyMethods, PyBytes, PyBytesMethods, PyModule, PyModuleMethods};
use pyo3::{
    Bound, FromPyObject, Py, PyAny, PyErr, PyResult, Python, create_exception, pyclass, pyfunction,
    pymethods, pymodule, wrap_pyfunction,
};
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PyDataType, PySchema};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::iter::{Chain, Fuse};
use std::sync::Arc;

#[derive(Debug)]
struct CloudReader {
    store: PolarsObjectStore,
    path: ObjectPath,
    position: u64,
    size: u64,
}

impl CloudReader {
    fn try_new(
        url: impl AsRef<str>,
        storage_options: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<Self, Error> {
        get_runtime().block_on(async {
            let options = if storage_options.is_empty() {
                None
            } else {
                let scheme = CloudScheme::from_path(url.as_ref());
                Some(CloudOptions::from_untyped_config(
                    scheme,
                    storage_options
                        .iter()
                        .map(|(k, v)| (k.as_ref(), v.as_ref())),
                )?)
            };
            let (cl, store) =
                build_object_store(url.as_ref().into(), options.as_ref(), false).await?;
            let path = ObjectPath::from(cl.prefix.as_str());
            let meta = store.head(&path).await?;
            Ok(Self {
                store,
                path,
                position: 0,
                size: meta.size as u64,
            })
        })
    }
}

impl Read for CloudReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.position >= self.size {
            return Ok(0);
        }
        let pos = usize::try_from(self.position).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "current file position doesn't fit in a usize",
            )
        })?;
        let end = (pos + buf.len()).min(usize::try_from(self.size).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "read too long for a usize")
        })?);
        let range = pos..end;
        let data = get_runtime()
            .block_on(self.store.get_range(&self.path, range))
            .map_err(io::Error::other)?;
        let n = data.len();
        buf[..n].copy_from_slice(&data);
        self.position += u64::try_from(n)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "read too long for a u64"))?;
        Ok(n)
    }
}

impl Seek for CloudReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos: u64 = match pos {
            SeekFrom::Start(p) => Ok(p),
            SeekFrom::Current(offset) => {
                if offset < 0 {
                    self.position
                        .checked_sub(offset.unsigned_abs())
                        .ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidInput, "seek before start of file")
                        })
                } else {
                    self.position
                        .checked_add(offset.unsigned_abs())
                        .ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidInput, "seek beyond u64")
                        })
                }
            }
            SeekFrom::End(offset) => {
                if offset > 0 {
                    self.size.checked_add(offset.unsigned_abs()).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "seek beyond u64")
                    })
                } else {
                    self.size.checked_sub(offset.unsigned_abs()).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "seek before start of file")
                    })
                }
            }
        }?;
        self.position = new_pos;
        Ok(new_pos)
    }
}

struct CloudWriter {
    store: PolarsObjectStore,
    path: ObjectPath,
}

impl CloudWriter {
    fn try_new(
        url: impl AsRef<str>,
        storage_options: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<Self, Error> {
        get_runtime().block_on(async {
            let options = if storage_options.is_empty() {
                None
            } else {
                let scheme = CloudScheme::from_path(url.as_ref());
                Some(CloudOptions::from_untyped_config(
                    scheme,
                    storage_options
                        .iter()
                        .map(|(k, v)| (k.as_ref(), v.as_ref())),
                )?)
            };
            let (cl, store) =
                build_object_store(url.as_ref().into(), options.as_ref(), false).await?;
            let path = ObjectPath::from(cl.prefix.as_str());
            Ok(Self { store, path })
        })
    }
}

impl Write for CloudWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        get_runtime()
            .block_on(async {
                let store = self.store.to_dyn_object_store().await;
                store.put(&self.path, PutPayload::from(buf.to_vec())).await
            })
            .map_err(io::Error::other)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Returns `true` when `url` begins with a recognised cloud scheme.
fn is_cloud_url(url: &str) -> bool {
    CloudScheme::from_path(url).is_some()
}

#[derive(Debug)]
enum ScanSource {
    File(File),
    Bytes(PyIO),
    Cloud(BufReader<CloudReader>),
}

impl Read for ScanSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            ScanSource::File(reader) => reader.read(buf),
            ScanSource::Bytes(cursor) => cursor.read(buf),
            ScanSource::Cloud(reader) => reader.read(buf),
        }
    }
}

impl Seek for ScanSource {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            ScanSource::File(reader) => reader.seek(pos),
            ScanSource::Bytes(cursor) => cursor.seek(pos),
            ScanSource::Cloud(reader) => reader.seek(pos),
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
    storage_options: Arc<[(String, String)]>,
    idx: usize,
}

impl PathIter {
    fn new(paths: Arc<[String]>, storage_options: Arc<[(String, String)]>) -> Self {
        Self {
            paths,
            storage_options,
            idx: 0,
        }
    }
}

impl Iterator for PathIter {
    type Item = Result<ScanSource, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.paths.get(self.idx).map(|path| {
            self.idx += 1;
            if is_cloud_url(path) {
                Ok(ScanSource::Cloud(BufReader::with_capacity(
                    4 * 1024 * 1024, // want a larger buffer for cloud reads
                    CloudReader::try_new(path, &self.storage_options)?,
                )))
            } else {
                match File::open(path) {
                    Ok(file) => Ok(ScanSource::File(file)),
                    Err(err) => Err(Error::IO(err, path.clone())),
                }
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
    storage_options: Arc<[(String, String)]>,
    schema: Option<Arc<Schema>>,
}

impl AvroSource {
    fn get_sources(&self) -> SourceIter {
        PathIter::new(self.paths.clone(), self.storage_options.clone())
            .chain(BytesIter::new(self.buffs.clone()))
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
    #[pyo3(signature = (paths, buffs, storage_options))]
    fn new(
        paths: Vec<String>,
        buffs: Vec<Py<PyAny>>,
        storage_options: Vec<(String, String)>,
    ) -> Result<Self, PyErr> {
        Ok(Self {
            paths: paths.into(),
            buffs: buffs
                .into_iter()
                .map(|obj| {
                    let mut buff = PyIO(Arc::new(obj));
                    buff.stream_position().map(move |pos| (buff, pos))
                })
                .collect::<Result<_, _>>()?,
            storage_options: storage_options.into(),
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

    #[pyo3(signature = ())]
    #[allow(clippy::unused_self)]
    fn close(&mut self) {}
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

    #[pyo3(signature = ())]
    #[allow(clippy::unused_self)]
    fn close(&mut self) {}
}

#[pyclass]
pub struct AvroCloudSink(Option<Writer<BufWriter<CloudWriter>>>);

#[pymethods]
impl AvroCloudSink {
    #[new]
    #[pyo3(signature = (url, fields, codec, storage_options))]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        url: &str,
        fields: PySchemaRef,
        codec: Codec,
        storage_options: Vec<(String, String)>,
    ) -> Result<Self, PyErr> {
        let cloud_writer = CloudWriter::try_new(url, &storage_options)?;
        Ok(Self(Some(Writer::try_new(
            BufWriter::with_capacity(4 * 1024 * 1024, cloud_writer),
            &fields.into(),
            codec.into(),
        )?)))
    }

    #[pyo3(signature = (batch))]
    #[allow(clippy::needless_pass_by_value)]
    fn write(&mut self, batch: PyDataFrame) -> Result<(), PyErr> {
        self.0
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("sink is already closed"))?
            .write(&batch.0)?;
        Ok(())
    }

    #[pyo3(signature = ())]
    fn close(&mut self) -> Result<(), PyErr> {
        let writer = self
            .0
            .take()
            .ok_or_else(|| PyIOError::new_err("sink is already closed"))?;
        let buf_writer = writer.into_inner()?;
        buf_writer
            .into_inner()
            .map_err(|e| PyIOError::new_err(format!("error flushing buffer: {e}")))?;
        Ok(())
    }
}

#[pyfunction]
#[pyo3(signature = (url,))]
fn py_is_cloud_url(url: &str) -> bool {
    is_cloud_url(url)
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        match value {
            Error::Polars(err) => PyPolarsErr::from(err).into(),
            Error::Arrow(err) => AvroError::new_err(err.to_string()),
            Error::ArrowAvro(err) => AvroError::new_err(err.to_string()),
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
    m.add_class::<AvroCloudSink>()?;
    m.add_class::<Codec>()?;
    m.add_function(wrap_pyfunction!(py_is_cloud_url, m)?)?;
    m.add("AvroError", py.get_type::<AvroError>())?;
    m.add("EmptySources", py.get_type::<EmptySources>())?;
    m.add("AvroSpecError", py.get_type::<AvroSpecError>())?;
    Ok(())
}
