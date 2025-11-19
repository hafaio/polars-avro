//! pyo3 bindings

use miniz_oxide::deflate::CompressionLevel;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::iter::{Chain, Fuse};
use std::sync::Arc;

use apache_avro::{
    Bzip2Settings, Codec as AvroCodec, DeflateSettings, XzSettings, ZstandardSettings,
};
use polars::prelude::file::Writeable;
use polars::prelude::{PlPathRef, PlSmallStr, Schema};
use polars_io::cloud::CloudOptions;
use pyo3::exceptions::{PyException, PyIOError, PyRuntimeError, PyValueError};
use pyo3::types::{PyAnyMethods, PyBytes, PyBytesMethods, PyModule, PyModuleMethods};
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python, create_exception, pyclass, pyfunction, pymethods,
    pymodule, wrap_pyfunction,
};
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PySchema};

use crate::{AvroIter, AvroScanner, Error, WriteOptions, sink_avro};

enum ScanSource {
    File(BufReader<File>),
    Bytes(BufReader<PyReader>),
}

impl Read for ScanSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            ScanSource::File(reader) => reader.read(buf),
            ScanSource::Bytes(cursor) => cursor.read(buf),
        }
    }
}
struct BytesIter {
    buffs: Arc<[Arc<Py<PyAny>>]>,
    idx: usize,
}

impl BytesIter {
    fn new(buffs: Arc<[Arc<Py<PyAny>>]>) -> Self {
        Self { buffs, idx: 0 }
    }
}

impl Iterator for BytesIter {
    type Item = Result<ScanSource, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffs.get(self.idx).map(|buff| {
            self.idx += 1;
            Ok(ScanSource::Bytes(BufReader::new(PyReader(buff.clone()))))
        })
    }
}

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
                Ok(file) => Ok(ScanSource::File(BufReader::new(file))),
                Err(err) => Err(Error::IO(err, path.clone())),
            }
        })
    }
}

type SourceIter = Chain<BytesIter, PathIter>;

#[pyclass]
pub struct PyAvroIter(Fuse<AvroIter<ScanSource, SourceIter>>);

#[pymethods]
impl PyAvroIter {
    fn next(&mut self) -> PyResult<Option<PyDataFrame>> {
        let PyAvroIter(inner) = self;
        Ok(inner.next().transpose().map(|op| op.map(PyDataFrame))?)
    }
}

struct PyReader(Arc<Py<PyAny>>);

impl Read for PyReader {
    fn read(&mut self, mut buf: &mut [u8]) -> io::Result<usize> {
        Python::attach(|py| {
            let res = self.0.bind(py).call_method1("read", (buf.len(),))?;
            let bytes = res.downcast_into::<PyBytes>()?;
            let raw = bytes.as_bytes();
            buf.write_all(raw)?;
            Ok(raw.len())
        })
        .map_err(|err: PyErr| io::Error::other(err.to_string()))
    }
}

struct PyWriter(Py<PyAny>);

impl Write for PyWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Python::attach(|py| {
            // let inp = PyBytes::new(py, buf); // FIXME necessarry?
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

impl Seek for PyWriter {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
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

#[pyclass]
pub struct AvroSource {
    paths: Arc<[String]>,
    buffs: Arc<[Arc<Py<PyAny>>]>,
    single_col_name: Option<PlSmallStr>,
    schema: Option<Arc<Schema>>,
    last_scanner: Option<AvroScanner<ScanSource, SourceIter>>,
}

impl AvroSource {
    /// If we created a scanner to get the schema, then take it, otherwise create a new one.
    fn take_scanner(&mut self) -> Result<AvroScanner<ScanSource, SourceIter>, Error> {
        if let Some(scanner) = self.last_scanner.take() {
            Ok(scanner)
        } else {
            // create a new scanner from the sources
            let scanner = AvroScanner::try_new(
                BytesIter::new(self.buffs.clone()).chain(PathIter::new(self.paths.clone())),
                self.single_col_name.clone(),
            )?;
            // ensure we store the schema
            if self.schema.is_none() {
                self.schema = Some(scanner.schema());
            }
            Ok(scanner)
        }
    }
}

#[pymethods]
impl AvroSource {
    #[new]
    #[pyo3(signature = (paths, buffs, single_col_name))]
    fn new(paths: Vec<String>, buffs: Vec<Py<PyAny>>, single_col_name: Option<String>) -> Self {
        Self {
            paths: paths.into(),
            buffs: buffs.into_iter().map(Arc::new).collect(),
            single_col_name: single_col_name.map(PlSmallStr::from),
            schema: None,
            last_scanner: None,
        }
    }

    fn schema(&mut self) -> PyResult<PySchema> {
        Ok(PySchema(match &mut self.schema {
            Some(schema) => schema.clone(),
            loc @ None => {
                let new_schema = if let Some(scanner) = &self.last_scanner {
                    scanner.schema()
                } else {
                    let new_scanner = AvroScanner::try_new(
                        BytesIter::new(self.buffs.clone()).chain(PathIter::new(self.paths.clone())),
                        self.single_col_name.clone(),
                    )?;
                    let schema = new_scanner.schema();
                    self.last_scanner = Some(new_scanner);
                    schema
                };
                loc.insert(new_schema).clone()
            }
        }))
    }

    #[pyo3(signature = (batch_size, with_columns))]
    #[allow(clippy::needless_pass_by_value)]
    fn batch_iter(
        &mut self,
        batch_size: usize,
        with_columns: Option<Vec<String>>,
    ) -> PyResult<PyAvroIter> {
        let scanner = self.take_scanner()?;
        let iter = scanner.try_into_iter(batch_size, with_columns.as_deref())?;
        Ok(PyAvroIter(iter))
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

fn create_codec(codec: Codec, compression_level: Option<u8>) -> AvroCodec {
    match codec {
        Codec::Null => AvroCodec::Null,
        Codec::Deflate => AvroCodec::Deflate(if let Some(compression_level) = compression_level {
            DeflateSettings::new(match compression_level {
                0 => CompressionLevel::NoCompression,
                1 => CompressionLevel::BestSpeed,
                9 => CompressionLevel::BestCompression,
                10 => CompressionLevel::UberCompression,
                6 => CompressionLevel::DefaultLevel,
                _ => CompressionLevel::DefaultCompression,
            })
        } else {
            DeflateSettings::default()
        }),
        Codec::Snappy => AvroCodec::Snappy,
        Codec::Bzip2 => AvroCodec::Bzip2(if let Some(compression_level) = compression_level {
            Bzip2Settings { compression_level }
        } else {
            Bzip2Settings::default()
        }),
        Codec::Xz => AvroCodec::Xz(if let Some(compression_level) = compression_level {
            XzSettings { compression_level }
        } else {
            XzSettings::default()
        }),
        Codec::Zstandard => {
            AvroCodec::Zstandard(if let Some(compression_level) = compression_level {
                ZstandardSettings { compression_level }
            } else {
                ZstandardSettings::default()
            })
        }
    }
}

// TODO Add credentials when stabalized
#[pyfunction]
#[pyo3(signature = (frames, dest, codec, promote_ints, promote_array, truncate_time, compression_level, cloud_options))]
#[allow(clippy::too_many_arguments)]
fn write_avro_file(
    frames: Vec<PyDataFrame>,
    dest: &str,
    codec: Codec,
    promote_ints: bool,
    promote_array: bool,
    truncate_time: bool,
    compression_level: Option<u8>,
    cloud_options: Option<Vec<(String, String)>>,
) -> PyResult<()> {
    let dest = PlPathRef::new(dest);
    let scheme = dest.as_cloud_path().map(|cp| cp.scheme());
    let options =
        CloudOptions::from_untyped_config(scheme.as_ref(), cloud_options.unwrap_or_default())
            .map_err(PyPolarsErr::from)?;
    let mut write = Writeable::try_new(dest, Some(&options)).map_err(PyPolarsErr::from)?;
    sink_avro(
        frames.into_iter().map(|PyDataFrame(frame)| frame),
        &mut *write,
        WriteOptions {
            codec: create_codec(codec, compression_level),
            promote_ints,
            promote_array,
            truncate_time,
        },
    )?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (frames, buff, codec, promote_ints, promote_array, truncate_time, compression_level))]
#[allow(clippy::too_many_arguments)]
fn write_avro_buff(
    frames: Vec<PyDataFrame>,
    buff: Py<PyAny>,
    codec: Codec,
    promote_ints: bool,
    promote_array: bool,
    truncate_time: bool,
    compression_level: Option<u8>,
) -> PyResult<()> {
    let buff = BufWriter::new(PyWriter(buff));
    sink_avro(
        frames.into_iter().map(|PyDataFrame(frame)| frame),
        buff,
        WriteOptions {
            codec: create_codec(codec, compression_level),
            promote_ints,
            promote_array,
            truncate_time,
        },
    )?;
    Ok(())
}

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        match value {
            Error::Polars(err) => PyPolarsErr::from(err).into(),
            Error::Avro(err) => AvroError::new_err(err.to_string()),
            Error::EmptySources => EmptySources::new_err("must scan at least one source"),
            Error::NonRecordSchema(schema) => {
                AvroSpecError::new_err(format!("top level avro schema must be a record: {schema}",))
            }
            Error::UnsupportedAvroType(schema) => {
                AvroSpecError::new_err(format!("unsupported type in read conversion: {schema}"))
            }
            Error::UnsupportedPolarsType(data_type) => {
                AvroSpecError::new_err(format!("unsupported type in write conversion: {data_type}"))
            }
            Error::NullEnum => AvroSpecError::new_err("enum schema contained null fields"),
            Error::MissingRefName(name) => {
                AvroSpecError::new_err(format!("couldn't find referenced {name} in schema"))
            }
            Error::NonMatchingSchemas => {
                AvroSpecError::new_err("encountered non-identical schemas in same batch")
            }
            Error::InvalidArrowType(data_type, arrow_data_type) => {
                PyRuntimeError::new_err(format!(
                    "encountered unhandled type conversion: {data_type} from {arrow_data_type:?}"
                ))
            }
            Error::InvalidAvroValue(value) => AvroSpecError::new_err(format!(
                "tried to deserialize an avro value that doesn't match the spec: {value:?}"
            )),
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
    m.add_class::<Codec>()?;
    m.add("AvroError", py.get_type::<AvroError>())?;
    m.add("EmptySources", py.get_type::<EmptySources>())?;
    m.add("AvroSpecError", py.get_type::<AvroSpecError>())?;
    m.add_function(wrap_pyfunction!(write_avro_file, m)?)?;
    m.add_function(wrap_pyfunction!(write_avro_buff, m)?)?;
    Ok(())
}
