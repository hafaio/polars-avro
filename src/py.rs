//! pyo3 bindings

use std::borrow::Cow;
use std::iter::Fuse;
use std::sync::Arc;

use apache_avro::Codec as AvroCodec;
use polars::prelude::Schema;
use polars_io::cloud::CloudOptions;
use polars_io::cloud::credential_provider::PlCredentialProvider;
use polars_plan::prelude::ScanSources;
use polars_python::prelude::Wrap;
use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError};
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{
    Bound, PyErr, PyObject, PyResult, Python, create_exception, pyclass, pyfunction, pymethods,
    pymodule, wrap_pyfunction,
};
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PySchema};

use crate::{AvroIter, AvroScanner, Error, WriteOptions, sink_avro};

fn parse_cloud_options(
    sources: &ScanSources,
    cloud_options: Option<Vec<(String, String)>>,
    credential_provider: Option<PyObject>,
    retries: usize,
    file_cache_ttl: Option<u64>,
) -> Result<Option<CloudOptions>, Error> {
    match (sources.first_path(), cloud_options) {
        (Some(first_path), Some(cloud_options)) => {
            let mut cloud_options =
                CloudOptions::from_untyped_config(&first_path.to_string_lossy(), cloud_options)?;
            cloud_options = cloud_options
                .with_max_retries(retries)
                .with_credential_provider(
                    credential_provider.map(PlCredentialProvider::from_python_func_object),
                );

            if let Some(file_cache_ttl) = file_cache_ttl {
                cloud_options.file_cache_ttl = file_cache_ttl;
            }
            Ok(Some(cloud_options))
        }
        (None, _) | (_, None) => Ok(None),
    }
}

enum CloudParams {
    Init {
        cloud_options: Option<Vec<(String, String)>>,
        credential_provider: Option<PyObject>,
        retries: usize,
        file_cache_ttl: Option<u64>,
    },
    Parsed(Option<CloudOptions>),
}

impl CloudParams {
    fn as_options<'a>(
        &'a mut self,
        sources: &ScanSources,
    ) -> Result<Option<&'a CloudOptions>, Error> {
        if let CloudParams::Init {
            cloud_options,
            credential_provider,
            retries,
            file_cache_ttl,
        } = self
        {
            let options = parse_cloud_options(
                sources,
                cloud_options.take(),
                credential_provider.take(),
                *retries,
                *file_cache_ttl,
            )?;
            *self = CloudParams::Parsed(options);
        }
        match self {
            CloudParams::Init { .. } => unreachable!(),
            CloudParams::Parsed(options) => Ok(options.as_ref()),
        }
    }
}

#[pyclass]
pub struct PyAvroIter(Fuse<AvroIter>);

#[pymethods]
impl PyAvroIter {
    fn next(&mut self) -> PyResult<Option<PyDataFrame>> {
        let PyAvroIter(inner) = self;
        Ok(inner.next().transpose().map(|op| op.map(PyDataFrame))?)
    }
}

#[pyclass]
pub struct AvroSource {
    sources: ScanSources,
    glob: bool,
    cloud_params: CloudParams,
    schema: Option<Arc<Schema>>,
    last_scanner: Option<AvroScanner>,
}

impl AvroSource {
    /// If we created a scanner to get the schema, then take it, otherwise create a new one.
    fn take_scanner(&mut self) -> Result<AvroScanner, Error> {
        if let Some(scanner) = self.last_scanner.take() {
            Ok(scanner)
        } else {
            let scanner = AvroScanner::new_from_sources(
                &self.sources,
                self.glob,
                self.cloud_params.as_options(&self.sources)?,
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
    #[pyo3(signature = (sources, glob, cloud_options, credential_provider, retries, file_cache_ttl))]
    fn new(
        sources: Wrap<ScanSources>,
        glob: bool,
        cloud_options: Option<Vec<(String, String)>>,
        credential_provider: Option<PyObject>,
        retries: usize,
        file_cache_ttl: Option<u64>,
    ) -> Self {
        let Wrap(sources) = sources;
        Self {
            sources,
            glob,
            cloud_params: CloudParams::Init {
                cloud_options,
                credential_provider,
                retries,
                file_cache_ttl,
            },
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
                    let new_scanner = AvroScanner::new_from_sources(
                        &self.sources,
                        self.glob,
                        self.cloud_params.as_options(&self.sources)?,
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

impl From<Codec> for AvroCodec {
    fn from(codec: Codec) -> Self {
        match codec {
            Codec::Null => AvroCodec::Null,
            Codec::Deflate => AvroCodec::Deflate,
            Codec::Snappy => AvroCodec::Snappy,
            Codec::Bzip2 => AvroCodec::Bzip2,
            Codec::Xz => AvroCodec::Xz,
            Codec::Zstandard => AvroCodec::Zstandard,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (frames, dest, codec, promote_ints, promote_array, truncate_time, cloud_options, credential_provider, retries))]
#[allow(clippy::too_many_arguments)]
fn write_avro(
    py: Python,
    frames: Vec<PyDataFrame>,
    dest: PyObject,
    codec: Codec,
    promote_ints: bool,
    promote_array: bool,
    truncate_time: bool,
    cloud_options: Option<Vec<(String, String)>>,
    credential_provider: Option<PyObject>,
    retries: usize,
) -> PyResult<()> {
    let cloud_options = if let Ok(path) = dest.extract::<Cow<str>>(py) {
        let base_options =
            CloudOptions::from_untyped_config(&path, cloud_options.unwrap_or_default())
                .map_err(Error::Polars)?;
        Some(
            base_options
                .with_max_retries(retries)
                .with_credential_provider(
                    credential_provider.map(PlCredentialProvider::from_python_func_object),
                ),
        )
    } else {
        None
    };

    let dest = polars_python::file::try_get_writeable(dest, cloud_options.as_ref())?;
    sink_avro(
        frames.into_iter().map(|PyDataFrame(frame)| frame),
        dest,
        WriteOptions {
            codec: codec.into(),
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
    m.add_function(wrap_pyfunction!(write_avro, m)?)?;
    Ok(())
}
