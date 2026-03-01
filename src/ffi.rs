//! Zero-copy conversion between arrow and polars-arrow via Arrow C Data Interface.
//!
//! Both crates implement `#[repr(C)]` FFI structs with identical memory layout,
//! so we transmute between them for zero-copy conversion, at the cost of unsafe
//! rust in trusting the data layout

use super::Error;
use arrow::array::{Array, ArrayRef, NullArray, RecordBatch, make_array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ffi::{self as arrow_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use polars::frame::DataFrame;
use polars::prelude::{Column, CompatLevel, DataType as PlDataType, Schema as PlSchema, SchemaExt};
use polars::series::Series;
use polars_arrow::array::Array as PlArray;
use polars_arrow::datatypes::{ArrowDataType, Field as PlField};
use polars_arrow::ffi::{self as pl_ffi, ArrowArray as PlArrowArray, ArrowSchema as PlArrowSchema};
use std::mem;
use std::sync::Arc;

/// Convert an arrow array to a polars-arrow array via FFI transmute.
fn arrow_array_to_polars(array: &dyn Array) -> Result<Box<dyn PlArray>, Error> {
    let data = array.to_data();
    let (ffi_array, ffi_schema) = arrow_ffi::to_ffi(&data)?;

    let pl_array: PlArrowArray =
        unsafe { mem::transmute::<FFI_ArrowArray, PlArrowArray>(ffi_array) };
    let pl_schema: PlArrowSchema =
        unsafe { mem::transmute::<FFI_ArrowSchema, PlArrowSchema>(ffi_schema) };

    let pl_field = unsafe { pl_ffi::import_field_from_c(&pl_schema)? };
    let result = unsafe { pl_ffi::import_array_from_c(pl_array, pl_field.dtype)? };
    Ok(result)
}

/// Convert a polars-arrow array to an arrow array via FFI transmute.
fn polars_array_to_arrow(array: Box<dyn PlArray>, field: &PlField) -> Result<ArrayRef, Error> {
    // polars NullArray has a buffer that arrow doesn't expect, so we convert manually
    let result = if array.dtype() == &ArrowDataType::Null {
        Arc::new(NullArray::new(array.len()))
    } else {
        let pl_array = pl_ffi::export_array_to_c(array);
        let pl_schema = pl_ffi::export_field_to_c(field);

        let ffi_array: FFI_ArrowArray =
            unsafe { mem::transmute::<PlArrowArray, FFI_ArrowArray>(pl_array) };
        let ffi_schema: FFI_ArrowSchema =
            unsafe { mem::transmute::<PlArrowSchema, FFI_ArrowSchema>(pl_schema) };

        let data = unsafe { arrow_ffi::from_ffi(ffi_array, &ffi_schema)? };
        make_array(data)
    };
    Ok(result)
}

/// Convert an arrow `RecordBatch` to a polars `DataFrame`.
pub fn recordbatch_to_dataframe(batch: &RecordBatch) -> Result<DataFrame, Error> {
    let schema = batch.schema();
    let columns: Vec<Column> = schema
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, array)| {
            let pl_array = arrow_array_to_polars(array.as_ref())?;
            let series = Series::from_arrow(field.name().into(), pl_array)?;
            Ok(series.into())
        })
        .collect::<Result<_, Error>>()?;

    Ok(DataFrame::new(batch.num_rows(), columns)?)
}

/// Convert a polars `DataFrame` to an arrow `RecordBatch`.
pub fn dataframe_to_recordbatch(df: &DataFrame) -> Result<RecordBatch, Error> {
    let mut fields = Vec::with_capacity(df.width());
    let mut arrays = Vec::with_capacity(df.width());

    for col in df.columns() {
        let series = col.as_materialized_series().rechunk();
        let pl_field = series
            .dtype()
            .to_arrow_field(series.name().clone(), CompatLevel::newest());
        let chunk = series.to_arrow(0, CompatLevel::newest());
        let arrow_array = polars_array_to_arrow(chunk, &pl_field)?;
        let arrow_field = polars_field_to_arrow(&pl_field)?;

        fields.push(arrow_field);
        arrays.push(arrow_array);
    }

    let schema = Arc::new(Schema::new(fields));
    Ok(RecordBatch::try_new(schema, arrays)?)
}

/// Convert a polars-arrow Field to an arrow Field via FFI.
fn polars_field_to_arrow(pl_field: &PlField) -> Result<Field, Error> {
    let field = if pl_field.dtype == ArrowDataType::Null {
        Field::new(pl_field.name.as_str(), DataType::Null, true)
    } else {
        let pl_schema = pl_ffi::export_field_to_c(pl_field);
        let ffi_schema: FFI_ArrowSchema =
            unsafe { mem::transmute::<PlArrowSchema, FFI_ArrowSchema>(pl_schema) };
        Field::try_from(&ffi_schema)?
    };
    Ok(field)
}

/// Convert a polars schema to an arrow schema
pub fn polars_schema_to_arrow(pl_schema: &PlSchema) -> Result<Schema, Error> {
    let mut fields = Vec::with_capacity(pl_schema.len());
    for pl_field in pl_schema.iter_fields() {
        let field = pl_field.to_arrow(CompatLevel::newest());
        fields.push(polars_field_to_arrow(&field)?);
    }
    Ok(Schema::new(fields))
}

/// Convert an arrow schema to a polars Schema.
pub fn arrow_schema_to_polars(schema: &Schema) -> Result<PlSchema, Error> {
    let mut pl_schema = PlSchema::with_capacity(schema.fields().len());
    for field in schema.fields() {
        if field.data_type() == &DataType::Null {
            pl_schema.insert(field.name().into(), PlDataType::Null);
        } else {
            let ffi_schema = FFI_ArrowSchema::try_from(field.as_ref())?;
            let pl_ffi: PlArrowSchema =
                unsafe { mem::transmute::<FFI_ArrowSchema, PlArrowSchema>(ffi_schema) };
            let pl_field = unsafe { pl_ffi::import_field_from_c(&pl_ffi)? };
            let dtype = PlDataType::from_arrow_field(&pl_field);
            pl_schema.insert(field.name().into(), dtype);
        }
    }
    Ok(pl_schema)
}
