//! Zero-copy conversion between arrow and polars-arrow via Arrow C Data Interface.
//!
//! Both crates implement `#[repr(C)]` FFI structs with identical memory layout,
//! so we transmute between them for zero-copy conversion, at the cost of unsafe
//! rust in trusting the data layout.
//!
//! The FFI import/export calls below return `Result`, but here they round-trip
//! data we just produced ourselves through a layout-compatible transmute. A
//! failure would mean the arrow/polars C ABIs have diverged — the same
//! invariant the surrounding `unsafe` already relies on — so we `expect` rather
//! than surface an unrecoverable error to callers.

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
fn arrow_array_to_polars(array: &dyn Array) -> Box<dyn PlArray> {
    let data = array.to_data();
    let (ffi_array, ffi_schema) =
        arrow_ffi::to_ffi(&data).expect("arrow array must export to the C data interface");

    let pl_array: PlArrowArray =
        unsafe { mem::transmute::<FFI_ArrowArray, PlArrowArray>(ffi_array) };
    let pl_schema: PlArrowSchema =
        unsafe { mem::transmute::<FFI_ArrowSchema, PlArrowSchema>(ffi_schema) };

    let pl_field = unsafe {
        pl_ffi::import_field_from_c(&pl_schema).expect("exported schema must import into polars")
    };
    unsafe {
        pl_ffi::import_array_from_c(pl_array, pl_field.dtype)
            .expect("exported array must import into polars")
    }
}

/// Convert a polars-arrow array to an arrow array via FFI transmute.
fn polars_array_to_arrow(array: Box<dyn PlArray>, field: &PlField) -> ArrayRef {
    // polars NullArray has a buffer that arrow doesn't expect, so we convert manually
    if array.dtype() == &ArrowDataType::Null {
        Arc::new(NullArray::new(array.len()))
    } else {
        let pl_array = pl_ffi::export_array_to_c(array);
        let pl_schema = pl_ffi::export_field_to_c(field);

        let ffi_array: FFI_ArrowArray =
            unsafe { mem::transmute::<PlArrowArray, FFI_ArrowArray>(pl_array) };
        let ffi_schema: FFI_ArrowSchema =
            unsafe { mem::transmute::<PlArrowSchema, FFI_ArrowSchema>(pl_schema) };

        let data = unsafe {
            arrow_ffi::from_ffi(ffi_array, &ffi_schema)
                .expect("exported polars array must import into arrow")
        };
        make_array(data)
    }
}

/// Convert an arrow `RecordBatch` to a polars `DataFrame`.
pub fn recordbatch_to_dataframe(batch: &RecordBatch) -> DataFrame {
    let schema = batch.schema();
    let columns: Vec<Column> = schema
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, array)| {
            let pl_array = arrow_array_to_polars(array.as_ref());
            let series = Series::from_arrow(field.name().into(), pl_array)
                .expect("imported polars array must form a series");
            series.into()
        })
        .collect();

    DataFrame::new(batch.num_rows(), columns).expect("columns share the record batch's row count")
}

/// Convert a polars `DataFrame` to an arrow `RecordBatch`.
pub fn dataframe_to_recordbatch(df: &DataFrame) -> RecordBatch {
    let mut fields = Vec::with_capacity(df.width());
    let mut arrays = Vec::with_capacity(df.width());

    for col in df.columns() {
        let series = col.as_materialized_series().rechunk();
        let pl_field = series
            .dtype()
            .to_arrow_field(series.name().clone(), CompatLevel::newest());
        let chunk = series.to_arrow(0, CompatLevel::newest());
        let arrow_array = polars_array_to_arrow(chunk, &pl_field);
        let arrow_field = polars_field_to_arrow(&pl_field);

        fields.push(arrow_field);
        arrays.push(arrow_array);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).expect("arrays were built to match this schema")
}

/// Convert a polars-arrow Field to an arrow Field via FFI.
fn polars_field_to_arrow(pl_field: &PlField) -> Field {
    if pl_field.dtype == ArrowDataType::Null {
        Field::new(pl_field.name.as_str(), DataType::Null, true)
    } else {
        let pl_schema = pl_ffi::export_field_to_c(pl_field);
        let ffi_schema: FFI_ArrowSchema =
            unsafe { mem::transmute::<PlArrowSchema, FFI_ArrowSchema>(pl_schema) };
        Field::try_from(&ffi_schema).expect("exported polars field must import into arrow")
    }
}

/// Convert a polars schema to an arrow schema
pub fn polars_schema_to_arrow(pl_schema: &PlSchema) -> Schema {
    let mut fields = Vec::with_capacity(pl_schema.len());
    for pl_field in pl_schema.iter_fields() {
        let field = pl_field.to_arrow(CompatLevel::newest());
        fields.push(polars_field_to_arrow(&field));
    }
    Schema::new(fields)
}

/// Convert an arrow schema to a polars Schema.
pub fn arrow_schema_to_polars(schema: &Schema) -> PlSchema {
    let mut pl_schema = PlSchema::with_capacity(schema.fields().len());
    for field in schema.fields() {
        if field.data_type() == &DataType::Null {
            pl_schema.insert(field.name().into(), PlDataType::Null);
        } else {
            let ffi_schema = FFI_ArrowSchema::try_from(field.as_ref())
                .expect("arrow field must export to the C data interface");
            let pl_ffi: PlArrowSchema =
                unsafe { mem::transmute::<FFI_ArrowSchema, PlArrowSchema>(ffi_schema) };
            let pl_field = unsafe {
                pl_ffi::import_field_from_c(&pl_ffi)
                    .expect("exported schema must import into polars")
            };
            let dtype = PlDataType::from_arrow_field(&pl_field);
            pl_schema.insert(field.name().into(), dtype);
        }
    }
    pl_schema
}
