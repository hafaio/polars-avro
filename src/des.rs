//! Utilidies for deserializing from from avro.

use std::any::Any;

use super::Error;
use apache_avro::Schema as AvroSchema;
use apache_avro::schema::{
    ArraySchema, DecimalSchema, EnumSchema, MapSchema, RecordField, RecordSchema,
};
use apache_avro::types::Value;
use polars::error::{PolarsError, PolarsResult};
use polars::prelude::{
    ArrowDataType, ArrowField, DataType, Field, Schema as PlSchema, TimeUnit, create_enum_dtype,
};
use polars_arrow::array::{
    Array, MutableArray, MutableBinaryViewArray, MutableBooleanArray, MutableListArray,
    MutableNullArray, MutablePrimitiveArray, StructArray, TryExtend, TryPush, Utf8ViewArray,
};
use polars_arrow::bitmap::MutableBitmap;

const KEY_FIELD: &str = "key";
const VALUE_FIELD: &str = "value";

pub fn try_from_schema(schema: &AvroSchema) -> Result<PlSchema, Error> {
    if let AvroSchema::Record(rec) = schema {
        Ok(rec
            .fields
            .iter()
            .map(|rf| {
                Ok(Field::new(
                    rf.name.as_str().into(),
                    try_from_dtype(&rf.schema)?,
                ))
            })
            .collect::<Result<_, Error>>()?)
    } else {
        Err(Error::NonRecordSchema(Box::new(schema.clone())))
    }
}

fn try_from_dtype(schema: &AvroSchema) -> Result<DataType, Error> {
    Ok(match schema {
        AvroSchema::Null => DataType::Null,
        AvroSchema::Boolean => DataType::Boolean,
        AvroSchema::Int => DataType::Int32,
        AvroSchema::Long => DataType::Int64,
        AvroSchema::Float => DataType::Float32,
        AvroSchema::Double => DataType::Float64,
        AvroSchema::Bytes | AvroSchema::Uuid | AvroSchema::Fixed(_) => DataType::Binary,
        AvroSchema::String => DataType::String,
        AvroSchema::Array(ArraySchema { items, .. }) => {
            DataType::List(Box::new(try_from_dtype(items)?))
        }
        AvroSchema::Record(RecordSchema { fields, .. }) => DataType::Struct(
            fields
                .iter()
                .map(|RecordField { name, schema, .. }| {
                    Ok(Field {
                        name: name.as_str().into(),
                        dtype: try_from_dtype(schema)?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?,
        ),
        AvroSchema::Enum(EnumSchema { symbols, .. }) => {
            create_enum_dtype(Utf8ViewArray::from_slice_values(symbols))
        }
        AvroSchema::Date => DataType::Date,
        AvroSchema::TimeMillis | AvroSchema::TimeMicros => DataType::Time,
        AvroSchema::TimestampMillis => {
            DataType::Datetime(TimeUnit::Milliseconds, Some("UTC".into()))
        }
        AvroSchema::TimestampMicros => {
            DataType::Datetime(TimeUnit::Microseconds, Some("UTC".into()))
        }
        AvroSchema::TimestampNanos => DataType::Datetime(TimeUnit::Nanoseconds, Some("UTC".into())),
        AvroSchema::LocalTimestampMillis => DataType::Datetime(TimeUnit::Milliseconds, None),
        AvroSchema::LocalTimestampMicros => DataType::Datetime(TimeUnit::Microseconds, None),
        AvroSchema::LocalTimestampNanos => DataType::Datetime(TimeUnit::Nanoseconds, None),
        AvroSchema::Union(union) => match union.variants() {
            [AvroSchema::Null, other] | [other, AvroSchema::Null] | [other] => {
                try_from_dtype(other)?
            }
            _ => return Err(Error::UnsupportedAvroType(Box::new(schema.clone()))),
        },
        AvroSchema::Map(MapSchema { types, .. }) => {
            DataType::List(Box::new(DataType::Struct(vec![
                Field::new(KEY_FIELD.into(), DataType::String),
                Field::new(VALUE_FIELD.into(), try_from_dtype(types)?),
            ])))
        }
        &AvroSchema::Decimal(DecimalSchema {
            precision, scale, ..
        }) => DataType::Decimal(Some(precision), Some(scale)),
        AvroSchema::BigDecimal | AvroSchema::Duration | AvroSchema::Ref { .. } => {
            return Err(Error::UnsupportedAvroType(Box::new(schema.clone())));
        }
    })
}

pub trait ValueBuilder: MutableArray {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()>;

    // NOTE this so maps can be deserialized as lists of structs without copies
    fn try_push_keyed_value(&mut self, _: &String, _: &Value) -> PolarsResult<()> {
        unreachable!();
    }
}

impl MutableArray for Box<dyn ValueBuilder> {
    fn dtype(&self) -> &ArrowDataType {
        self.as_ref().dtype()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.as_ref().validity()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        self.as_mut().as_box()
    }

    fn as_any(&self) -> &dyn Any {
        self.as_ref().as_any()
    }

    fn as_mut_any(&mut self) -> &mut dyn Any {
        self.as_mut().as_mut_any()
    }

    fn push_null(&mut self) {
        self.as_mut().push_null();
    }

    fn reserve(&mut self, additional: usize) {
        self.as_mut().reserve(additional);
    }

    fn shrink_to_fit(&mut self) {
        self.as_mut().shrink_to_fit();
    }
}

impl ValueBuilder for Box<dyn ValueBuilder> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        self.as_mut().try_push_value(value)
    }

    fn try_push_keyed_value(&mut self, key: &String, value: &Value) -> PolarsResult<()> {
        self.as_mut().try_push_keyed_value(key, value)
    }
}

/// Helper since values may be inside union types
fn unwrap_union(value: &Value) -> &Value {
    match value {
        Value::Union(_, inner) => inner,
        other => other,
    }
}

impl ValueBuilder for MutableNullArray {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected null but got {value:?}").into(),
                ));
            }
        }
        Ok(())
    }
}

impl ValueBuilder for MutableBooleanArray {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            &Value::Boolean(val) => self.push_value(val),
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected bool but got {value:?}").into(),
                ));
            }
        }
        Ok(())
    }
}

impl ValueBuilder for MutableBinaryViewArray<str> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            Value::String(val) => self.push_value(val),
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected string but got {value:?}").into(),
                ));
            }
        }
        Ok(())
    }
}

impl ValueBuilder for MutableBinaryViewArray<[u8]> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            Value::Bytes(val) | Value::Fixed(_, val) => self.push_value(val),
            Value::Uuid(val) => self.push_value(val.as_bytes()),
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected bytes but got {value:?}").into(),
                ));
            }
        }
        Ok(())
    }
}

impl ValueBuilder for MutablePrimitiveArray<i32> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            &Value::Int(val) | &Value::Date(val) => self.push_value(val),
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected int but got {value:?}").into(),
                ));
            }
        }
        Ok(())
    }
}

impl ValueBuilder for MutablePrimitiveArray<i64> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            &Value::Long(val)
            | &Value::LocalTimestampNanos(val)
            | &Value::LocalTimestampMicros(val)
            | &Value::LocalTimestampMillis(val)
            | &Value::TimestampNanos(val)
            | &Value::TimestampMicros(val)
            | &Value::TimestampMillis(val) => self.push_value(val),
            // NOTE arrow only supports time in nano, so we must scale up
            &Value::TimeMicros(val) => self.push_value(val * 1_000),
            &Value::TimeMillis(val) => self.push_value(i64::from(val) * 1_000_000),
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected long, timestamp, or time but got {value:?}").into(),
                ));
            }
        }
        Ok(())
    }
}

impl ValueBuilder for MutablePrimitiveArray<i128> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            Value::Decimal(val) => {
                let raw_bytes: Vec<_> = val.try_into().map_err(|err| {
                    PolarsError::ShapeMismatch(
                        format!("decimal can't be unwrapped into bytes: {err}").into(),
                    )
                })?;
                let len = raw_bytes.len();
                let filled: [u8; 16] = raw_bytes
                    .try_into()
                    .map_err(|_| PolarsError::ShapeMismatch(format!("polars only supports decimals with up to 16 bytes of precision, but got: {len}").into()))?;
                let parsed = i128::from_be_bytes(filled);
                self.push_value(parsed);
            }
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected long, timestamp, or time but got {value:?}").into(),
                ));
            }
        }
        Ok(())
    }
}

impl ValueBuilder for MutablePrimitiveArray<u32> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            &Value::Enum(val, _) => self.push_value(val),
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected enum but got {value:?}").into(),
                ));
            }
        }
        Ok(())
    }
}

impl ValueBuilder for MutablePrimitiveArray<f32> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            &Value::Float(val) => self.push_value(val),
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected float but got {value:?}").into(),
                ));
            }
        }
        Ok(())
    }
}

impl ValueBuilder for MutablePrimitiveArray<f64> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            &Value::Double(val) => self.push_value(val),
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected double but got {value:?}").into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ListBuilder {
    inner: MutableListArray<i64, Box<dyn ValueBuilder>>,
}

impl ListBuilder {
    pub fn with_capacity(field: &ArrowField, capacity: usize) -> Self {
        Self {
            inner: MutableListArray::new_from(
                new_value_builder(&field.dtype, capacity),
                ArrowDataType::LargeList(Box::new(field.clone())),
                capacity,
            ),
        }
    }
}

impl<'a> TryExtend<Option<&'a Value>> for Box<dyn ValueBuilder> {
    fn try_extend<I: IntoIterator<Item = Option<&'a Value>>>(
        &mut self,
        iter: I,
    ) -> PolarsResult<()> {
        for item in iter {
            self.try_push_value(item.unwrap())?;
        }
        Ok(())
    }
}

impl<'a> TryExtend<Option<(&'a String, &'a Value)>> for Box<dyn ValueBuilder> {
    fn try_extend<I: IntoIterator<Item = Option<(&'a String, &'a Value)>>>(
        &mut self,
        iter: I,
    ) -> PolarsResult<()> {
        for item in iter {
            let (key, val) = item.unwrap();
            self.try_push_keyed_value(key, val)?;
        }
        Ok(())
    }
}

impl MutableArray for ListBuilder {
    fn dtype(&self) -> &ArrowDataType {
        self.inner.dtype()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.inner.validity()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        self.inner.as_box()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn Any {
        self
    }

    fn push_null(&mut self) {
        self.inner.push_null();
    }

    fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }
}

impl ValueBuilder for ListBuilder {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => {
                self.push_null();
                Ok(())
            }
            Value::Array(vals) => self.inner.try_push(Some(vals.iter().map(Some))),
            Value::Map(vals) => self.inner.try_push(Some(vals.iter().map(Some))),
            _ => Err(PolarsError::SchemaMismatch(
                format!("expected array but got {value:?}").into(),
            )),
        }
    }
}

#[derive(Debug)]
pub struct StructBuilder {
    dtype: ArrowDataType,
    len: usize,
    values: Vec<Box<dyn ValueBuilder>>,
    validity: Option<MutableBitmap>,
}

impl StructBuilder {
    pub fn with_capacity(fields: &[ArrowField], capacity: usize) -> Self {
        let values = fields
            .iter()
            .map(|field| new_value_builder(&field.dtype, capacity))
            .collect();
        Self {
            dtype: ArrowDataType::Struct(Vec::from(fields)),
            len: 0,
            values,
            validity: None,
        }
    }
}

impl MutableArray for StructBuilder {
    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(StructArray::new(
            self.dtype.clone(),
            self.len,
            self.values.iter_mut().map(MutableArray::as_box).collect(),
            self.validity.as_ref().map(|bmp| bmp.clone().freeze()),
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn Any {
        self
    }

    fn push_null(&mut self) {
        match &mut self.validity {
            Some(val) => {
                val.push(false);
            }
            empty @ None => {
                let mut val = MutableBitmap::from_len_set(self.len);
                val.push(false);
                *empty = Some(val);
            }
        }
        for val in &mut self.values {
            val.push_null();
        }
        self.len += 1;
    }

    fn reserve(&mut self, additional: usize) {
        if let Some(val) = &mut self.validity {
            val.reserve(additional);
        }
        for val in &mut self.values {
            val.push_null();
        }
    }

    fn shrink_to_fit(&mut self) {
        if let Some(val) = &mut self.validity {
            val.shrink_to_fit();
        }
        for val in &mut self.values {
            val.shrink_to_fit();
        }
    }
}

impl ValueBuilder for StructBuilder {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            Value::Record(rec) => {
                if let Some(val) = &mut self.validity {
                    val.push(true);
                }
                for (arr, (_, val)) in self.values.iter_mut().zip(rec) {
                    arr.try_push_value(val)?;
                }
                self.len += 1;
            }
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected record but got {value:?}").into(),
                ));
            }
        }
        Ok(())
    }

    fn try_push_keyed_value(&mut self, key: &String, value: &Value) -> PolarsResult<()> {
        if let Some(val) = &mut self.validity {
            val.push(true);
        }
        let mut iter = self.values.iter_mut();

        // push key
        let key_array = iter.next().unwrap();
        let keys: &mut MutableBinaryViewArray<str> = key_array.as_mut_any().downcast_mut().unwrap();
        keys.push_value(key);

        // push val
        let val_array = iter.next().unwrap();
        val_array.try_push_value(value)?;

        debug_assert!(iter.next().is_none());

        self.len += 1;
        Ok(())
    }
}

pub fn new_value_builder(dtype: &ArrowDataType, capacity: usize) -> Box<dyn ValueBuilder> {
    match dtype {
        ArrowDataType::Boolean => Box::new(MutableBooleanArray::with_capacity(capacity)),
        ArrowDataType::Null => Box::new(MutableNullArray::new(ArrowDataType::Null, 0)),
        ArrowDataType::Int32 | ArrowDataType::Date32 => {
            Box::new(MutablePrimitiveArray::<i32>::with_capacity(capacity))
        }
        ArrowDataType::Int64 | ArrowDataType::Timestamp(_, _) | ArrowDataType::Time64(_) => {
            Box::new(MutablePrimitiveArray::<i64>::with_capacity(capacity))
        }
        ArrowDataType::Decimal(_, _) => {
            Box::new(MutablePrimitiveArray::<i128>::with_capacity(capacity))
        }
        ArrowDataType::Float32 => Box::new(MutablePrimitiveArray::<f32>::with_capacity(capacity)),
        ArrowDataType::Float64 => Box::new(MutablePrimitiveArray::<f64>::with_capacity(capacity)),
        ArrowDataType::Utf8View => Box::new(MutableBinaryViewArray::<str>::with_capacity(capacity)),
        ArrowDataType::BinaryView => {
            Box::new(MutableBinaryViewArray::<[u8]>::with_capacity(capacity))
        }
        // NOTE technically a dictionary array, but due to the series interface, we actually want to push raw u32
        ArrowDataType::Dictionary(_, _, _) => {
            Box::new(MutablePrimitiveArray::<u32>::with_capacity(capacity))
        }
        ArrowDataType::LargeList(field) => Box::new(ListBuilder::with_capacity(field, capacity)),
        ArrowDataType::Struct(fields) => Box::new(StructBuilder::with_capacity(fields, capacity)),
        ArrowDataType::Int8
        | ArrowDataType::Int16
        | ArrowDataType::Int128
        | ArrowDataType::UInt8
        | ArrowDataType::UInt16
        | ArrowDataType::UInt32
        | ArrowDataType::UInt64
        | ArrowDataType::Float16
        | ArrowDataType::Date64
        | ArrowDataType::Time32(_)
        | ArrowDataType::Utf8
        | ArrowDataType::LargeUtf8
        | ArrowDataType::Binary
        | ArrowDataType::LargeBinary
        | ArrowDataType::FixedSizeBinary(_)
        | ArrowDataType::FixedSizeList(_, _)
        | ArrowDataType::List(_)
        | ArrowDataType::Duration(_)
        | ArrowDataType::Interval(_)
        | ArrowDataType::Map(_, _)
        | ArrowDataType::Decimal256(_, _)
        | ArrowDataType::Extension(_)
        | ArrowDataType::Unknown
        | ArrowDataType::Union(_) => unreachable!("{dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use apache_avro::types::Value;
    use polars::prelude::{ArrowDataType, ArrowField};
    use polars_arrow::array::{
        MutableArray, MutableBinaryViewArray, MutableBooleanArray, MutableNullArray,
        MutablePrimitiveArray,
    };

    use super::{ListBuilder, StructBuilder, ValueBuilder};

    /// Test that Box of mutable array is a value builder
    #[test]
    fn test_box_dyn_value_builder() {
        let mut builder: Box<dyn ValueBuilder> = Box::new(MutableBooleanArray::with_capacity(4));

        assert_eq!(builder.dtype(), &ArrowDataType::Boolean);
        assert_eq!(builder.len(), 0);
        assert!(builder.validity().is_none());
        assert_eq!(builder.as_box().dtype(), &ArrowDataType::Boolean);
        builder
            .as_any()
            .downcast_ref::<MutableBooleanArray>()
            .unwrap();
        builder
            .as_mut_any()
            .downcast_mut::<MutableBooleanArray>()
            .unwrap();

        builder.push_null();
        assert_eq!(builder.len(), 1);

        builder.push_null();
        assert_eq!(builder.len(), 2);

        builder.reserve(3);
        builder.shrink_to_fit();
    }

    /// Test that list builder works
    #[test]
    fn test_list_builder() {
        let field = ArrowField::new("elem".into(), ArrowDataType::Boolean, true);
        let mut builder = ListBuilder::with_capacity(&field, 3);

        assert_eq!(
            builder.dtype(),
            &ArrowDataType::LargeList(Box::new(field.clone()))
        );
        assert_eq!(builder.len(), 0);
        assert!(builder.validity().is_none());
        assert_eq!(
            builder.as_box().dtype(),
            &ArrowDataType::LargeList(Box::new(field.clone()))
        );
        builder.as_any().downcast_ref::<ListBuilder>().unwrap();
        builder.as_mut_any().downcast_mut::<ListBuilder>().unwrap();

        builder.push_null();
        assert_eq!(builder.len(), 1);

        builder.reserve(3);
        builder.shrink_to_fit();
    }

    /// Test that struct builder works
    #[test]
    fn test_struct_builder() {
        let field = ArrowField::new("elem".into(), ArrowDataType::Boolean, true);
        let mut builder = StructBuilder::with_capacity(&[field.clone()], 3);

        assert_eq!(builder.dtype(), &ArrowDataType::Struct(vec![field.clone()]));
        assert_eq!(builder.len(), 0);
        assert!(builder.validity().is_none());
        assert_eq!(
            builder.as_box().dtype(),
            &ArrowDataType::Struct(vec![field.clone()])
        );
        builder.as_any().downcast_ref::<StructBuilder>().unwrap();
        builder
            .as_mut_any()
            .downcast_mut::<StructBuilder>()
            .unwrap();

        builder.push_null();
        assert_eq!(builder.len(), 1);

        builder.reserve(3);
        builder.shrink_to_fit();
    }

    #[test]
    fn test_incorrect_type_failures() {
        let mut builder = MutableNullArray::new(ArrowDataType::Null, 0);
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = MutableBooleanArray::new();
        assert!(builder.try_push_value(&Value::Int(0)).is_err());

        let mut builder = MutablePrimitiveArray::<i32>::new();
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = MutablePrimitiveArray::<i64>::new();
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = MutablePrimitiveArray::<f32>::new();
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = MutablePrimitiveArray::<f64>::new();
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = MutablePrimitiveArray::<u32>::new();
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = MutableBinaryViewArray::<[u8]>::new();
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = MutableBinaryViewArray::<str>::new();
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = ListBuilder::with_capacity(
            &ArrowField::new("elem".into(), ArrowDataType::Boolean, true),
            0,
        );
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = StructBuilder::with_capacity(
            &[ArrowField::new("elem".into(), ArrowDataType::Boolean, true)],
            0,
        );
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());
    }
}
