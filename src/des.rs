//! Utilidies for deserializing from from avro.

use std::any::Any;

use super::Error;
use apache_avro::Schema as AvroSchema;
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
        Err(Error::NonRecordSchema(schema.clone()))
    }
}

fn try_from_dtype(schema: &AvroSchema) -> Result<DataType, Error> {
    match schema {
        AvroSchema::Null => Ok(DataType::Null),
        AvroSchema::Boolean => Ok(DataType::Boolean),
        AvroSchema::Int => Ok(DataType::Int32),
        AvroSchema::Long => Ok(DataType::Int64),
        AvroSchema::Float => Ok(DataType::Float32),
        AvroSchema::Double => Ok(DataType::Float64),
        AvroSchema::Bytes => Ok(DataType::Binary),
        AvroSchema::String => Ok(DataType::String),
        AvroSchema::Array(array_schema) => Ok(DataType::List(Box::new(try_from_dtype(
            &array_schema.items,
        )?))),
        AvroSchema::Record(record_schema) => Ok(DataType::Struct(
            record_schema
                .fields
                .iter()
                .map(|field| {
                    Ok(Field {
                        name: field.name.as_str().into(),
                        dtype: try_from_dtype(&field.schema)?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?,
        )),
        AvroSchema::Enum(enum_schema) => Ok(create_enum_dtype(Utf8ViewArray::from_slice_values(
            &enum_schema.symbols,
        ))),
        AvroSchema::Date => Ok(DataType::Date),
        AvroSchema::TimeMillis | AvroSchema::TimeMicros => Ok(DataType::Time),
        AvroSchema::TimestampMillis => Ok(DataType::Datetime(TimeUnit::Milliseconds, None)),
        AvroSchema::TimestampMicros => Ok(DataType::Datetime(TimeUnit::Microseconds, None)),
        AvroSchema::TimestampNanos => Ok(DataType::Datetime(TimeUnit::Nanoseconds, None)),
        AvroSchema::Union(union) => {
            let mut variants = union
                .variants()
                .iter()
                .filter(|var| !matches!(var, AvroSchema::Null));
            if union.variants().is_empty() {
                Err(Error::UnsupportedAvroType(schema.clone()))
            } else if let Some(rem) = variants.next() {
                if variants.next().is_some() {
                    // union with more than 1 non-null element
                    Err(Error::UnsupportedAvroType(schema.clone()))
                } else {
                    // else try again on non-union
                    try_from_dtype(rem)
                }
            } else {
                // must have removed null, so return null
                Ok(DataType::Null)
            }
        }
        AvroSchema::Map(_)
        | AvroSchema::Fixed(_)
        | AvroSchema::Decimal(_)
        | AvroSchema::BigDecimal
        | AvroSchema::Uuid
        | AvroSchema::LocalTimestampMillis
        | AvroSchema::LocalTimestampMicros
        | AvroSchema::LocalTimestampNanos
        | AvroSchema::Duration
        | AvroSchema::Ref { .. } => Err(Error::UnsupportedAvroType(schema.clone())),
    }
}

pub trait ValueBuilder: MutableArray {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()>;
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
        };
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
        };
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
        };
        Ok(())
    }
}

impl ValueBuilder for MutableBinaryViewArray<[u8]> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            Value::Bytes(val) => self.push_value(val),
            _ => {
                return Err(PolarsError::SchemaMismatch(
                    format!("expected bytes but got {value:?}").into(),
                ));
            }
        };
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
        };
        Ok(())
    }
}

impl ValueBuilder for MutablePrimitiveArray<i64> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            &Value::Long(val)
            // NOTE for these we preserve the unit so no conversion is necessary
            | &Value::TimestampNanos(val)
            | &Value::TimestampMicros(val)
            | &Value::TimestampMillis(val) => self.push_value(val),
            // NOTE arrow only supports time in nano, so we must scale up
            &Value::TimeMicros(val) => self.push_value(val * 1_000),
            &Value::TimeMillis(val) => self.push_value(i64::from(val) * 1_000_000),
            _ => return Err(PolarsError::SchemaMismatch(format!("expected long, timestamp, or time but got {value:?}").into())),
        };
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
        };
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
        };
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
        };
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
        };
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
        | ArrowDataType::Decimal(_, _)
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

        builder.reserve(3);
        builder.shrink_to_fit();
    }

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
