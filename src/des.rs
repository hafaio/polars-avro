//! Utilidies for deserializing from from avro.

use std::any::Any;
use std::collections::BTreeMap;

use super::Error;
use apache_avro::Schema as AvroSchema;
use apache_avro::schema::{
    ArraySchema, DecimalSchema, EnumSchema, FixedSchema, MapSchema, RecordField, RecordSchema,
};
use apache_avro::types::Value;
use polars::error::{PolarsError, PolarsResult};
use polars::prelude::{
    ArrowDataType, CompatLevel, DataType, Field, FrozenCategories, PlSmallStr, Schema as PlSchema,
    TimeUnit, TimeZone,
};
use polars_arrow::array::{
    Array, MutableArray, MutableBinaryViewArray, MutableBooleanArray, MutableDictionaryArray,
    MutableListArray, MutableNullArray, MutablePrimitiveArray, StructArray, TryExtend, TryPush,
};
use polars_arrow::bitmap::MutableBitmap;

/// Name of mapping key when converted to entries
const KEY_FIELD: &str = "key";
/// Name of mapping value when convered to entries
const VALUE_FIELD: &str = "value";

pub fn try_from_schema(
    schema: &AvroSchema,
    single_column_name: Option<&PlSmallStr>,
) -> Result<PlSchema, Error> {
    let dtype = DataTypeParser::try_from_dtype(schema)?;
    if let DataType::Struct(fields) = dtype {
        Ok(PlSchema::from_iter(fields))
    } else if let Some(col_name) = single_column_name {
        Ok(PlSchema::from_iter([(col_name.clone(), dtype)]))
    } else {
        Err(Error::NonRecordSchema(Box::new(schema.clone())))
    }
}

#[derive(Debug, Default)]
struct DataTypeParser {
    names: BTreeMap<String, DataType>,
}

impl DataTypeParser {
    fn try_from_dtype(schema: &AvroSchema) -> Result<DataType, Error> {
        let mut parser = DataTypeParser::default();
        parser.try_from_dtype_inner(schema)
    }

    fn try_from_dtype_inner(&mut self, schema: &AvroSchema) -> Result<DataType, Error> {
        Ok(match schema {
            AvroSchema::Null => DataType::Null,
            AvroSchema::Boolean => DataType::Boolean,
            AvroSchema::Int => DataType::Int32,
            AvroSchema::Long => DataType::Int64,
            AvroSchema::Float => DataType::Float32,
            AvroSchema::Double => DataType::Float64,
            AvroSchema::Bytes | AvroSchema::Uuid => DataType::Binary,
            AvroSchema::String => DataType::String,
            AvroSchema::Array(ArraySchema { items, .. }) => {
                DataType::List(Box::new(self.try_from_dtype_inner(items)?))
            }
            AvroSchema::Record(RecordSchema { fields, name, .. }) => {
                let fullname = name.fullname(None);
                let dtype = DataType::Struct(
                    fields
                        .iter()
                        .map(|RecordField { name, schema, .. }| {
                            Ok(Field {
                                name: name.as_str().into(),
                                dtype: self.try_from_dtype_inner(schema)?,
                            })
                        })
                        .collect::<Result<Vec<_>, Error>>()?,
                );
                let old = self.names.insert(fullname, dtype.clone());
                debug_assert!(old.is_none(), "found duplicate name {name:?}");
                dtype
            }
            AvroSchema::Enum(EnumSchema { symbols, name, .. }) => {
                let fullname = name.fullname(None);
                let categories = FrozenCategories::new(symbols.iter().map(String::as_ref))?;
                let dtype = DataType::from_frozen_categories(categories);
                let old = self.names.insert(fullname, dtype.clone());
                debug_assert!(old.is_none(), "found duplicate name {name:?}");
                dtype
            }
            AvroSchema::Fixed(FixedSchema { name, .. }) => {
                let fullname = name.fullname(None);
                let dtype = DataType::Binary;
                let old = self.names.insert(fullname, dtype.clone());
                debug_assert!(old.is_none(), "found duplicate name {name:?}");
                dtype
            }
            AvroSchema::Date => DataType::Date,
            AvroSchema::TimeMillis | AvroSchema::TimeMicros => DataType::Time,
            AvroSchema::TimestampMillis => {
                DataType::Datetime(TimeUnit::Milliseconds, Some(TimeZone::UTC))
            }
            AvroSchema::TimestampMicros => {
                DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC))
            }
            AvroSchema::TimestampNanos => {
                DataType::Datetime(TimeUnit::Nanoseconds, Some(TimeZone::UTC))
            }
            AvroSchema::LocalTimestampMillis => DataType::Datetime(TimeUnit::Milliseconds, None),
            AvroSchema::LocalTimestampMicros => DataType::Datetime(TimeUnit::Microseconds, None),
            AvroSchema::LocalTimestampNanos => DataType::Datetime(TimeUnit::Nanoseconds, None),
            AvroSchema::Union(union) => match union.variants() {
                [AvroSchema::Null, other] | [other, AvroSchema::Null] | [other] => {
                    self.try_from_dtype_inner(other)?
                }
                _ => return Err(Error::UnsupportedAvroType(Box::new(schema.clone()))),
            },
            AvroSchema::Map(MapSchema { types, .. }) => {
                DataType::List(Box::new(DataType::Struct(vec![
                    Field::new(KEY_FIELD.into(), DataType::String),
                    Field::new(VALUE_FIELD.into(), self.try_from_dtype_inner(types)?),
                ])))
            }
            &AvroSchema::Decimal(DecimalSchema {
                precision, scale, ..
            }) => DataType::Decimal(precision, scale),
            AvroSchema::Ref { name } => {
                let resolved = name.fullname(None);
                if let Some(referenced) = self.names.get(&resolved) {
                    referenced.clone()
                } else {
                    return Err(Error::MissingRefName(resolved));
                }
            }
            AvroSchema::BigDecimal | AvroSchema::Duration => {
                return Err(Error::UnsupportedAvroType(Box::new(schema.clone())));
            }
        })
    }
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

impl ValueBuilder for MutableDictionaryArray<u8, MutableBinaryViewArray<str>> {
    fn try_push_value(&mut self, value: &Value) -> PolarsResult<()> {
        match unwrap_union(value) {
            Value::Null => self.push_null(),
            Value::Enum(_, val) => self.try_push(Some(val.as_ref()))?,
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
    pub fn with_capacity(dtype: &DataType, capacity: usize) -> Self {
        Self {
            inner: MutableListArray::new_from(
                new_value_builder(dtype, capacity),
                ArrowDataType::LargeList(Box::new(
                    dtype.to_arrow_field("item".into(), CompatLevel::newest()),
                )),
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
    pub fn with_capacity(fields: &[Field], capacity: usize) -> Self {
        let values = fields
            .iter()
            .map(|field| new_value_builder(&field.dtype, capacity))
            .collect();
        Self {
            dtype: ArrowDataType::Struct(
                fields
                    .iter()
                    .map(|field| {
                        field
                            .dtype
                            .to_arrow_field(field.name.clone(), CompatLevel::newest())
                    })
                    .collect(),
            ),
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

pub fn new_value_builder(dtype: &DataType, capacity: usize) -> Box<dyn ValueBuilder> {
    match dtype {
        DataType::Boolean => Box::new(MutableBooleanArray::with_capacity(capacity)),
        DataType::Null => Box::new(MutableNullArray::new(ArrowDataType::Null, 0)),
        DataType::Int32 => Box::new(MutablePrimitiveArray::<i32>::with_capacity(capacity)),
        DataType::Date => Box::new(
            MutablePrimitiveArray::<i32>::try_new(
                ArrowDataType::Date32,
                Vec::with_capacity(capacity),
                None,
            )
            .unwrap(),
        ),
        DataType::Int64 => Box::new(MutablePrimitiveArray::<i64>::with_capacity(capacity)),
        DataType::Datetime(_, _) | DataType::Time => Box::new(
            MutablePrimitiveArray::<i64>::try_new(
                dtype.to_arrow(CompatLevel::newest()),
                Vec::with_capacity(capacity),
                None,
            )
            .unwrap(),
        ),
        &DataType::Decimal(precision, scale) => Box::new(
            MutablePrimitiveArray::<i128>::try_new(
                ArrowDataType::Decimal(precision, scale),
                Vec::with_capacity(capacity),
                None,
            )
            .unwrap(),
        ),
        DataType::Float32 => Box::new(MutablePrimitiveArray::<f32>::with_capacity(capacity)),
        DataType::Float64 => Box::new(MutablePrimitiveArray::<f64>::with_capacity(capacity)),
        DataType::String => Box::new(MutableBinaryViewArray::<str>::with_capacity(capacity)),
        DataType::Binary => Box::new(MutableBinaryViewArray::<[u8]>::with_capacity(capacity)),
        DataType::Enum(_, mapping) | DataType::Categorical(_, mapping) => {
            let num = mapping.num_cats_upper_bound();
            let mut vals = MutableBinaryViewArray::<str>::with_capacity(num);
            for cat in 0..u32::try_from(num).expect("enum values didn't fit in u32") {
                let cat_str = mapping.cat_to_str(cat).unwrap_or_default();
                vals.push_value_ignore_validity(cat_str);
            }
            // NOTE fails if there are more than u32 values, or they're non-unique, but these are guaranteed by avro spec
            let mut array =
                MutableDictionaryArray::<u8, MutableBinaryViewArray<str>>::from_values(vals)
                    .unwrap();
            array.reserve(capacity);
            Box::new(array)
        }
        DataType::List(dtype) => Box::new(ListBuilder::with_capacity(dtype, capacity)),
        DataType::Struct(fields) => Box::new(StructBuilder::with_capacity(fields, capacity)),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use apache_avro::types::Value;
    use polars::prelude::{ArrowDataType, ArrowField, DataType, Field};
    use polars_arrow::array::{
        MutableArray, MutableBinaryViewArray, MutableBooleanArray, MutableDictionaryArray,
        MutableNullArray, MutablePrimitiveArray,
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
        let mut builder = ListBuilder::with_capacity(&DataType::Boolean, 3);
        let field = ArrowField::new("item".into(), ArrowDataType::Boolean, true);

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
        let mut builder =
            StructBuilder::with_capacity(&[Field::new("elem".into(), DataType::Boolean)], 3);

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

        let mut builder = MutableDictionaryArray::<u8, MutableBinaryViewArray<str>>::new();
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = MutableBinaryViewArray::<[u8]>::new();
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = MutableBinaryViewArray::<str>::new();
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder = ListBuilder::with_capacity(&DataType::Boolean, 0);
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());

        let mut builder =
            StructBuilder::with_capacity(&[Field::new("elem".into(), DataType::Boolean)], 0);
        assert!(builder.try_push_value(&Value::Boolean(true)).is_err());
    }
}
