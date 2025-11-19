use std::collections::BTreeMap;

use super::Error;
use apache_avro::Schema as AvroSchema;
use apache_avro::schema::{
    DecimalSchema, EnumSchema, FixedSchema, Name, RecordField, RecordFieldOrder, RecordSchema,
    UnionSchema,
};
use apache_avro::types::Value;
use polars::prelude::{
    AnyValue, CategoricalMapping, DataType, Field, Schema as PlSchema, TimeUnit, TimeZone,
};

pub fn try_as_schema(
    schema: &PlSchema,
    promote_ints: bool,
    promote_array: bool,
    truncate_time: bool,
) -> Result<AvroSchema, Error> {
    let mut ser = Serializer::new(promote_ints, promote_array, truncate_time);
    let fields = schema
        .iter()
        .enumerate()
        .map(|(idx, (name, dtype))| {
            Ok(RecordField {
                name: name.as_str().into(),
                doc: None,
                aliases: None,
                default: None,
                schema: ser.try_as_schema(dtype)?,
                order: RecordFieldOrder::Ignore,
                position: idx,
                custom_attributes: BTreeMap::new(),
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let lookup: BTreeMap<_, _> = fields
        .iter()
        .enumerate()
        .map(|(idx, field)| (field.name.as_str().to_string(), idx))
        .collect();
    Ok(AvroSchema::Record(RecordSchema {
        name: Name {
            name: "polars_avro_schema".into(),
            namespace: None,
        },
        aliases: None,
        doc: None,
        fields,
        lookup,
        attributes: BTreeMap::new(),
    }))
}

struct Serializer {
    promote_ints: bool,
    promote_array: bool,
    truncate_time: bool,
    counter: usize,
}

impl Serializer {
    fn new(promote_ints: bool, promote_array: bool, truncate_time: bool) -> Self {
        Self {
            promote_ints,
            promote_array,
            truncate_time,
            counter: 0,
        }
    }

    fn inc(&mut self) -> usize {
        let res = self.counter;
        self.counter += 1;
        res
    }

    fn create_enum_schema(&mut self, mapping: &CategoricalMapping) -> Result<EnumSchema, Error> {
        let num_cats = mapping
            .num_cats_upper_bound()
            .try_into()
            .map_err(|_| Error::NullEnum)?;
        Ok(EnumSchema {
            name: Name {
                name: format!("polars_avro_enum_{}", self.inc()),
                namespace: None,
            },
            aliases: None,
            doc: None,
            symbols: (0..num_cats)
                .map(|cat| {
                    mapping
                        .cat_to_str(cat)
                        .map(str::to_string)
                        .ok_or(Error::NullEnum)
                })
                .collect::<Result<_, _>>()?,
            default: None,
            attributes: BTreeMap::new(),
        })
    }

    fn create_decimal_schema(&mut self, precision: usize, scale: usize) -> DecimalSchema {
        DecimalSchema {
            precision,
            scale,
            inner: Box::new(AvroSchema::Fixed(FixedSchema {
                name: Name {
                    name: format!("polars_avro_decimal_{}", self.inc()),
                    namespace: None,
                },
                aliases: None,
                doc: None,
                size: 16, // polars uses i128s so this is the max size
                default: None,
                attributes: BTreeMap::new(),
            })),
        }
    }

    fn create_record_schema(&mut self, fields: &[Field]) -> Result<RecordSchema, Error> {
        let fields = fields
            .iter()
            .enumerate()
            .map(|(idx, field)| {
                Ok(RecordField {
                    name: field.name.as_str().into(),
                    doc: None,
                    aliases: None,
                    default: None,
                    schema: self.try_as_schema(field.dtype())?,
                    order: RecordFieldOrder::Ignore,
                    position: idx,
                    custom_attributes: BTreeMap::new(),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let lookup: BTreeMap<_, _> = fields
            .iter()
            .enumerate()
            .map(|(idx, field)| (field.name.as_str().to_string(), idx))
            .collect();
        Ok(RecordSchema {
            name: Name {
                name: format!("polars_avro_record_{}", self.inc()),
                namespace: None,
            },
            aliases: None,
            doc: None,
            fields,
            lookup,
            attributes: BTreeMap::new(),
        })
    }

    fn try_as_schema(&mut self, dtype: &DataType) -> Result<AvroSchema, Error> {
        let base = match dtype {
            DataType::Null => return Ok(AvroSchema::Null),
            DataType::Boolean => AvroSchema::Boolean,
            DataType::Int8 | DataType::Int16 | DataType::UInt8 | DataType::UInt16
                if self.promote_ints =>
            {
                AvroSchema::Int
            }
            DataType::Int32 => AvroSchema::Int,
            DataType::UInt32 if self.promote_ints => AvroSchema::Long,
            DataType::Int64 => AvroSchema::Long,
            DataType::Float32 => AvroSchema::Float,
            DataType::Float64 => AvroSchema::Double,
            &DataType::Decimal(precision, scale) => {
                AvroSchema::Decimal(self.create_decimal_schema(precision, scale))
            }
            DataType::String => AvroSchema::String,
            DataType::Binary => AvroSchema::Bytes,
            DataType::Date => AvroSchema::Date,
            DataType::Datetime(TimeUnit::Milliseconds, Some(tz)) if tz == &TimeZone::UTC => {
                AvroSchema::TimestampMillis
            }
            DataType::Datetime(TimeUnit::Microseconds, Some(tz)) if tz == &TimeZone::UTC => {
                AvroSchema::TimestampMicros
            }
            DataType::Datetime(TimeUnit::Nanoseconds, Some(tz)) if tz == &TimeZone::UTC => {
                AvroSchema::TimestampNanos
            }
            DataType::Datetime(TimeUnit::Milliseconds, None) => AvroSchema::LocalTimestampMillis,
            DataType::Datetime(TimeUnit::Microseconds, None) => AvroSchema::LocalTimestampMicros,
            DataType::Datetime(TimeUnit::Nanoseconds, None) => AvroSchema::LocalTimestampNanos,
            DataType::Time if self.truncate_time => AvroSchema::TimeMicros,
            DataType::Array(elem_type, _) if self.promote_array => {
                AvroSchema::array(self.try_as_schema(elem_type)?)
            }
            DataType::List(elem_type) => AvroSchema::array(self.try_as_schema(elem_type)?),
            DataType::Categorical(_, mapping) | DataType::Enum(_, mapping) => {
                AvroSchema::Enum(self.create_enum_schema(mapping)?)
            }
            DataType::Struct(fields) => AvroSchema::Record(self.create_record_schema(fields)?),
            // wildcard to cover different features
            _ => return Err(Error::UnsupportedPolarsType(dtype.clone())),
        };
        Ok(AvroSchema::Union(UnionSchema::new(vec![
            AvroSchema::Null,
            base,
        ])?))
    }
}

// NOTE we first convert rows to AnyValue, then convert those to Value's then
// serialize those. We could be lazier and read values "natively" by downcasting
// Arrays, bur we'll still need to convert to Value's so the overhear of first
// converting to AnyValue is minimal.
pub fn try_as_value(schema: &PlSchema, record: &[AnyValue]) -> Result<Value, Error> {
    let mapped: Result<Vec<_>, Error> = schema
        .iter()
        .zip(record)
        .map(|((name, dtype), val)| Ok((name.as_str().to_string(), as_value(dtype, val))))
        .collect();
    Ok(Value::Record(mapped?))
}

#[allow(clippy::too_many_lines)]
fn as_value(dtype: &DataType, value: &AnyValue) -> Value {
    if let DataType::Null = dtype {
        // if datatype is null, only value is null
        Value::Null
    } else {
        let res = match value {
            // NOTE if datatype is not null, then null is the first element of nullable union
            AnyValue::Null => return Value::Union(0, Box::new(Value::Null)),
            &AnyValue::Boolean(val) => Value::Boolean(val),
            &AnyValue::String(val) => Value::String(val.to_owned()),
            AnyValue::StringOwned(val) => Value::String(val.to_string()),
            &AnyValue::Binary(items) => Value::Bytes(Vec::from(items)),
            AnyValue::BinaryOwned(items) => Value::Bytes(items.clone()),
            &AnyValue::UInt8(val) => Value::Int(i32::from(val)),
            &AnyValue::UInt16(val) => Value::Int(i32::from(val)),
            &AnyValue::UInt32(val) => Value::Long(i64::from(val)),
            &AnyValue::Int8(val) => Value::Int(i32::from(val)),
            &AnyValue::Int16(val) => Value::Int(i32::from(val)),
            &AnyValue::Int32(val) => Value::Int(val),
            &AnyValue::Int64(val) => Value::Long(val),
            AnyValue::Decimal(val, _, _) => Value::Decimal(val.to_be_bytes().into()),
            &AnyValue::Float32(val) => Value::Float(val),
            &AnyValue::Float64(val) => Value::Double(val),
            &AnyValue::Date(val) => Value::Date(val),
            &AnyValue::Datetime(val, TimeUnit::Milliseconds, Some(tz)) if tz == &TimeZone::UTC => {
                Value::TimestampMillis(val)
            }
            AnyValue::DatetimeOwned(val, TimeUnit::Milliseconds, Some(tz))
                if **tz == TimeZone::UTC =>
            {
                Value::TimestampMillis(*val)
            }
            &AnyValue::Datetime(val, TimeUnit::Microseconds, Some(tz)) if tz == &TimeZone::UTC => {
                Value::TimestampMicros(val)
            }
            AnyValue::DatetimeOwned(val, TimeUnit::Microseconds, Some(tz))
                if **tz == TimeZone::UTC =>
            {
                Value::TimestampMicros(*val)
            }
            &AnyValue::Datetime(val, TimeUnit::Nanoseconds, Some(tz)) if tz == &TimeZone::UTC => {
                Value::TimestampNanos(val)
            }
            AnyValue::DatetimeOwned(val, TimeUnit::Nanoseconds, Some(tz))
                if **tz == TimeZone::UTC =>
            {
                Value::TimestampNanos(*val)
            }
            &AnyValue::Datetime(val, TimeUnit::Milliseconds, None)
            | &AnyValue::DatetimeOwned(val, TimeUnit::Milliseconds, None) => {
                Value::LocalTimestampMillis(val)
            }
            &AnyValue::Datetime(val, TimeUnit::Microseconds, None)
            | &AnyValue::DatetimeOwned(val, TimeUnit::Microseconds, None) => {
                Value::LocalTimestampMicros(val)
            }
            &AnyValue::Datetime(val, TimeUnit::Nanoseconds, None)
            | &AnyValue::DatetimeOwned(val, TimeUnit::Nanoseconds, None) => {
                Value::LocalTimestampNanos(val)
            }
            &AnyValue::Time(val) => Value::TimeMicros((val + 500) / 1000),
            // NOTE these indixes should be safe, so potentially we could use
            // cat_to_str_unchecked, but I don't know enough about the safety to
            // enforce it.
            &AnyValue::Categorical(val, mapping) | &AnyValue::Enum(val, mapping) => Value::Enum(
                val,
                mapping
                    .cat_to_str(val)
                    .expect("invalid enum category")
                    .to_owned(),
            ),
            AnyValue::CategoricalOwned(val, mapping) | AnyValue::EnumOwned(val, mapping) => {
                Value::Enum(
                    *val,
                    mapping
                        .cat_to_str(*val)
                        .expect("invalid categorical category")
                        .to_owned(),
                )
            }
            AnyValue::List(series) | AnyValue::Array(series, _) => {
                if let DataType::List(etype) | DataType::Array(etype, _) = dtype {
                    Value::Array(series.iter().map(|item| as_value(etype, &item)).collect())
                } else {
                    unreachable!();
                }
            }
            &AnyValue::Struct(_, _, fields) => Value::Record(
                value
                    ._iter_struct_av()
                    .zip(fields)
                    .map(|(val, field)| {
                        (field.name().as_str().into(), as_value(field.dtype(), &val))
                    })
                    .collect(),
            ),
            AnyValue::StructOwned(struct_val) => {
                let (values, fields) = struct_val.as_ref();
                Value::Record(
                    values
                        .iter()
                        .zip(fields)
                        .map(|(val, field)| {
                            (field.name().as_str().into(), as_value(field.dtype(), val))
                        })
                        .collect(),
                )
            }
            _ => unreachable!("unhandled value: {value:?}"),
        };
        // wrap as second element of union, e.g. non-null
        Value::Union(1, Box::new(res))
    }
}
