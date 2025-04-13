use std::collections::BTreeMap;

use super::Error;
use apache_avro::Schema as AvroSchema;
use apache_avro::schema::{
    DecimalSchema, EnumSchema, FixedSchema, Name, RecordField, RecordFieldOrder, RecordSchema,
    UnionSchema,
};
use apache_avro::types::Value;
use polars::prelude::{AnyValue, DataType, Field, RevMapping, Schema as PlSchema, TimeUnit};

pub fn try_as_schema(
    schema: &PlSchema,
    promote_ints: bool,
    promote_array: bool,
    truncate_time: bool,
) -> Result<AvroSchema, Error> {
    let ser = Serializer {
        promote_ints,
        promote_array,
        truncate_time,
    };
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
}

fn create_enum_schema(rev_mapping: &RevMapping) -> Result<EnumSchema, Error> {
    Ok(EnumSchema {
        name: "polars_avro_enum".into(),
        aliases: None,
        doc: None,
        symbols: rev_mapping
            .get_categories()
            .iter()
            .map(|val| val.map(str::to_string).ok_or(Error::NullEnum))
            .collect::<Result<_, _>>()?,
        default: None,
        attributes: BTreeMap::new(),
    })
}

fn create_decimal_schema(precision: usize, scale: usize) -> DecimalSchema {
    DecimalSchema {
        precision,
        scale,
        inner: Box::new(AvroSchema::Fixed(FixedSchema {
            name: "polars_avro_decimal".into(),
            aliases: None,
            doc: None,
            size: 16, // polars uses i128s so this is the max size
            default: None,
            attributes: BTreeMap::new(),
        })),
    }
}

impl Serializer {
    fn create_record_schema(&self, fields: &[Field]) -> Result<RecordSchema, Error> {
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
                name: "polars_avro_record".into(),
                namespace: None,
            },
            aliases: None,
            doc: None,
            fields,
            lookup,
            attributes: BTreeMap::new(),
        })
    }

    fn try_as_schema(&self, dtype: &DataType) -> Result<AvroSchema, Error> {
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
            &DataType::Decimal(Some(precision), Some(scale)) => {
                AvroSchema::Decimal(create_decimal_schema(precision, scale))
            }
            DataType::String => AvroSchema::String,
            DataType::Binary => AvroSchema::Bytes,
            DataType::Date => AvroSchema::Date,
            DataType::Datetime(TimeUnit::Milliseconds, Some(tz)) if tz == "UTC" => {
                AvroSchema::TimestampMillis
            }
            DataType::Datetime(TimeUnit::Microseconds, Some(tz)) if tz == "UTC" => {
                AvroSchema::TimestampMicros
            }
            DataType::Datetime(TimeUnit::Nanoseconds, Some(tz)) if tz == "UTC" => {
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
            DataType::Categorical(rev_mapping, _) | DataType::Enum(rev_mapping, _) => {
                if let Some(rev_mapping) = rev_mapping {
                    AvroSchema::Enum(create_enum_schema(rev_mapping)?)
                } else {
                    return Err(Error::NullEnum);
                }
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
            AnyValue::Decimal(val, _) => Value::Decimal(val.to_be_bytes().into()),
            &AnyValue::Float32(val) => Value::Float(val),
            &AnyValue::Float64(val) => Value::Double(val),
            &AnyValue::Date(val) => Value::Date(val),
            &AnyValue::Datetime(val, TimeUnit::Milliseconds, Some(tz)) if tz == "UTC" => {
                Value::TimestampMillis(val)
            }
            AnyValue::DatetimeOwned(val, TimeUnit::Milliseconds, Some(tz)) if **tz == "UTC" => {
                Value::TimestampMillis(*val)
            }
            &AnyValue::Datetime(val, TimeUnit::Microseconds, Some(tz)) if tz == "UTC" => {
                Value::TimestampMicros(val)
            }
            AnyValue::DatetimeOwned(val, TimeUnit::Microseconds, Some(tz)) if **tz == "UTC" => {
                Value::TimestampMicros(*val)
            }
            &AnyValue::Datetime(val, TimeUnit::Nanoseconds, Some(tz)) if tz == "UTC" => {
                Value::TimestampNanos(val)
            }
            AnyValue::DatetimeOwned(val, TimeUnit::Nanoseconds, Some(tz)) if **tz == "UTC" => {
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
            &AnyValue::Categorical(val, rev_mapping, _) | &AnyValue::Enum(val, rev_mapping, _) => {
                Value::Enum(val, rev_mapping.get(val).to_owned())
            }
            AnyValue::CategoricalOwned(val, rev_mapping, _)
            | AnyValue::EnumOwned(val, rev_mapping, _) => {
                Value::Enum(*val, rev_mapping.get(*val).to_owned())
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
