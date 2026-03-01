use super::Error;
use super::ffi;
use apache_avro::schema::{RecordSchema, Schema as ApacheSchema};
use arrow_avro::reader::ReaderBuilder;
use polars::prelude::Schema as PlSchema;
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::io::BufRead;
use std::ops::Deref;
use std::sync::Arc;

/// Get a polars schema from an avro reader
///
/// # Errors
/// If the avro schema can't be converted into a polars schema, or any errors
/// from the reader
pub fn get_schema<R: BufRead>(reader: R) -> Result<PlSchema, Error> {
    let reader = ReaderBuilder::new().build(reader)?;
    ffi::arrow_schema_to_polars(reader.schema().as_ref())
}

/// Something that can be used to project a schema
pub trait Projection {
    /// Project an avro schema
    ///
    /// # Errors
    /// If there's a problem implementing the projection
    fn project(&self, schema: &ApacheSchema) -> Result<ApacheSchema, Error>;
}

impl Projection for Infallible {
    fn project(&self, _: &ApacheSchema) -> Result<ApacheSchema, Error> {
        match *self {}
    }
}

impl<S: AsRef<str>> Projection for Vec<S> {
    fn project(&self, schema: &ApacheSchema) -> Result<ApacheSchema, Error> {
        NameProj(&**self).project(schema)
    }
}

impl<S: AsRef<str>> Projection for &[S] {
    fn project(&self, schema: &ApacheSchema) -> Result<ApacheSchema, Error> {
        NameProj(&**self).project(schema)
    }
}

impl<S: AsRef<str>> Projection for Arc<[S]> {
    fn project(&self, schema: &ApacheSchema) -> Result<ApacheSchema, Error> {
        NameProj(&**self).project(schema)
    }
}

impl<S: AsRef<str>> Projection for Box<[S]> {
    fn project(&self, schema: &ApacheSchema) -> Result<ApacheSchema, Error> {
        NameProj(&**self).project(schema)
    }
}

/// A projection of column names
pub struct NameProj<A>(pub A);

impl<S: AsRef<str>, A: Deref<Target = [S]>> Projection for NameProj<A> {
    fn project(&self, schema: &ApacheSchema) -> Result<ApacheSchema, Error> {
        let RecordSchema {
            name,
            aliases,
            doc,
            fields,
            lookup,
            attributes,
        } = match schema {
            ApacheSchema::Record(rec) => Ok(rec),
            _ => Err(Error::NonRecordSchema),
        }?;

        let mut new_fields = Vec::new();
        let mut new_lookup = BTreeMap::new();
        for (new_ind, name) in self.0.iter().enumerate() {
            let &old_ind = lookup
                .get(name.as_ref())
                .ok_or_else(|| Error::ColumnNotFound(name.as_ref().into()))?;
            new_fields.push(fields[old_ind].clone());
            new_lookup.insert(name.as_ref().into(), new_ind);
        }

        Ok(ApacheSchema::Record(RecordSchema {
            name: name.clone(),
            aliases: aliases.clone(),
            doc: doc.clone(),
            fields: new_fields,
            lookup: new_lookup,
            attributes: attributes.clone(),
        }))
    }
}

/// A projection of column indices
pub struct IndProj<A>(pub A);

impl<A: Deref<Target = [usize]>> Projection for IndProj<A> {
    fn project(&self, schema: &ApacheSchema) -> Result<ApacheSchema, Error> {
        let RecordSchema {
            name,
            aliases,
            doc,
            fields,
            attributes,
            ..
        } = match schema {
            ApacheSchema::Record(rec) => Ok(rec),
            _ => Err(Error::NonRecordSchema),
        }?;

        let mut new_fields = Vec::new();
        let mut new_lookup = BTreeMap::new();
        for (new_ind, old_ind) in self.0.iter().enumerate() {
            let new_field = fields
                .get(*old_ind)
                .ok_or(Error::ColumnIndexOutOfBounds(*old_ind))?
                .clone();
            new_lookup.insert(new_field.name.clone(), new_ind);
            new_fields.push(new_field);
        }

        Ok(ApacheSchema::Record(RecordSchema {
            name: name.clone(),
            aliases: aliases.clone(),
            doc: doc.clone(),
            fields: new_fields,
            lookup: new_lookup,
            attributes: attributes.clone(),
        }))
    }
}
