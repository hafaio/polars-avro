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
    Ok(ffi::arrow_schema_to_polars(reader.schema().as_ref()))
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

#[cfg(test)]
mod tests {
    use super::super::Error;
    use super::{IndProj, NameProj, Projection, get_schema};
    use apache_avro::schema::{RecordField, RecordSchema, Schema as ApacheSchema};
    use std::sync::Arc;

    #[test]
    fn test_get_schema_bad_header() {
        let err = get_schema(&b"this is not an avro file"[..]).unwrap_err();
        assert!(
            matches!(err, Error::Arrow(_) | Error::ArrowAvro(_)),
            "{err:?}"
        );
    }

    /// A three field record (`a: int`, `b: string`, `c: boolean`) to project against.
    fn record_schema() -> ApacheSchema {
        ApacheSchema::Record(
            RecordSchema::builder()
                .name("base".into())
                .fields(vec![
                    RecordField::builder()
                        .name("a".into())
                        .schema(ApacheSchema::Int)
                        .build(),
                    RecordField::builder()
                        .name("b".into())
                        .schema(ApacheSchema::String)
                        .build(),
                    RecordField::builder()
                        .name("c".into())
                        .schema(ApacheSchema::Boolean)
                        .build(),
                ])
                .lookup([("a".into(), 0), ("b".into(), 1), ("c".into(), 2)].into())
                .build(),
        )
    }

    /// Pull the ordered field names out of a projected record schema.
    fn field_names(schema: &ApacheSchema) -> Vec<String> {
        let ApacheSchema::Record(record) = schema else {
            panic!("expected a record schema");
        };
        record
            .fields
            .iter()
            .map(|field| field.name.clone())
            .collect()
    }

    #[test]
    fn test_name_proj_reorders_and_subsets() {
        let projected = NameProj(["c", "a"].as_slice())
            .project(&record_schema())
            .unwrap();
        assert_eq!(field_names(&projected), ["c", "a"]);
        let ApacheSchema::Record(record) = projected else {
            panic!("expected a record schema");
        };
        assert_eq!(record.lookup.get("c"), Some(&0));
        assert_eq!(record.lookup.get("a"), Some(&1));
    }

    #[test]
    fn test_name_proj_missing_column() {
        let err = NameProj(["nope"].as_slice())
            .project(&record_schema())
            .unwrap_err();
        assert!(matches!(err, Error::ColumnNotFound(col) if col == "nope"));
    }

    #[test]
    fn test_name_proj_non_record() {
        let err = NameProj(["a"].as_slice())
            .project(&ApacheSchema::String)
            .unwrap_err();
        assert!(matches!(err, Error::NonRecordSchema));
    }

    #[test]
    fn test_ind_proj_reorders_and_subsets() {
        let projected = IndProj([2usize, 0].as_slice())
            .project(&record_schema())
            .unwrap();
        assert_eq!(field_names(&projected), ["c", "a"]);
        let ApacheSchema::Record(record) = projected else {
            panic!("expected a record schema");
        };
        assert_eq!(record.lookup.get("c"), Some(&0));
        assert_eq!(record.lookup.get("a"), Some(&1));
    }

    #[test]
    fn test_ind_proj_out_of_bounds() {
        let err = IndProj([99usize].as_slice())
            .project(&record_schema())
            .unwrap_err();
        assert!(matches!(err, Error::ColumnIndexOutOfBounds(99)));
    }

    #[test]
    fn test_ind_proj_non_record() {
        let err = IndProj([0usize].as_slice())
            .project(&ApacheSchema::String)
            .unwrap_err();
        assert!(matches!(err, Error::NonRecordSchema));
    }

    /// The `Vec`, slice, `Arc` and `Box` impls all delegate to [`NameProj`].
    #[test]
    fn test_delegating_impls_match_name_proj() {
        let schema = record_schema();
        let expected = field_names(&NameProj(["b", "a"].as_slice()).project(&schema).unwrap());

        let owned: Vec<String> = vec!["b".into(), "a".into()];
        assert_eq!(field_names(&owned.project(&schema).unwrap()), expected);
        assert_eq!(
            field_names(&(&owned[..]).project(&schema).unwrap()),
            expected
        );

        let arc: Arc<[String]> = Arc::from(owned.clone());
        assert_eq!(field_names(&arc.project(&schema).unwrap()), expected);

        let boxed: Box<[String]> = owned.clone().into_boxed_slice();
        assert_eq!(field_names(&boxed.project(&schema).unwrap()), expected);
    }
}
