//! Simple abstractions around polars io
//!
use std::{io::Cursor, sync::Arc};

use super::Error;
use polars_io::{HiveOptions, cloud::CloudOptions, file_cache::FileCacheEntry};
use polars_plan::{plans::ScanSources, prelude::FileScanOptions};
use polars_utils::mmap::MemSlice;

pub struct SourceIter {
    sources: ScanSources,
    are_cloud_urls: bool,
    cache_entries: Option<Vec<Arc<FileCacheEntry>>>,
    idx: usize,
}

impl SourceIter {
    pub fn try_from(
        sources: &ScanSources,
        cloud_options: Option<&CloudOptions>,
        glob: bool,
    ) -> Result<Self, Error> {
        // expand paths
        let scan_options = FileScanOptions {
            slice: None,
            with_columns: None,
            cache: true,
            row_index: None,
            rechunk: false,
            file_counter: 0,
            hive_options: HiveOptions {
                enabled: None,
                hive_start_idx: 0,
                schema: None,
                try_parse_dates: false,
            },
            glob,
            include_file_paths: None,
            allow_missing_columns: false,
        };
        let sources = sources.expand_paths(&scan_options, cloud_options)?;

        // cache cloud files
        let are_cloud_urls = sources.is_cloud_url();
        let cache_entries = {
            if are_cloud_urls {
                Some(polars_io::file_cache::init_entries_from_uri_list(
                    sources
                        .as_paths()
                        .unwrap()
                        .iter()
                        .map(|path| Arc::from(path.to_str().unwrap()))
                        .collect::<Vec<_>>()
                        .as_slice(),
                    cloud_options,
                )?)
            } else {
                None
            }
        };

        // initialize iterator
        Ok(Self {
            sources,
            are_cloud_urls,
            cache_entries,
            idx: 0,
        })
    }
}

impl Iterator for SourceIter {
    type Item = Result<Cursor<MemSlice>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let source = self.sources.get(self.idx);
        self.idx += 1;
        source.map(|source| {
            let memslice = source.to_memslice_possibly_async(
                self.are_cloud_urls,
                self.cache_entries.as_ref(),
                0,
            )?;
            Ok(Cursor::new(memslice))
        })
    }
}
