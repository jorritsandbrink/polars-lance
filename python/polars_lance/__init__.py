from collections.abc import Iterator
from pathlib import Path

import polars as pl
from polars.io.plugins import register_io_source

from polars_lance import _polars_lance

__all__ = ["scan_lance"]


def scan_lance(
    source: str | Path,
    *,
    storage_options: dict[str, str] | None = None,
) -> pl.LazyFrame:
    source_str = str(source)

    def io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        lance_scanner = _polars_lance.LanceScanner(
            uri=source_str,
            with_columns=with_columns,
            predicate=predicate,
            n_rows=n_rows,
            batch_size=batch_size,
            storage_options=storage_options,
        )

        while (df := lance_scanner.next()) is not None:
            yield df

    return register_io_source(
        io_source=io_source,
        schema=_polars_lance.LanceScanner.schema_for_uri(
            uri=source_str,
            storage_options=storage_options,
        ),
    )
