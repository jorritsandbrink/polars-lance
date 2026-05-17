from __future__ import annotations

from typing import Literal

import polars as pl

class LanceScanner:
    def __init__(
        self,
        uri: str,
        with_columns: list[str] | None = None,
        predicate: pl.Expr | None = None,
        n_rows: int | None = None,
        batch_size: int | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> None: ...
    @staticmethod
    def schema_for_uri(
        uri: str,
        storage_options: dict[str, str] | None = None,
    ) -> dict[str, pl.DataType]: ...
    def next(self) -> pl.DataFrame | None: ...

def write_lance(
    df: pl.DataFrame,
    target: str,
    *,
    mode: Literal["error", "append", "overwrite"] = "error",
) -> None: ...
