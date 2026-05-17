from pathlib import Path

import lance
import polars as pl
import pytest

from polars_lance import write_lance
from tests.utils import SUPPORTED_DATA_TYPES_ARROW_TABLE


def test_write_lance_data_types(tmp_path: Path) -> None:
    df = pl.DataFrame(SUPPORTED_DATA_TYPES_ARROW_TABLE)
    dataset_path = tmp_path / "test.lance"

    write_lance(df, target=dataset_path)

    ds = lance.dataset(dataset_path)
    assert pl.DataFrame(ds.to_table()).equals(df)


def test_write_lance_str_target(tmp_path: Path) -> None:
    df = pl.DataFrame(SUPPORTED_DATA_TYPES_ARROW_TABLE)
    dataset_path = tmp_path / "test.lance"

    write_lance(df, target=str(dataset_path))

    ds = lance.dataset(dataset_path)
    assert ds.count_rows() == SUPPORTED_DATA_TYPES_ARROW_TABLE.num_rows


def test_write_lance_empty_dataframe(tmp_path: Path) -> None:
    df = pl.DataFrame(schema={"id": pl.Int64, "val": pl.String})
    assert df.is_empty()
    dataset_path = tmp_path / "empty.lance"

    write_lance(df, target=dataset_path)

    ds = lance.dataset(dataset_path)
    assert ds.count_rows() == 0


def test_write_lance_error_mode(tmp_path: Path) -> None:
    first = pl.DataFrame({"id": [1], "val": ["a"]})
    second = pl.DataFrame({"id": [2], "val": ["b"]})
    dataset_path = tmp_path / "test.lance"
    write_lance(first, target=dataset_path)

    with pytest.raises(RuntimeError, match="already exists"):
        write_lance(second, target=dataset_path, mode="error")


def test_write_lance_append_mode(tmp_path: Path) -> None:
    first = pl.DataFrame({"id": [1], "val": ["a"]})
    second = pl.DataFrame({"id": [2], "val": ["b"]})
    dataset_path = tmp_path / "test.lance"
    write_lance(first, target=dataset_path)

    write_lance(second, target=dataset_path, mode="append")

    ds = lance.dataset(dataset_path)
    expected_df = pl.concat([first, second], how="vertical")
    assert pl.DataFrame(ds.to_table()).equals(expected_df)


def test_write_lance_overwrite_mode(tmp_path: Path) -> None:
    first = pl.DataFrame({"id": [1], "val": ["a"]})
    second = pl.DataFrame({"id": [2], "val": ["b"]})
    dataset_path = tmp_path / "test.lance"
    write_lance(first, target=dataset_path)

    write_lance(second, target=dataset_path, mode="overwrite")

    ds = lance.dataset(dataset_path)
    assert pl.DataFrame(ds.to_table()).equals(second)
