from pathlib import Path

import lance
import polars as pl
import pyarrow as pa
import pytest
from testcontainers.azurite import AzuriteContainer
from testcontainers.minio import MinioContainer

from polars_lance import scan_lance
from tests.utils import SUPPORTED_DATA_TYPES_ARROW_TABLE, to_polars_arrow


def test_scan_lance_data_types(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(SUPPORTED_DATA_TYPES_ARROW_TABLE, dataset_path)

    df = scan_lance(dataset_path).collect()
    assert df.to_arrow() == to_polars_arrow(SUPPORTED_DATA_TYPES_ARROW_TABLE)


def test_scan_lance_str_source(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(SUPPORTED_DATA_TYPES_ARROW_TABLE, dataset_path)

    df = scan_lance(str(dataset_path)).collect()
    assert df.height == SUPPORTED_DATA_TYPES_ARROW_TABLE.num_rows


# This test fails if the `with_columns` arg in the `io_source` passed to
# `register_io_source` in `scan_lance` is not correctly applied in the Rust
# `LanceScanner` impl.
def test_scan_lance_project(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(SUPPORTED_DATA_TYPES_ARROW_TABLE, dataset_path)

    with_columns = ["int32", "string"]

    df = scan_lance(dataset_path).select(with_columns).collect()
    assert df.to_arrow() == to_polars_arrow(
        SUPPORTED_DATA_TYPES_ARROW_TABLE,
        with_columns=with_columns,
    )


# This test fails if the `predicate` arg in the `io_source` passed to
# `register_io_source` in `scan_lance` is not correctly applied in the Rust
# `LanceScanner` impl.
def test_scan_lance_filter(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(SUPPORTED_DATA_TYPES_ARROW_TABLE, dataset_path)

    predicate = pl.col("int32") > 0

    df = scan_lance(dataset_path).filter(predicate).collect()
    assert df.to_arrow() == to_polars_arrow(
        SUPPORTED_DATA_TYPES_ARROW_TABLE,
        predicate=predicate,
    )


# This test fails if the `n_rows` arg in the `io_source` passed to
# `register_io_source` in `scan_lance` is not correctly applied in the Rust
# `LanceScanner` impl.
def test_scan_lance_slice(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(SUPPORTED_DATA_TYPES_ARROW_TABLE, dataset_path)

    n_rows = 1
    assert n_rows < SUPPORTED_DATA_TYPES_ARROW_TABLE.num_rows  # sanity check

    df = scan_lance(dataset_path).head(n_rows).collect()
    assert df.to_arrow() == to_polars_arrow(
        SUPPORTED_DATA_TYPES_ARROW_TABLE,
        n_rows=n_rows,
    )


@pytest.mark.needs_docker
def test_scan_lance_s3_storage_options(minio: tuple[MinioContainer, str]) -> None:
    minio_container, bucket_name = minio
    table = pa.table({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    uri = f"s3://{bucket_name}/my_dataset.lance"
    config = minio_container.get_config()
    storage_options = {
        "aws_endpoint": f"http://{config['endpoint']}",
        "aws_access_key_id": config["access_key"],
        "aws_secret_access_key": config["secret_key"],
        # Including the region is not strictly necessary, but it makes requests
        # much faster.
        "aws_region": "us-east-1",
        "aws_allow_http": "true",
    }
    lance.write_dataset(table, uri, storage_options=storage_options)

    df = scan_lance(uri, storage_options=storage_options).collect()

    assert df.to_arrow() == to_polars_arrow(table)


@pytest.mark.needs_docker
def test_scan_lance_az_storage_options(azurite: tuple[AzuriteContainer, str]) -> None:
    azurite_container, container_name = azurite
    table = pa.table({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    uri = f"az://{container_name}/my_dataset.lance"
    conn_str = azurite_container.get_connection_string()
    endpoint = next(
        v
        for p in conn_str.split(";")
        if "=" in p
        for k, v in [p.split("=", 1)]
        if k == "BlobEndpoint"
    )
    storage_options = {
        "azure_endpoint": endpoint,
        "azure_storage_account_name": azurite_container.account_name,
        "azure_storage_account_key": azurite_container.account_key,
        "azure_allow_http": "true",
    }
    lance.write_dataset(table, uri, storage_options=storage_options)

    df = scan_lance(uri, storage_options=storage_options).collect()

    assert df.to_arrow() == to_polars_arrow(table)
