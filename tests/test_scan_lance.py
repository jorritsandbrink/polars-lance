from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import lance
import polars as pl
import pyarrow as pa
import pytest
from testcontainers.azurite import AzuriteContainer
from testcontainers.minio import MinioContainer

from polars_lance import scan_lance

# we exclude some types because they are currently not supported by Lance and/or Polars
# all data types: https://arrow.apache.org/docs/python/api/datatypes.html
# excluded types: month_day_nano_interval, binary_view, string_view, decimal256,
#   list_view, large_list_view, map_, run_end_encoded, fixed_shape_tensor, union,
#   dense_union, sparse_union, opaque, bool8, uuid, json_
SUPPORTED_DATA_TYPES_COLUMNS = [
    ("null", pa.null(), [None, None]),
    ("bool", pa.bool_(), [True, False]),
    ("int8", pa.int8(), [1, -2]),
    ("int16", pa.int16(), [1, -2]),
    ("int32", pa.int32(), [1, -2]),
    ("int64", pa.int64(), [1, -2]),
    ("uint8", pa.uint8(), [1, 2]),
    ("uint16", pa.uint16(), [1, 2]),
    ("uint32", pa.uint32(), [1, 2]),
    ("uint64", pa.uint64(), [1, 2]),
    ("float16", pa.float16(), [1.5, -2.5]),
    ("float32", pa.float32(), [1.5, -2.5]),
    ("float64", pa.float64(), [1.5, -2.5]),
    ("date32", pa.date32(), [date(2024, 1, 1), date(2024, 1, 2)]),
    ("date64", pa.date64(), [date(2024, 1, 1), date(2024, 1, 2)]),
    ("time32_s", pa.time32("s"), [time(1, 2, 3), time(4, 5, 6)]),
    ("time32_ms", pa.time32("ms"), [time(1, 2, 3), time(4, 5, 6)]),
    ("time64_us", pa.time64("us"), [time(1, 2, 3), time(4, 5, 6)]),
    ("time64_ns", pa.time64("ns"), [time(1, 2, 3), time(4, 5, 6)]),
    (
        "timestamp_s",
        pa.timestamp("s"),
        [datetime(2024, 1, 1, 1, 2, 3), datetime(2024, 1, 2, 4, 5, 6)],
    ),
    (
        "timestamp_ms",
        pa.timestamp("ms"),
        [datetime(2024, 1, 1, 1, 2, 3), datetime(2024, 1, 2, 4, 5, 6)],
    ),
    (
        "timestamp_us",
        pa.timestamp("us"),
        [datetime(2024, 1, 1, 1, 2, 3), datetime(2024, 1, 2, 4, 5, 6)],
    ),
    (
        "timestamp_ns",
        pa.timestamp("ns"),
        [datetime(2024, 1, 1, 1, 2, 3), datetime(2024, 1, 2, 4, 5, 6)],
    ),
    (
        "timestamp_tz",
        pa.timestamp("us", tz="UTC"),
        [
            datetime(2024, 1, 1, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2024, 1, 2, 4, 5, 6, tzinfo=timezone.utc),
        ],
    ),
    ("duration_s", pa.duration("s"), [timedelta(seconds=1), timedelta(seconds=2)]),
    (
        "duration_ms",
        pa.duration("ms"),
        [timedelta(seconds=1), timedelta(seconds=2)],
    ),
    (
        "duration_us",
        pa.duration("us"),
        [timedelta(seconds=1), timedelta(seconds=2)],
    ),
    (
        "duration_ns",
        pa.duration("ns"),
        [timedelta(seconds=1), timedelta(seconds=2)],
    ),
    ("binary_variable_size", pa.binary(), [b"a", b"bb"]),
    ("binary_fixed_size", pa.binary(3), [b"aaa", b"bbb"]),
    ("string", pa.string(), ["a", "bb"]),
    ("utf8", pa.utf8(), ["a", "bb"]),
    ("large_binary", pa.large_binary(), [b"a", b"bb"]),
    ("large_string", pa.large_string(), ["a", "bb"]),
    ("large_utf8", pa.large_utf8(), ["a", "bb"]),
    ("decimal128", pa.decimal128(10, 2), [Decimal("1.23"), Decimal("4.56")]),
    ("list_variable_size_int32", pa.list_(pa.int32()), [[1, 2], [3]]),
    ("list_fixed_size_int32", pa.list_(pa.int32(), 2), [[1, 2], [3, 4]]),
    ("large_list_int32", pa.large_list(pa.int32()), [[1, 2], [3]]),
    (
        "struct",
        pa.struct(
            [
                ("x", pa.int32()),
                (
                    "nested",
                    pa.struct(
                        [
                            ("flag", pa.bool_()),
                            ("values", pa.list_(pa.int32())),
                        ]
                    ),
                ),
            ]
        ),
        [
            {"x": 1, "nested": {"flag": True, "values": [1, 2]}},
            {"x": 2, "nested": {"flag": False, "values": [3]}},
        ],
    ),
    ("dictionary", pa.dictionary(pa.int8(), pa.string()), ["a", "b"]),
]
SUPPORTED_DATA_TYPES_ARROW_TABLE = pa.table(
    {
        name: pa.array(values, type=dtype)
        for name, dtype, values in SUPPORTED_DATA_TYPES_COLUMNS
    }
)


def to_polars_arrow(
    table: pa.Table,
    *,
    with_columns: list[str] | None = None,
    predicate: pl.Expr | None = None,
    n_rows: int | None = None,
) -> pa.Table:
    expected_df = pl.from_arrow(table)
    assert isinstance(expected_df, pl.DataFrame)

    lf = expected_df.lazy()

    if with_columns is not None:
        lf = lf.select(with_columns)

    if predicate is not None:
        lf = lf.filter(predicate)

    if n_rows is not None:
        lf = lf.head(n_rows)

    return lf.collect().to_arrow()


def test_scan_lance_roundtrip(tmp_path: Path) -> None:
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
