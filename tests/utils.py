from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import polars as pl
import pyarrow as pa

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
