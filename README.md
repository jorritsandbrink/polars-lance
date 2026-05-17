# polars-lance

Polars plugin for reading Lance datasets into Polars dataframes and writing Polars dataframes to Lance datasets.

## Installation

```bash
pip install polars-lance
```

## Read

```python
from polars_lance import scan_lance

lf = scan_lance("data/example.lance")
df = lf.collect()
```

## Write

```python
import polars as pl

from polars_lance import write_lance

df = pl.DataFrame({"id": [1, 2], "val": ["a", "b"]})
write_lance(df, "data/example.lance")
```

Pass `mode` to control behavior in case the target dataset already exists. Valid values are `error` (default), `append`, and `overwrite`.

```python
write_lance(df, "data/example.lance", mode="append")
```

## Cloud storage

Pass `storage_options` to access Lance datasets stored in AWS S3, Azure Blob Storage, or Google Cloud Storage.

```python
from polars_lance import scan_lance

lf = scan_lance(
    "s3://my-bucket/example.lance",
    storage_options={
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "aws_region": "us-east-1",
    },
)
df = lf.collect()
```

Cloud storage IO is powered by the `object_store` Rust crate. These are the supported schemes and options:
- `s3://`: [AWS options](https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html)
- `az://`: [Azure options](https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html)
- `gs://`: [GCP options](https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html)
