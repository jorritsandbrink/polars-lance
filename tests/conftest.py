from collections.abc import Iterator

import pytest
from azure.storage.blob import BlobServiceClient
from testcontainers.azurite import AzuriteContainer
from testcontainers.minio import MinioContainer


@pytest.fixture
def minio() -> Iterator[tuple[MinioContainer, str]]:
    bucket_name = "my-bucket"
    with MinioContainer() as minio_container:
        minio_container.get_client().make_bucket(bucket_name)
        yield minio_container, bucket_name


@pytest.fixture
def azurite() -> Iterator[tuple[AzuriteContainer, str]]:
    container_name = "my-container"
    with AzuriteContainer() as azurite_container:
        conn_str = azurite_container.get_connection_string()
        BlobServiceClient.from_connection_string(
            conn_str, api_version="2025-11-05"
        ).create_container(container_name)
        yield azurite_container, container_name
