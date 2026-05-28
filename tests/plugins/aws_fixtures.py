"""Pytest fixtures for mocking AWS S3 (moto). Never contact real AWS in tests."""

import random
import string
from pathlib import Path

import boto3
import pytest
from cloudpathlib import S3Client, S3Path
from moto import mock_aws


@pytest.fixture(scope="session", autouse=True)
def mock_aws_credentials(monkeypatch_session):
    """Mocked AWS credentials for moto."""
    monkeypatch_session.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch_session.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch_session.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch_session.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch_session.delenv("AWS_REGION", raising=False)
    monkeypatch_session.delenv("AWS_DEFAULT_REGION", raising=False)


@pytest.fixture(scope="session", autouse=True)
def set_up_cloudpathlib_s3client(mock_aws_credentials, monkeypatch_session):
    """Default cloudpathlib S3 client uses moto for the whole test session."""
    monkeypatch_session.setenv("CLOUDPATHLIB_FILE_CACHE_MODE", "close_file")
    with mock_aws():
        client = S3Client()
        client.set_as_default_client()


@pytest.fixture
def mock_s3_context(mock_aws_credentials):
    """S3 resource inside a fresh moto context."""
    with mock_aws():
        session = boto3.Session()
        yield session.resource("s3", region_name="us-east-1")


@pytest.fixture
def create_mock_bucket(mock_s3_context):
    """Create a mock S3 bucket (optional fixed name)."""
    s3 = mock_s3_context
    local_random = random.Random()

    def _create_bucket(bucket_name: str | None = None):
        if not bucket_name:
            bucket_name = "".join(local_random.choice(string.ascii_letters) for _ in range(16))
        bucket = s3.Bucket(bucket_name)
        if not bucket.creation_date:
            bucket.create()
        return bucket

    return _create_bucket


@pytest.fixture
def write_file_to_s3(mock_s3_context, create_mock_bucket):
    """Upload a local file to mocked S3."""

    def _write(filepath: Path, uri: str | S3Path, exists_ok: bool = False) -> S3Path:
        content = filepath.read_bytes()
        s3_path = S3Path(uri)
        create_mock_bucket(s3_path.bucket)
        if not exists_ok and s3_path.exists():
            raise ValueError(f"Object {uri} already exists in mock bucket.")
        s3_path.mkdir(parents=True)
        s3_path.write_bytes(content)
        return s3_path

    return _write
