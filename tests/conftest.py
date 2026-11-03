import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings
from app.main import create_app

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")


def make_client(tmp_path, **kw):
    return TestClient(create_app(Settings(data_dir=str(tmp_path / "data"), **kw)))


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def upload_samples(c):
    files = [("files", (n, read_bytes(os.path.join(SAMPLES, n)))) for n in sorted(os.listdir(SAMPLES))]
    return c.post("/ingest", files=files)


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path)


@pytest.fixture
def loaded(tmp_path):
    c = make_client(tmp_path)
    assert upload_samples(c).status_code == 200
    return c
