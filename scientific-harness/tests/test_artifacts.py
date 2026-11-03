"""Check complete-file writes, overwrite rejection, fingerprints, and rejected Python objects."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from controller.artifacts import (
    checksum,
    read_arrays,
    read_json,
    write_arrays,
    write_bytes,
    write_json,
)


def test_artifact_round_trip_and_overwrite_rejection(tmp_path: Path) -> None:
    path = tmp_path / "arrays.npz"
    values = np.arange(6.0).reshape(2, 3)
    digest = write_arrays(path, values=values, member_ids=np.array(["a", "b", "c"]))
    loaded = read_arrays(path, digest)
    np.testing.assert_array_equal(loaded["values"], values)
    assert loaded["values"].dtype == np.float64
    assert loaded["member_ids"].tolist() == ["a", "b", "c"]
    with pytest.raises(FileExistsError):
        write_arrays(path, values=np.zeros(1))
    assert checksum(path) == digest
    write_json(tmp_path / "manifest.json", {"artifact_sha256": digest})
    assert read_json(tmp_path / "manifest.json")["artifact_sha256"] == digest


def test_changed_or_pickle_based_artifacts_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "arrays.npz"
    digest = write_arrays(path, values=np.ones(3))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_arrays(path, digest)
    with pytest.raises(ValueError, match="object arrays"):
        write_arrays(tmp_path / "objects.npz", values=np.array([{}], dtype=object))
    assert not (tmp_path / "objects.npz").exists()


def test_only_one_concurrent_writer_publishes_a_name(tmp_path: Path) -> None:
    path = tmp_path / "record.bin"

    def publish(value: bytes) -> bool:
        try:
            write_bytes(path, value)
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, [b"first", b"second"]))
    assert sum(results) == 1
    assert path.read_bytes() in (b"first", b"second")
    assert list(tmp_path.iterdir()) == [path]


def test_nonfinite_json_is_not_published(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_json(tmp_path / "record.json", {"value": float("nan")})
    assert not (tmp_path / "record.json").exists()
