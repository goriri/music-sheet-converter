"""Unit tests for batch_eval mode in app.audio.job_main."""
import pytest
from app.audio.job_main import parse_gcs_uri, shard_manifest


def test_parse_gcs_uri_valid():
    bucket, blob = parse_gcs_uri("gs://my-bucket/path/to/file.wav")
    assert bucket == "my-bucket"
    assert blob == "path/to/file.wav"

    bucket2, blob2 = parse_gcs_uri("gs://cellular-cider-495602-r9-sheet-eval/manifest.json")
    assert bucket2 == "cellular-cider-495602-r9-sheet-eval"
    assert blob2 == "manifest.json"


def test_parse_gcs_uri_invalid():
    with pytest.raises(ValueError, match="must start with gs://"):
        parse_gcs_uri("https://storage.googleapis.com/b/obj")

    with pytest.raises(ValueError, match="missing bucket or object path"):
        parse_gcs_uri("gs://")

    with pytest.raises(ValueError, match="missing bucket or object path"):
        parse_gcs_uri("gs://bucket-only")


def test_shard_manifest_even_distribution():
    items = [{"id": f"item_{i:02d}"} for i in range(24)]
    task_count = 6

    all_sharded = []
    for task_index in range(task_count):
        sharded = shard_manifest(items, task_index, task_count)
        assert len(sharded) == 4
        all_sharded.extend(sharded)

    # All items should be covered exactly once without duplication
    assert len(all_sharded) == 24
    sharded_ids = sorted(it["id"] for it in all_sharded)
    original_ids = sorted(it["id"] for it in items)
    assert sharded_ids == original_ids


def test_shard_manifest_deterministic_sorting():
    items = [{"id": "z_track"}, {"id": "a_track"}, {"id": "m_track"}]
    # Sorted order should be a_track (idx 0), m_track (idx 1), z_track (idx 2)
    assert shard_manifest(items, 0, 3) == [{"id": "a_track"}]
    assert shard_manifest(items, 1, 3) == [{"id": "m_track"}]
    assert shard_manifest(items, 2, 3) == [{"id": "z_track"}]


def test_shard_manifest_invalid_indices():
    items = [{"id": "track_1"}]
    with pytest.raises(ValueError, match="task_count must be > 0"):
        shard_manifest(items, 0, 0)

    with pytest.raises(ValueError, match="task_index must be in"):
        shard_manifest(items, 3, 3)

    with pytest.raises(ValueError, match="task_index must be in"):
        shard_manifest(items, -1, 3)
