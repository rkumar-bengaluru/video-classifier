import pandas as pd

from download import download_sample, find_video, ytdlp_cmd


def plan():
    return pd.DataFrame([
        {"video_id": "001", "youtube_id": "a", "stratum": "s|A", "role": "primary"},
        {"video_id": "002", "youtube_id": "b", "stratum": "s|A", "role": "primary"},
        {"video_id": "003", "youtube_id": "c", "stratum": "s|A", "role": "spare"},
        {"video_id": "004", "youtube_id": "d", "stratum": "s|A", "role": "spare"},
        {"video_id": "005", "youtube_id": "e", "stratum": "s|B", "role": "primary"},
    ])


def fake_fetch(out_dir, dead):
    def fetch(row):
        if row.video_id in dead:
            return False
        (out_dir / f"{row.video_id}.mp4").write_bytes(b"x")
        return True
    return fetch


def test_dead_primary_is_replaced_by_spare_from_same_stratum(tmp_path):
    final, missing = download_sample(plan(), tmp_path, fetch=fake_fetch(tmp_path, dead={"002"}))
    assert final.video_id.tolist() == ["001", "003", "005"]
    assert final.set_index("video_id").loc["003", "replaced"] == "002"
    assert missing == []


def test_missing_when_spares_exhausted(tmp_path):
    final, missing = download_sample(plan(), tmp_path, fetch=fake_fetch(tmp_path, dead={"002", "003", "004"}))
    assert missing == ["002"]
    assert len(final) == 2


def test_existing_file_is_not_refetched(tmp_path):
    (tmp_path / "001.webm").write_bytes(b"x")
    calls = []
    def fetch(row):
        calls.append(row.video_id)
        (tmp_path / f"{row.video_id}.mp4").write_bytes(b"x")
        return True
    download_sample(plan(), tmp_path, fetch=fetch)
    assert "001" not in calls
    assert find_video(tmp_path, "001").suffix == ".webm"


def test_ytdlp_cmd_is_240p_video_only():
    cmd = ytdlp_cmd("abc", "out", "007")
    assert "https://www.youtube.com/watch?v=abc" in cmd
    assert any("height<=240" in c for c in cmd)
