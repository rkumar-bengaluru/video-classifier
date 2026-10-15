import numpy as np
import pandas as pd
import pytest

from labels import build_labels, contact_sheet, load_overrides, sheet_pages


def sample():
    return pd.DataFrame({
        "video_id": ["001", "002"], "youtube_id": ["a", "b"], "duration": ["short", "long"],
        "domain": ["Sports Competition", "Knowledge"], "sub_category": ["Football", "Technology"],
    })


def test_auto_labels_from_mapping_without_overrides():
    df = build_labels(sample())
    assert df.label_auto.tolist() == ["sports_broadcast", "cinematic_documentary"]
    assert (df.label_final == df.label_auto).all()
    assert not df.label_changed.any()


def test_overrides_change_final_label_only():
    o = pd.DataFrame({"video_id": ["002"], "label_final": ["presentation_screen"], "note": ["slides"]})
    df = build_labels(sample(), o).set_index("video_id")
    assert df.loc["002", "label_auto"] == "cinematic_documentary"
    assert df.loc["002", "label_final"] == "presentation_screen"
    assert bool(df.loc["002", "label_changed"]) and df.loc["002", "note"] == "slides"


def test_override_with_unknown_class_is_rejected():
    o = pd.DataFrame({"video_id": ["001"], "label_final": ["podcast"], "note": [""]})
    with pytest.raises(ValueError):
        build_labels(sample(), o)


def test_missing_overrides_file_is_empty(tmp_path):
    assert len(load_overrides(tmp_path / "none.csv")) == 0


def test_contact_sheet_and_pages():
    rgb = np.zeros((16, 256, 256, 3), np.uint8)
    sheet = contact_sheet(rgb, "001 | Football -> sports_broadcast")
    assert sheet.ndim == 3 and sheet.shape[1] == 4 * 160
    pages = sheet_pages([sheet] * 7, per_page=6)
    assert len(pages) == 2
