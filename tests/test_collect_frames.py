import csv

import numpy as np

from tools import collect_frames


def test_collect_frames_creates_unlabelled_dataset(tmp_path, monkeypatch):
    class FakeCapture:
        def __init__(self):
            self.index = 0

        def isOpened(self):
            return True

        def read(self):
            self.index += 1
            if self.index > 5:
                return False, None
            return True, np.full((20, 30, 3), self.index, dtype=np.uint8)

        def set(self, prop, value):
            return True

        def release(self):
            pass

    monkeypatch.setattr(collect_frames.cv2, "VideoCapture", lambda source: FakeCapture())
    output = tmp_path / "heldout"
    assert collect_frames.collect_frames(0, output, every_n=2) == 2
    with (output / "labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["image_path"] for row in rows] == ["frame_0001.png", "frame_0002.png"]
    assert all(row["present"] == "" for row in rows)
