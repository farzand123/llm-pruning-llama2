import json
import threading
import time

import pytest

from llmprune import io_utils as U


def test_write_json_is_atomic_and_handles_numpy(tmp_path):
    import numpy as np

    p = tmp_path / "a" / "b.json"
    U.write_json(p, {"x": np.float64(1.5), "y": np.arange(3), "z": {"s": {2, 1}}})
    assert json.loads(p.read_text()) == {"x": 1.5, "y": [0, 1, 2], "z": {"s": [1, 2]}}
    assert [f.name for f in p.parent.iterdir()] == ["b.json"]  # no temp files left behind


def test_failed_write_leaves_the_old_file_intact(tmp_path):
    p = tmp_path / "r.json"
    U.write_json(p, {"ok": 1})

    class Boom:
        def tolist(self):
            raise RuntimeError("cannot serialise")

    with pytest.raises(RuntimeError):
        U.write_json(p, {"bad": Boom()})
    assert json.loads(p.read_text()) == {"ok": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["r.json"]


def test_package_versions_reports_missing_packages():
    v = U.package_versions(["numpy", "definitely_not_installed_pkg"])
    assert v["numpy"] and v["definitely_not_installed_pkg"] is None


def test_file_lock_serialises_critical_sections(tmp_path):
    order = []

    def worker(name):
        with U.file_lock(tmp_path / "x.lock", poll=0.01):
            order.append(("in", name))
            time.sleep(0.05)
            order.append(("out", name))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # sections never interleave: every "in" is immediately followed by the same worker's "out"
    for i in range(0, 6, 2):
        assert order[i][0] == "in" and order[i + 1] == ("out", order[i][1])
