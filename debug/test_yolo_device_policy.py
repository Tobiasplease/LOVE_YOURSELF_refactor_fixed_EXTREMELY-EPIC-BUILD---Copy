"""Sep 13: the YOLO device policy without a GPU — nvidia-smi and the model class
are faked. Covers: CPU from the start on a full card; a fresh CPU model on OOM;
the periodic CUDA retry (success swaps the model, failure keeps the CPU one and
doubles the wait). Run: .venv/bin/python debug/test_yolo_device_policy.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import perception.object_detection as od  # noqa: E402

LOG = []


class _Inner:
    def __init__(self, fail):
        self.fail, self.device = fail, "cpu"

    def to(self, dev):
        if self.fail and dev == "cuda":
            raise RuntimeError("CUDA error: out of memory")
        self.device = dev
        return self


class FakeYOLO:
    fail_cuda = True
    loads = 0

    def __init__(self, path):
        FakeYOLO.loads += 1
        self.path, self.model = path, _Inner(FakeYOLO.fail_cuda)


od.YOLO = FakeYOLO
od.print = lambda *a, **k: LOG.append(" ".join(str(x) for x in a))
od.YOLO_VRAM_MIN_MIB, od.YOLO_CUDA_RETRY_S = 1200, 120.0
free = {"v": 300}
od._free_vram_mib = lambda: free["v"]

t = od.ObjectDetectionThread(model_path="fake.pt")
assert t.force_cpu, "300 MiB free at start must mean CPU"
assert FakeYOLO.loads == 1

t.force_cpu = False  # pretend we were on CUDA and hit OOM
first = t.model
t._fall_back_to_cpu(now=1000.0)
assert t.force_cpu and t.model is not first and FakeYOLO.loads == 2, "OOM must reload a fresh model"

t._maybe_retry_cuda(now=1050.0)
assert FakeYOLO.loads == 2, "no retry before the wait has passed"
t._maybe_retry_cuda(now=1200.0)
assert FakeYOLO.loads == 2 and t.force_cpu, "no retry while the card is full"

free["v"] = 3000
FakeYOLO.fail_cuda = True
t._maybe_retry_cuda(now=1400.0)
assert t.force_cpu and FakeYOLO.loads == 3 and t._cuda_retry_wait == 240.0, (t.force_cpu, FakeYOLO.loads, t._cuda_retry_wait)
assert t.model.model.device == "cpu", "the live model must be untouched by the failed attempt"

t._maybe_retry_cuda(now=1500.0)
assert FakeYOLO.loads == 3, "backoff: 240 s must pass before the next try"
FakeYOLO.fail_cuda = False
t._maybe_retry_cuda(now=1700.0)
assert not t.force_cpu and t.model.model.device == "cuda" and FakeYOLO.loads == 4, "room appeared → back on CUDA"
assert t._cuda_retry_wait == 240.0, "the wait is kept on success (no flapping)"
print("OK —", len(LOG), "log lines:", *[l[:70] for l in LOG], sep="\n  ")
