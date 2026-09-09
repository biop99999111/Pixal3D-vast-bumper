import contextlib
import time

from .common import safe_error, write_json


class Recorder:
    def __init__(self, output):
        self.output = output
        self.events = []

    @contextlib.contextmanager
    def stage(self, name):
        import torch
        event = {"stage": name, "status": "running"}
        self.events.append(event)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        write_json(self.output, self.events)
        try:
            yield
            torch.cuda.synchronize()
            event["status"] = "complete"
        except Exception as error:
            event.update(status="failed", error=safe_error(error))
            raise
        finally:
            event.update(seconds=time.monotonic()-start,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(self.output, self.events)

    def wrap(self, target, name, label=None):
        original = getattr(target, name)
        def measured(*args, **kwargs):
            with self.stage(label or name):
                return original(*args, **kwargs)
        setattr(target, name, measured)
