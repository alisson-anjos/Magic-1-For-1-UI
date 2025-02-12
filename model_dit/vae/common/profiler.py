import datetime
import torch
from torch.profiler.profiler import ProfilerAction, profile

from ..common.distributed import get_global_rank, get_local_rank
from ..common.fs import mkdir


class Profiler(profile):
    def __init__(self, save_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        self.snapshot_recording = False
        self.timestamp = None
        self.action_map[(ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE)].append(
            self.start_snapshot
        )
        self.action_map[(ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE)].append(
            self.start_snapshot
        )
        self.action_map[(ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE)].append(
            self.start_snapshot
        )
        self.action_map[(ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE)].append(
            self.start_snapshot
        )

    def start_snapshot(self):
        torch.cuda.memory._record_memory_history(max_entries=100000)
        self.snapshot_recording = True

    def stop_snapshot(self):
        if self.snapshot_recording:
            timestamp = [datetime.datetime.now().strftime("%Y%m%d%H%M%S")]
            torch.distributed.broadcast_object_list(timestamp)
            self.timestamp = timestamp

            # Set local saving path.
            if get_local_rank() == 0:
                mkdir(f"{self.save_dir}/{timestamp[0]}")
            torch.distributed.barrier()

            torch.cuda.memory._dump_snapshot(
                f"{self.save_dir}/{timestamp[0]}/"
                + f"rank_{get_global_rank()}"
                + "_snapshot.pickle"
            )
            torch.cuda.memory._record_memory_history(enabled=None)
            self.snapshot_recording = False

    def stop_trace(self):
        self.stop_snapshot()
        super().stop_trace()

    def stop(self):
        self.stop_snapshot()
        super().stop()
