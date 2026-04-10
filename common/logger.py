import json
import os
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Simple logger for tracking scalar metrics."""

    def __init__(self, log_dir, use_tensorboard=True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.metrics = defaultdict(list)
        self.use_tensorboard = use_tensorboard
        self.writer = SummaryWriter(log_dir) if use_tensorboard else None

        self.log_file = os.path.join(log_dir, "metrics.json")

    def log(self, key, value, step):
        self.metrics[key].append((step, float(value)))

        if self.writer is not None:
            self.writer.add_scalar(key, value, step)

    def log_dict(self, metrics, step):
        for key, value in metrics.items():
            self.log(key, value, step)

    def save(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(dict(self.metrics), f, indent=2)

    def close(self):
        self.save()
        if self.writer is not None:
            self.writer.close()
