from pathlib import Path
from typing import Optional


class Logger:
    def __init__(self, log_dir: str, tag: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag or "run"
        self._log_file = self.log_dir / f"{self.tag}.log"
        self._csv_file = self.log_dir / f"{self.tag}.csv"
        self._header_written = False

    def log(self, step: int, kv: dict):
        line = f"[Step {step}] " + " | ".join(f"{k}={v:.4f}" for k, v in kv.items())
        print(line)
        with open(self._log_file, "a") as f:
            f.write(line + "\n")

    def log_csv(self, step: int, kv: dict):
        keys = ["step", *kv.keys()]
        values = [step, *kv.values()]
        with open(self._csv_file, "a") as f:
            if not self._header_written:
                f.write(",".join(keys) + "\n")
                self._header_written = True
            f.write(",".join(str(v) for v in values) + "\n")
