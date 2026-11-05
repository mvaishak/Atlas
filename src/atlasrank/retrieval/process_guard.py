"""Independent POSIX parent/deadline watchdog, including blocking native children.

Launch this wrapper in its own session. The command inherits that process group;
if the original supervisor dies, the wrapper kills its entire group. No ML deps.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def run(parent: int, deadline: float, rss_limit: int, command: list[str]) -> int:
    if os.getpgrp() != os.getpid():
        raise PermissionError("Watchdog requires a dedicated process session")
    if os.getppid() != parent or time.monotonic() >= deadline:
        raise PermissionError("Supervisor/deadline lost before command startup")
    env = dict(os.environ)
    for name in ("ATLAS_COLAB_PARENT_PID", "ATLAS_MAC_PARENT_PID"):
        if name in env:
            env[name] = str(os.getpid())
    child = subprocess.Popen(command, env=env)
    try:
        while child.poll() is None:
            reason = None
            if os.getppid() != parent:
                reason = "supervisor_lost"
            elif time.monotonic() >= deadline:
                reason = "deadline"
            status = Path(f"/proc/{child.pid}/status")
            if status.exists():
                try:
                    for line in status.read_text().splitlines():
                        if line.startswith("VmRSS:") and int(line.split()[1]) * 1024 > rss_limit:
                            reason = "rss_limit"
                except FileNotFoundError:
                    pass  # Child exited between poll and /proc read.
            if reason:
                print(json.dumps({"event": "watchdog_stop", "reason": reason}), flush=True)
                os.killpg(os.getpgrp(), signal.SIGKILL)
            time.sleep(0.05)
    finally:
        if child.poll() is None:
            os.killpg(os.getpgrp(), signal.SIGKILL)
    return child.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--rss-limit", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Command required")
    raise SystemExit(run(args.parent, args.deadline, args.rss_limit, command))


if __name__ == "__main__":
    main()
