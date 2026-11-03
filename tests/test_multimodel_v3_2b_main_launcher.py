from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "slurm/multimodel/aspd_v3_2b_main.sbatch"


def test_main_launcher_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_main_launcher_waits_for_signal_checkpoint_before_requeue() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'pgrep -P "${TRAINER_PID}"' in text
    assert 'kill -USR1 "${worker_pids[@]}"' in text
    assert 'kill -USR1 -- "-${TRAINER_PID}"' not in text
    assert '"${TRAINER_RC}" -eq 138' in text
    assert 'kill -0 "${TRAINER_PID}"' in text
    assert '"${END_COMPLETE_STEP}" -le "${START_COMPLETE_STEP}"' in text
    assert "manifest_sha256" in text
    assert "Refusing requeue without a new verified COMPLETE checkpoint" in text
    assert '"${TRAINER_RC}" -eq 99' not in text


def test_main_launcher_recovers_sqlite_state_before_initializing_nccl() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    recovery = text.index("ledger.recover_to_receipt(receipt)")
    torchrun = text.index("setsid ../repo/.venv/bin/torchrun")
    assert recovery < torchrun
    assert "SQLiteStreamDeduplicator" in text
    assert "PRE_NCCL_RECOVERY_PASS" in text
    assert 'dedup.rollback(int(stream_state["dedup_high_water"]))' in text


def test_main_launcher_preserves_interim_report_before_requeue() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'signal_reports/benchmark_report-step-${END_COMPLETE_STEP}.json' in text
    assert 'mv "${RUN_DIR}/benchmark_report.json" "${SIGNAL_REPORT}"' in text


def test_torchrun_workers_can_checkpoint_when_signalled_without_parent(tmp_path: Path) -> None:
    worker = tmp_path / "signal_worker.py"
    worker.write_text(
        """
import os
import signal
import time
from pathlib import Path

rank = os.environ["LOCAL_RANK"]
root = Path(os.environ["SIGNAL_TEST_DIR"])
stop = False

def request_stop(_signum, _frame):
    global stop
    stop = True

signal.signal(signal.SIGUSR1, request_stop)
(root / f"ready-{rank}").write_text("ready")
while not stop:
    time.sleep(0.01)
(root / f"checkpoint-{rank}").write_text("complete")
raise SystemExit(99)
""".lstrip(),
        encoding="utf-8",
    )
    torchrun = Path(sys.executable).with_name("torchrun")
    env = dict(os.environ, SIGNAL_TEST_DIR=str(tmp_path))
    process = subprocess.Popen(
        [
            str(torchrun),
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=3",
            str(worker),
        ],
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while len(list(tmp_path.glob("ready-*"))) != 3 and time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    f"torchrun exited before workers were ready: rc={process.returncode}\n"
                    f"stdout={stdout}\nstderr={stderr}"
                )
            time.sleep(0.05)
        assert len(list(tmp_path.glob("ready-*"))) == 3

        children = subprocess.check_output(
            ["pgrep", "-P", str(process.pid)], text=True
        ).split()
        assert len(children) == 3
        for child in children:
            os.kill(int(child), signal.SIGUSR1)

        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode != 0, (stdout, stderr)
        assert len(list(tmp_path.glob("checkpoint-*"))) == 3, (stdout, stderr)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
