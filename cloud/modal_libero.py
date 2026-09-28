"""LIBERO rollouts of ActQuant's 3-bit Pi 0.5 on Modal, one L4 container per suite.

Runs scripts/libero_rollout.py unchanged. The container image already holds the runtime stage
(CUDA runtime wheels, the prebuilt sm_89 package, the checkpoint) and the client stage (Python
3.8 venv, LIBERO), so a run only repeats those stages as checks, then installs the server venv,
starts the policy server on the GPU and runs openpi's main.py. Run folders go to the
`eaq-libero` Volume, committed every few minutes so a crash still leaves evidence.

    modal run --detach cloud/modal_libero.py --suites libero_spatial --trials 1
    modal volume get eaq-libero runs/<run name> ./modal-runs

No secrets: the checkpoint and the GitHub release are public. Editing anything in scripts/ or
requirements/ rebuilds the setup layer of the image (about 10 minutes, once).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import threading

import modal


APP_NAME = "eaq-libero"
REPO = Path(__file__).resolve().parents[1]
WORK = "/opt/eaq-work"  # EAQ_WORK_DIR: shared by actquant_build.py and libero_rollout.py
PIN = "actquant-prebuilt-sm89.json"  # the L4 is sm_89; requirements/actquant-prebuilt.json is Molab's sm_120
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
COMMIT_SECONDS = 300

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(APP_NAME, create_if_missing=True)

image = (
    # The runtime was built in this image (Debian 13, glibc 2.41, Python 3.13), so pi05.so imports.
    modal.Image.from_registry("python:3.13-slim-trixie")
    .apt_install("git", "ca-certificates", "libegl1", "libegl-mesa0", "libgl1", "libgl1-mesa-dri", "libosmesa6")
    .pip_install("huggingface_hub==0.34.4", "uv==0.12.15")
    .env({"EAQ_WORK_DIR": WORK, "PYTHONUNBUFFERED": "1"})
    # Setup layer: the runtime and client stages need no GPU, so they run once at image build.
    .add_local_dir(REPO / "scripts", "/setup/scripts", copy=True, ignore=["__pycache__"])
    .add_local_dir(REPO / "requirements", "/setup/requirements", copy=True)
    .run_commands(f"python /setup/scripts/libero_rollout.py --output /setup/run --stages runtime client "
                  f"--prebuilt-pin /setup/requirements/{PIN}")
    # The files a run uses, mounted at start: edits here do not rebuild anything.
    .add_local_dir(REPO / "scripts", "/repo/scripts", ignore=["__pycache__"])
    .add_local_dir(REPO / "requirements", "/repo/requirements")
)


def _commit_until(process):
    while True:
        try:
            process.wait(timeout=COMMIT_SECONDS)
            return
        except subprocess.TimeoutExpired:
            volume.commit()


# cpu is physical cores (2 vCPU each). The Molab rollout used about 1.1 cores, almost all of it
# the simulator and renderer; the policy server used 0.07.
@app.function(image=image, gpu="L4", cpu=2.0, memory=16384, timeout=24 * 3600, volumes={"/results": volume})
def evaluate(suite: str, trials: int, run_name: str, client_threads: int = 1) -> dict:
    output = Path("/results/runs") / run_name / suite
    command = [sys.executable, "/repo/scripts/libero_rollout.py", "--output", str(output),
               "--suites", suite, "--trials", str(trials), "--client-threads", str(client_threads),
               "--prebuilt-pin", f"/repo/requirements/{PIN}"]
    process = subprocess.Popen(command)  # its log lines go to the Modal logs
    committer = threading.Thread(target=_commit_until, args=(process,), daemon=True)
    committer.start()
    process.wait()
    committer.join()
    volume.commit()
    try:
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"suite": suite, "status": "no report", "returncode": process.returncode}
    stages = report.get("stages", {})
    result = stages.get("rollout", {}).get("results", {}).get(suite, {})
    failed = report.get("failed_stage")
    return {"suite": suite, "status": report.get("status"), "returncode": process.returncode,
            "gpu": report.get("host", {}).get("gpu"),
            **({"failed_stage": failed, "error": stages.get(failed, {}).get("error")} if failed else {}),
            **{key: result.get(key) for key in ("valid", "problems", "episodes", "expected", "successes",
                                               "success_rate", "wilson95", "actquant_reported", "seconds")}}


@app.local_entrypoint()
def main(suites: str = "libero_spatial", trials: int = 1, client_threads: int = 1):
    """--suites: comma-separated, or "all"; each suite gets its own L4 container, in parallel."""
    chosen = list(SUITES) if suites == "all" else [name.strip() for name in suites.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in SUITES]
    if unknown:
        raise SystemExit(f"Unknown suites {unknown}; choose from {', '.join(SUITES)} or 'all'.")
    if not 1 <= trials <= 50:
        raise SystemExit("--trials must be between 1 and 50.")
    run_name = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-t{trials}"
    print(f"run {run_name}: {', '.join(chosen)}, {trials} trial(s) per task")
    print(f"results: modal volume get {APP_NAME} runs/{run_name} ./modal-runs")
    for summary in evaluate.starmap([(suite, trials, run_name, client_threads) for suite in chosen]):
        print(json.dumps(summary, indent=2))
