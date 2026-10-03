"""LIBERO rollouts of ActQuant's 3-bit Pi 0.5 on a Colab T4, driven from this machine.

Runs scripts/libero_rollout.py unchanged on a Colab VM through the Colab CLI (Linux/macOS only;
on Windows run this in WSL):

    python3 cloud/colab_libero.py --suites libero_spatial --trials 1

1. `colab new --gpu T4` (or reuse --session), upload the two scripts and their pins;
2. start libero_rollout.py as a detached process on the VM, so it does not depend on this
   machine staying connected;
3. watch it in cells that block for up to --watch-minutes each. A busy kernel keeps the session
   alive, and each cell ends on its own, so a dropped connection loses nothing;
4. after every watch cell, download the run folder without videos to runs/colab/<run name>/;
   at the end download everything and stop the session (unless --keep).

The VM's disk is gone when the session ends, so the local copy is the record. A run that is
interrupted can be picked up again with --attach <run name> while the session is alive.
Colab's T4 runtime: 2 vCPUs (one physical core), 12 GB RAM, driver 580 (CUDA 13.0), glibc 2.39.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


REPO = Path(__file__).resolve().parents[1]
PIN = "actquant-prebuilt-sm75.json"  # the T4 is sm_75
UPLOADS = ("scripts/libero_rollout.py", "scripts/actquant_build.py", f"requirements/{PIN}",
           "requirements/actquant-cuda-toolkit.txt", "requirements/libero-client.txt",
           "requirements/libero-server.txt", "requirements/fp16-export.txt")
REMOTE = "/content/eaq"
DRIVE_CACHE = "/content/drive/MyDrive/eaq-cache"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
STAGES = ("runtime", "client", "server", "profile", "rollout")

# Runs on the VM: start libero_rollout.py in its own session, detached from the kernel.
LAUNCH = r'''
import json, os, subprocess, sys
run = {run!r}
out = f"{remote}/runs/{{run}}"
os.makedirs(out, exist_ok=True)
state = f"{remote}/runs/{{run}}.json"
if os.path.exists(state):
    raise SystemExit(f"run {{run}} already started: " + open(state).read())
env = {{**os.environ, "EAQ_WORK_DIR": "{remote}/work", "PYTHONUNBUFFERED": "1"}}
for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV"):
    env.pop(name, None)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub==0.34.4"], check=True)
command = [sys.executable, "{remote}/scripts/libero_rollout.py", "--output", out, *{args!r}]
with open(f"{remote}/runs/{{run}}.stdout", "ab") as handle:
    process = subprocess.Popen(command, env=env, cwd="{remote}", stdin=subprocess.DEVNULL, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
json.dump({{"pid": process.pid, "command": command}}, open(state, "w"))
print(f"EAQ started pid {{process.pid}}")
'''

# Runs on the VM: print new log lines until the run ends or the time is up, then pack the run folder.
WATCH = r'''
import json, os, tarfile, time
run, remote, minutes = {run!r}, {remote!r}, {minutes!r}
state = json.load(open(f"{{remote}}/runs/{{run}}.json"))
log_path = f"{{remote}}/runs/{{run}}.stdout"
offset = state.get("offset", 0)
def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = open(f"/proc/{{pid}}/stat").read()
    return stat.rsplit(")", 1)[1].split()[0] != "Z"
deadline = time.monotonic() + minutes * 60
while True:
    with open(log_path, "rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
    if chunk:
        print(chunk.decode("utf-8", "replace"), end="", flush=True)
        offset += len(chunk)
    running = alive(state["pid"])
    if not running or time.monotonic() > deadline:
        break
    time.sleep(10)
state["offset"] = offset
json.dump(state, open(f"{{remote}}/runs/{{run}}.json", "w"))
out = f"{{remote}}/runs/{{run}}"
with tarfile.open(f"{{remote}}/runs/{{run}}-light.tar.gz", "w:gz") as archive:
    archive.add(out, arcname=run, filter=lambda info: None if "/videos" in info.name else info)
if not running:
    with tarfile.open(f"{{remote}}/runs/{{run}}-full.tar.gz", "w:gz") as archive:
        archive.add(out, arcname=run)
print(f"EAQ_WATCH {{json.dumps({{'running': running}})}}")
'''


def colab(*args, stdin=None, check=True, capture=False):
    command = ["colab", *args]
    result = subprocess.run(command, input=stdin, text=True, capture_output=capture)
    if check and result.returncode != 0:
        raise SystemExit(f"{' '.join(command)} failed ({result.returncode})"
                         + (f":\n{result.stdout}{result.stderr}" if capture else ""))
    return result


def unpack(tarball, local_runs):
    with tarfile.open(tarball) as archive:
        archive.extractall(local_runs, filter="data")
    tarball.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites", default="libero_spatial", help="comma-separated, or 'all' (run in sequence)")
    parser.add_argument("--trials", type=int, default=1, help="trials per task (ActQuant uses 50)")
    parser.add_argument("--stages", default="runtime,client,server,rollout",
                        help="libero_rollout.py stages, comma-separated; add 'profile' to time inference first")
    parser.add_argument("--profile-requests", type=int, default=20, help="requests per phase of the profile stage")
    parser.add_argument("--model", choices=["actquant-3bpw", "fp16", "q8"], default="actquant-3bpw",
                        help="the released 3-bit checkpoint, or the FP16 or Q8_0 reference (exported on the VM)")
    parser.add_argument("--drive", action="store_true",
                        help="mount Google Drive (colab drivemount) and keep reference exports in "
                             f"{DRIVE_CACHE}, so later sessions copy it instead of exporting again")
    parser.add_argument("--session", default="eaq-libero")
    parser.add_argument("--attach", metavar="RUN", help="watch a run already started in --session")
    parser.add_argument("--keep", action="store_true", help="leave the session running at the end")
    parser.add_argument("--watch-minutes", type=float, default=15.0)
    parser.add_argument("--suite-hours", type=float, default=11.0, help="libero_rollout.py time limit per suite")
    parser.add_argument("--client-threads", type=int, default=1)
    parser.add_argument("--server-threads", type=int, default=1, help="the T4 VM has one physical core")
    parser.add_argument("--local-runs", default=str(REPO / "runs" / "colab"))
    args = parser.parse_args()
    if not shutil.which("colab"):
        raise SystemExit("The Colab CLI is not installed (uv tool install google-colab-cli); it runs on Linux/macOS.")
    suites = list(SUITES) if args.suites == "all" else [name.strip() for name in args.suites.split(",") if name.strip()]
    if unknown := [name for name in suites if name not in SUITES]:
        raise SystemExit(f"Unknown suites {unknown}; choose from {', '.join(SUITES)} or 'all'.")
    stages = [name.strip() for name in args.stages.split(",") if name.strip()]
    if unknown := [name for name in stages if name not in STAGES]:
        raise SystemExit(f"Unknown stages {unknown}; choose from {', '.join(STAGES)}.")
    if not (REPO / "requirements" / PIN).is_file():
        raise SystemExit(f"requirements/{PIN} is missing: build the runtime with cuda_arch=75 and add its pin.")
    local_runs = Path(args.local_runs)
    local_runs.mkdir(parents=True, exist_ok=True)

    sessions = colab("sessions", capture=True, check=False).stdout
    if args.session not in sessions:
        if args.attach:
            raise SystemExit(f"Session {args.session} is not running; the run on it is lost.")
        print(f"[local] creating Colab session {args.session} (T4)", flush=True)
        colab("new", "-s", args.session, "--gpu", "T4")

    if args.attach:
        run = args.attach
    else:
        run = (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-t{args.trials}"
               + ("" if args.model == "actquant-3bpw" else f"-{args.model}"))
        colab("exec", "-s", args.session, stdin=f"import os\nfor d in ('scripts', 'requirements', 'runs'): "
                                                  f"os.makedirs('{REMOTE}/' + d, exist_ok=True)\n")
        for path in UPLOADS:
            colab("upload", "-s", args.session, str(REPO / path), f"{REMOTE}/{path}")
        if args.drive:
            print("[local] mounting Google Drive; approve the request if one appears", flush=True)
            colab("drivemount", "-s", args.session, "/content/drive")
        rollout_args = ["--model", args.model, *(["--model-cache", DRIVE_CACHE] if args.drive else []),
                        "--stages", *stages, "--profile-requests", str(args.profile_requests),
                        "--suites", *suites, "--trials", str(args.trials), "--suite-hours", str(args.suite_hours),
                        "--client-threads", str(args.client_threads), "--server-threads", str(args.server_threads),
                        "--prebuilt-pin", f"{REMOTE}/requirements/{PIN}",
                        "--toolkit-requirements", f"{REMOTE}/requirements/actquant-cuda-toolkit.txt",
                        "--client-requirements", f"{REMOTE}/requirements/libero-client.txt",
                        "--server-requirements", f"{REMOTE}/requirements/libero-server.txt"]
        colab("exec", "-s", args.session, "--timeout", "600",
              stdin=LAUNCH.format(run=run, remote=REMOTE, args=rollout_args))
    print(f"[local] run {run}; copies go to {local_runs / run}", flush=True)

    while True:
        watch = subprocess.Popen(["colab", "exec", "-s", args.session, "--timeout", str(args.watch_minutes * 60 + 600)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        watch.stdin.write(WATCH.format(run=run, remote=REMOTE, minutes=args.watch_minutes))
        watch.stdin.close()
        marker = None
        for line in watch.stdout:  # the run's log, streamed as the cell prints it
            if line.startswith("EAQ_WATCH "):
                marker = line.strip()
            else:
                print(line, end="", flush=True)
        watch.wait()
        if marker is None:
            if args.session not in colab("sessions", capture=True, check=False).stdout:
                print(f"[local] session {args.session} is gone; the last copy is in {local_runs / run}")
                return 1
            print("[local] watch cell failed; retrying in 60 s", flush=True)
            time.sleep(60)
            continue
        running = json.loads(marker[len("EAQ_WATCH "):])["running"]
        name = "light" if running else "full"
        tarball = local_runs / f"{run}-{name}.tar.gz"
        if colab("download", "-s", args.session, f"{REMOTE}/runs/{run}-{name}.tar.gz", str(tarball),
                 check=False, capture=True).returncode == 0:
            unpack(tarball, local_runs)
        if not running:
            break

    report = json.loads((local_runs / run / "report.json").read_text(encoding="utf-8"))
    print(f"[local] model: {report.get('options', {}).get('model')}, status: {report.get('status')}")
    client = report.get("stages", {}).get("client", {})
    if client.get("render"):
        print(f"[local] rendering: {client.get('gl')}, {client['render'].get('renderer')}, "
              f"{client['render'].get('step_ms')} ms per step")
    for phase, entry in report.get("stages", {}).get("profile", {}).get("phases", {}).items():
        total = entry["pi05_ms"].get("chain/Total", {}).get("p50")
        print(f"[local] profile {phase}: pi05 total p50 {total} ms, round trip p50 {entry['round_trip_ms']['p50']} ms")
    for suite, result in report.get("stages", {}).get("rollout", {}).get("results", {}).items():
        print(f"[local] {suite}: {result.get('successes')}/{result.get('episodes')} "
              f"(ActQuant's 3-bit model card: {result.get('actquant_reported')}), valid={result.get('valid')}")
    if not args.keep:
        colab("stop", "-s", args.session, check=False)
    return 0 if report.get("status") == "rollout_ok" else 1


if __name__ == "__main__":
    sys.exit(main())
