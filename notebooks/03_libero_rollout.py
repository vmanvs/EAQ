import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import json
    import os
    import shutil
    import signal
    import subprocess
    import sys
    import tempfile
    from datetime import datetime, timezone
    from importlib import metadata as importlib_metadata
    from pathlib import Path
    from zipfile import ZIP_DEFLATED, ZipFile

    import marimo as mo

    return (
        Path,
        ZIP_DEFLATED,
        ZipFile,
        datetime,
        importlib_metadata,
        io,
        json,
        mo,
        os,
        shutil,
        signal,
        subprocess,
        sys,
        tempfile,
        timezone,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # EAQ — LIBERO rollouts of ActQuant Pi 0.5

        Closed-loop LIBERO evaluation of ActQuant's released 3-bit Pi 0.5
        checkpoint (`ActQuant-Pi05-LIBERO-3bpw`) on the attached GPU, the way
        ActQuant evaluates it: its `tools/pi0.5/serve_policy.py` wraps the
        `pi05.so` binding in a WebSocket policy server, and openpi's
        `examples/libero/main.py` runs the LIBERO simulator as the client. Both
        files run **unmodified**, fetched at pinned commits and checked against
        pinned SHA-256 digests. The runtime is the CI-built package pinned in
        `requirements/actquant-prebuilt.json` (see the ActQuant runtime notebook).

        **What counts as a valid result.** `serve_policy.py` answers a failed
        inference with zero actions and then unnormalizes them, so the client
        cannot tell them from real actions. The script watches the server log:
        any `Inference failed` line, or a server crash, stops the run and marks
        it invalid. A suite is also invalid if the client caught an exception or
        finished fewer episodes than expected. Only valid suites have a success
        rate worth reading. It is shown with a 95% interval next to the rate
        ActQuant reports for 500 trials.

        **Stages**

        | Stage | What it does |
        | --- | --- |
        | `runtime` | CUDA runtime, prebuilt package and checkpoint, via `actquant_build.py` (reuses the work folder) |
        | `client` | Python 3.8 with the locked client packages, LIBERO at openpi's pinned commit, openpi's `main.py`; renders one LIBERO scene to pick a MuJoCo backend (EGL, Mesa EGL, OSMesa) |
        | `server` | `serve_policy.py` in a venv for the Python `pi05.so` was built for; three requests over the openpi protocol |
        | `rollout` | The policy server plus `main.py` for each selected suite, with every episode recorded |

        **Before running**

        1. Attach the GPU and open the **Server** preview.
        2. From the **Packages** panel install `huggingface_hub` (pinned in
           `requirements/actquant-build.txt`). Molab ships `uv`; install it
           there too only if the environment below says it is missing.

        The first run downloads about 5 GB (CUDA runtime, checkpoint, Python 3.8
        packages, LIBERO with its assets). Start small: one trial per task on
        `libero_spatial` is 10 episodes. A full suite at ActQuant's 50 trials
        per task is 500 episodes and takes hours. Molab is meant for interactive
        use and resets sandboxes that run long jobs, which wipes the work
        folder, so download the run ZIP as you go.
        """
    )
    return


@app.cell
def _(Path, mo, os):
    def locate_repository():
        starts = []
        try:
            notebook_location = mo.notebook_location()
            if notebook_location is not None:
                notebook_path = Path(notebook_location)
                if notebook_path.is_file() or notebook_path.suffix == ".py":
                    notebook_path = notebook_path.parent
                starts.append(notebook_path.resolve())
        except (OSError, RuntimeError, TypeError):
            pass
        starts.append(Path.cwd())
        for start in starts:
            for candidate in (start, *start.parents):
                if all((candidate / relative).is_file() for relative in FETCHED_FILES):
                    return candidate.resolve()
        return None

    def fetch_repository(repo, ref):
        """Download the run files at one resolved commit when Molab syncs only the notebook."""
        import json as _json
        import tempfile as _tempfile
        import urllib.error as _urlerror
        import urllib.request as _urlrequest

        token = os.environ.get("GITHUB_TOKEN")
        headers = {"User-Agent": "eaq-molab-libero-rollout", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        def get(url, accept):
            request = _urlrequest.Request(url, headers={**headers, "Accept": accept})
            with _urlrequest.urlopen(request, timeout=30) as response:
                return response.read()

        try:
            commit = _json.loads(get(f"https://api.github.com/repos/{repo}/commits/{ref}",
                                     "application/vnd.github+json"))["sha"]
            root = Path(_tempfile.gettempdir()) / "eaq-repo" / commit
            for relative in FETCHED_FILES:
                data = get(f"https://api.github.com/repos/{repo}/contents/{relative}?ref={commit}",
                           "application/vnd.github.raw")
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_bytes(data)
        except _urlerror.HTTPError as exc:
            # Never include headers or the token; the status code is enough to diagnose.
            hint = ""
            if exc.code == 404 and not token:
                hint = " The repository is private: add a read-only GITHUB_TOKEN to Molab Secrets."
            elif exc.code in (403, 429) and exc.headers.get("X-RateLimit-Remaining") == "0":
                hint = (" GitHub's unauthenticated API limit (60/hour per IP) is used up: wait for the "
                        "reset or add a read-only GITHUB_TOKEN to Molab Secrets (5,000/hour).")
            return None, None, f"GitHub HTTP {exc.code} fetching {repo}@{ref}.{hint}"
        except (_urlerror.URLError, OSError, KeyError, ValueError) as exc:
            return None, None, f"Could not fetch {repo}@{ref}: {type(exc).__name__}"
        return root, commit, None

    FETCHED_FILES = (
        "scripts/libero_rollout.py",
        "scripts/actquant_build.py",
        "requirements/actquant-build.txt",
        "requirements/actquant-cuda-toolkit.txt",
        "requirements/actquant-prebuilt.json",
        "requirements/libero-client.txt",
        "requirements/libero-server.txt",
    )
    repository_root = locate_repository()
    fetch_error = None
    eaq_commit = "unknown (local checkout)"
    repository_source = "local checkout" if repository_root is not None else None
    if repository_root is None:
        github_repo = os.environ.get("EAQ_GITHUB_REPO", "vmanvs/EAQ")
        github_ref = os.environ.get("EAQ_GIT_REF", "main")
        fetched_root, fetched_commit, fetch_error = fetch_repository(github_repo, github_ref)
        if fetched_root is not None:
            repository_root = fetched_root
            eaq_commit = fetched_commit
            repository_source = f"GitHub API: {github_repo}@{github_ref} → {fetched_commit[:12]}"
    return eaq_commit, fetch_error, repository_root, repository_source


@app.cell
def _(
    Path,
    eaq_commit,
    fetch_error,
    importlib_metadata,
    json,
    mo,
    os,
    repository_root,
    repository_source,
    shutil,
    subprocess,
    tempfile,
):
    work_dir = Path(os.environ.get("EAQ_WORK_DIR", Path(tempfile.gettempdir()) / "eaq-actquant"))

    def _version(name):
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return "not installed"

    _pins = {}
    if repository_root is not None:
        for _line in (repository_root / "requirements" / "actquant-build.txt").read_text(
                encoding="utf-8").splitlines():
            _requirement = _line.partition("#")[0].strip()
            if "==" in _requirement:
                _name, _pin = _requirement.split("==", 1)
                _pins[_name.strip()] = _pin.strip()

    try:
        _gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=10,
        ).stdout.strip() or "not visible"
    except (OSError, subprocess.SubprocessError):
        _gpu = "not visible (`nvidia-smi` unavailable)"

    try:
        _pin = json.loads((repository_root / "requirements" / "actquant-prebuilt.json").read_text(
            encoding="utf-8")) if repository_root is not None else {}
    except (OSError, ValueError):
        _pin = {}
    _pin_text = (f"`{_pin['tag']}`" if _pin.get("tag") else
                 "**none**: run the *ActQuant build* workflow and pin its release first")

    _existing = [label for label, relative in (
        ("CUDA runtime", "cuda-toolkit"), ("package", "prebuilt/current.json"),
        ("checkpoint", "checkpoints/actquant-pi05-libero-3bpw/pi05.gguf"),
        ("client venv", "libero/client-venv"), ("LIBERO", "libero/LIBERO"),
        ("server venv", "libero/server-venv")) if (work_dir / relative).exists()]
    _client = {}
    try:
        _client = json.loads((work_dir / "libero" / "client.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass

    _lines = [
        "## Environment",
        "",
        "- Run files: " + (f"`{repository_root}` ({repository_source})" if repository_root is not None
                           else f"not found — {fetch_error or 'no fetch attempted'}"),
        f"- EAQ commit: `{eaq_commit}`",
        f"- Runtime package: {_pin_text}",
        f"- GPU: `{_gpu}`",
        f"- `uv`: `{shutil.which('uv') or 'not found — install it from the Packages panel'}`",
        f"- Work folder `{work_dir}`: "
        + (", ".join(_existing) if _existing else "empty (the first run sets everything up)")
        + (f"; rendering backend `{_client['gl']}`" if _client.get("gl") else ""),
        "",
        "| Package | Installed | Pinned |",
        "| --- | --- | --- |",
        *(f"| `{name}` | `{_version(name)}` | `{_pins.get(name, '—')}` |" for name in ("huggingface_hub",)),
    ]
    mo.md("\n".join(_lines))
    return (work_dir,)


@app.cell
def _(mo):
    stage_picker = mo.ui.multiselect(
        options=["runtime", "client", "server", "rollout"],
        value=["runtime", "client", "server", "rollout"],
        label="Stages (always run in this order)",
    )
    suite_picker = mo.ui.multiselect(
        options=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        value=["libero_spatial"],
        label="Suites (10 tasks each)",
    )
    trials_picker = mo.ui.dropdown(
        options=["1", "2", "5", "10", "20", "50"],
        value="1",
        label="Trials per task (ActQuant: 50)",
    )
    run_rollout = mo.ui.run_button(
        label="Run LIBERO rollout",
        kind="success",
        tooltip="Starts scripts/libero_rollout.py with these settings in a new run folder.",
    )
    stop_rollout = mo.ui.run_button(
        label="Stop run",
        kind="danger",
        tooltip="Stops the running script; it shuts the policy server down and saves its report.",
    )
    mo.vstack([stage_picker, suite_picker, trials_picker, mo.hstack([run_rollout, stop_rollout], justify="start")])
    return run_rollout, stage_picker, stop_rollout, suite_picker, trials_picker


@app.cell
def _(
    Path,
    datetime,
    eaq_commit,
    fetch_error,
    json,
    os,
    repository_root,
    repository_source,
    run_rollout,
    signal,
    stage_picker,
    stop_rollout,
    subprocess,
    suite_picker,
    sys,
    timezone,
    trials_picker,
    work_dir,
):
    # The script runs detached with its output in a log file, so a disconnect does not stop it.
    # libero_last_run.json lets the status cell (and a new session) find it.
    last_run_file = work_dir / "libero_last_run.json"

    def read_json(path):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def rollout_alive(pid):
        # A finished child of this kernel is a zombie with an empty cmdline, so it counts as ended.
        try:
            return b"libero_rollout" in Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False

    launch = {"notice": None, "process": None}
    _last = read_json(last_run_file)
    if stop_rollout.value:
        if _last and rollout_alive(_last["pid"]):
            # The script's process group: the script itself and the LIBERO client. The script
            # stops the policy server, which runs in its own group.
            os.killpg(_last["pid"], signal.SIGTERM)
            launch["notice"] = f"Stop requested for run `{_last['run_id']}`."
        else:
            launch["notice"] = "No run is in progress."
    elif run_rollout.value:
        if repository_root is None:
            launch["notice"] = f"**Could not start:** run files not available: {fetch_error}"
        elif not stage_picker.value or ("rollout" in stage_picker.value and not suite_picker.value):
            launch["notice"] = "**Could not start:** select at least one stage, and a suite for `rollout`."
        elif _last and rollout_alive(_last["pid"]):
            launch["notice"] = f"Run `{_last['run_id']}` is still running, so no new run was started."
        else:
            _run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + os.urandom(3).hex()
            _run_dir = work_dir / "libero-runs" / _run_id
            _run_dir.mkdir(parents=True, exist_ok=False)
            _requirements = repository_root / "requirements"
            _command = [
                sys.executable, str(repository_root / "scripts" / "libero_rollout.py"),
                "--output", str(_run_dir), "--work-dir", str(work_dir),
                "--stages", *stage_picker.value,
                "--suites", *(suite_picker.value or ["libero_spatial"]),
                "--trials", trials_picker.value,
                "--actquant-script", str(repository_root / "scripts" / "actquant_build.py"),
                "--toolkit-requirements", str(_requirements / "actquant-cuda-toolkit.txt"),
                "--prebuilt-pin", str(_requirements / "actquant-prebuilt.json"),
                "--client-requirements", str(_requirements / "libero-client.txt"),
                "--server-requirements", str(_requirements / "libero-server.txt"),
            ]
            (_run_dir / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": _run_id,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "eaq_commit": eaq_commit,
                "repository_source": repository_source,
                "command": _command,
            }, indent=2) + "\n", encoding="utf-8")
            _env = os.environ.copy()
            _env["PYTHONUNBUFFERED"] = "1"
            with open(_run_dir / "notebook.log", "w", encoding="utf-8") as _log_handle:
                launch["process"] = subprocess.Popen(
                    _command, stdin=subprocess.DEVNULL, stdout=_log_handle, stderr=subprocess.STDOUT,
                    env=_env, start_new_session=True)
            last_run_file.write_text(json.dumps(
                {"run_id": _run_id, "run_dir": str(_run_dir), "pid": launch["process"].pid}) + "\n",
                encoding="utf-8")
    return last_run_file, launch, read_json, rollout_alive


@app.cell
def _(mo):
    # Re-runs the status cell on a timer instead of blocking the kernel while the rollout runs.
    status_refresh = mo.ui.refresh(options=["5s", "15s", "60s"], default_interval="15s",
                                   label="Rollout status refresh")
    status_refresh
    return (status_refresh,)


@app.cell
def _(
    Path,
    ZIP_DEFLATED,
    ZipFile,
    datetime,
    io,
    json,
    last_run_file,
    launch,
    mo,
    os,
    read_json,
    rollout_alive,
    status_refresh,
    timezone,
):
    status_refresh.value  # dependency: re-run on every refresh tick

    def _log_tail(path, lines=30):
        try:
            with open(path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 65536))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _archive(run_dir):
        # Report, logs, episodes, probe images and replay videos; venvs and the checkpoint stay behind.
        buffer = io.BytesIO()
        with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
            for file in sorted(run_dir.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(run_dir))
        return buffer.getvalue()

    def _percent(value):
        return "—" if value is None else f"{value * 100:.1f}%"

    def _image(path, caption):
        return mo.vstack([mo.image(src=path.read_bytes(), width=256), mo.md(caption)]) if path.is_file() else None

    _items = [mo.md(launch["notice"])] if launch["notice"] else []
    _last = read_json(last_run_file)
    if _last is None:
        _items.append(mo.md("No run yet. Choose settings and click **Run LIBERO rollout**."))
    else:
        _run_dir = Path(_last["run_dir"])
        _process = launch["process"]
        _running = (_process.poll() is None if _process is not None and _process.pid == _last["pid"]
                    else rollout_alive(_last["pid"]))
        _report = read_json(_run_dir / "report.json")
        _manifest = read_json(_run_dir / "manifest.json") or {}
        _progress = read_json(_run_dir / "progress.json")
        # Built lazily, only when clicked: the snapshot is current at download time.
        _download = mo.download(
            data=lambda run_dir=_run_dir: _archive(run_dir),
            filename=f"eaq-libero-rollout-{_last['run_id']}.zip",
            mimetype="application/zip",
            label="Download run artifacts (.zip)" + (", snapshot so far" if _running else ""),
        )

        if _running:
            try:
                _elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(_manifest["started_utc"])
                _elapsed_text = f"{int(_elapsed.total_seconds()) // 60} min"
            except (KeyError, ValueError):
                _elapsed_text = "unknown time"
            _stage = next((name for name, entry in ((_report or {}).get("stages") or {}).items()
                           if entry.get("status") == "running"), "starting")
            _text = [f"**Running** run `{_last['run_id']}`: stage `{_stage}`, started {_elapsed_text} ago."]
            if _progress and _stage == "rollout":
                _done, _expected = _progress["episodes_done"], _progress["episodes_expected"]
                _text.append(
                    f"\n\nEpisodes {_done} / {_expected}, successes {_progress['successes']}"
                    + (f" ({_progress['successes'] / _done * 100:.0f}%)" if _done else "")
                    + f"; suite `{_progress.get('suite')}`."
                )
                _episode = _progress.get("last_episode")
                if _episode:
                    _text.append(f"\n\nLast: task {_episode['task_index']} trial {_episode['trial']}, "
                                 f"{'success' if _episode['success'] else 'failure'} in {_episode['seconds']} s"
                                 f" — {_episode['task']}")
                _problems = {key: count for key, count in (_progress.get("server_problems") or {}).items() if count}
                if _problems:
                    _text.append(f"\n\n**Server problems:** `{_problems}` — the run is being stopped.")
            _text.append(f"\n\n```text\n{_log_tail(_run_dir / 'notebook.log')}\n```")
            _items.append(mo.md("".join(_text)))
            _items.append(_download)
        else:
            _code_file = _run_dir / "exit_code"
            _code_text = _code_file.read_text(encoding="utf-8").strip() if _code_file.is_file() else ""
            _exit_code = _process.returncode if _process is not None and _process.pid == _last["pid"] else (
                int(_code_text) if _code_text.lstrip("-").isdigit() else None)
            if "finished_utc" not in _manifest:
                _manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
                _manifest["exit_code"] = _exit_code
                (_run_dir / "manifest.json").write_text(json.dumps(_manifest, indent=2) + "\n",
                                                        encoding="utf-8")
            if _report is None:
                _items.append(mo.md(
                    f"**No report was written** for run `{_last['run_id']}` (exit code `{_exit_code}`).\n\n"
                    "```text\n" + _log_tail(_run_dir / "notebook.log") + "\n```"
                ))
            else:
                # A report still saying "running" means the process was killed mid-stage.
                _status = "interrupted" if _report.get("status") == "running" else _report.get("status")
                _rows = ["| Stage | Status | Time | Note |", "| --- | --- | --- | --- |"]
                for _name, _entry in _report.get("stages", {}).items():
                    _note = _entry.get("error") or _entry.get("reason") or ""
                    _entry_status = "interrupted" if _entry["status"] == "running" else _entry["status"]
                    _rows.append(f"| `{_name}` | **{_entry_status}** | {_entry.get('seconds', '')} s | "
                                 f"{str(_note).replace('|', '/')[:400]} |")
                _items.append(mo.md(f"## Result: `{_status}`\n\nRun folder: `{_run_dir}`\n\n" + "\n".join(_rows)))

                _stages = _report.get("stages", {})
                _results = (_stages.get("rollout") or {}).get("results") or {}
                if _results:
                    _table = ["| Suite | Valid | Episodes | Success | 95% interval | ActQuant (500 trials) | "
                              "Server infer p50 / p95 |", "| --- | --- | --- | --- | --- | --- | --- |"]
                    for _suite, _result in _results.items():
                        _interval = _result.get("wilson95")
                        _timing = (_result.get("server") or {}).get("infer_ms") or {}
                        _table.append(
                            f"| `{_suite}` | {'yes' if _result['valid'] else '**no**'} | "
                            f"{_result['episodes']} / {_result['expected']} | "
                            f"{_result['successes']} ({_percent(_result['success_rate'])}) | "
                            + (f"{_percent(_interval[0])}–{_percent(_interval[1])}" if _interval else "—")
                            + f" | {_percent(_result.get('actquant_reported'))} | "
                            f"{_timing.get('p50', '—')} / {_timing.get('p95', '—')} ms |"
                        )
                    _notes = [f"- `{_suite}`: " + "; ".join(_result["problems"])
                              for _suite, _result in _results.items() if _result["problems"]]
                    _items.append(mo.md("### LIBERO results\n\n" + "\n".join(_table)
                                        + ("\n\n**Invalid:**\n\n" + "\n".join(_notes) if _notes else "")))
                    _items.append(mo.accordion({
                        "Per-task results": mo.md("\n".join(
                            ["| Suite | Task | Successes |", "| --- | --- | --- |"]
                            + [f"| `{_suite}` | {_task} | {_counts['successes']} / {_counts['episodes']} |"
                               for _suite, _result in _results.items()
                               for _task, _counts in _result["per_task"].items()]))
                    }))

                _client = _stages.get("client") or {}
                _server = _stages.get("server") or {}
                _checks = []
                if _client.get("render"):
                    _checks.append(f"- Rendering: `{_client['gl']}`, {_client['render']['step_ms']} ms per "
                                   f"simulator step; Python `{_client.get('python')}`")
                if (_server.get("probe") or {}).get("actions_shape"):
                    _probe = _server["probe"]
                    _checks.append(f"- Policy server: actions `{_probe['actions_shape']}`, range "
                                   f"[{_probe['min']:.3f}, {_probe['max']:.3f}], round trips "
                                   f"{_probe['round_trip_ms']} ms, repeat diff `{_probe['repeat_max_abs_diff']:.2e}`")
                if _checks:
                    _items.append(mo.md("### Checks\n\n" + "\n".join(_checks)))
                _images = [image for image in (
                    _image(_run_dir / (_client.get("render_image") or "missing"), "Render probe (what `main.py` sends)"),
                    _image(_run_dir / "server_first_image.png", "First image the server received"),
                ) if image is not None]
                if _images:
                    _items.append(mo.hstack(_images, justify="start"))

                _deviations = "\n".join(f"- {item}" for item in _report.get("deviations", [])) or "- none"
                _items.append(mo.md(f"### Deviations from ActQuant's documented setup\n\n{_deviations}"))

                _failed = _report.get("failed_stage")
                if _failed:
                    _tail = _stages[_failed].get("tail") or _log_tail(_run_dir / "notebook.log")
                    _items.append(mo.md(f"### `{_failed}` output tail\n\n```text\n{_tail}\n```"))
                _items.append(mo.accordion({
                    "report.json": mo.md("```json\n" + json.dumps(_report, indent=2) + "\n```")
                }))
            _items.append(_download)
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
