import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import json
    import os
    import shutil
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
        subprocess,
        sys,
        tempfile,
        timezone,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # EAQ — ActQuant runtime on Molab

        Runs [ActQuant](https://github.com/arashakb/ActQuant)'s Pi 0.5 runtime,
        at its pinned commit, on the attached GPU: **one** inference with the
        released 3-bit checkpoint (`ActQuant-Pi05-LIBERO-3bpw`, about 2.4 GB).

        The runtime is **not compiled here**. Long compiles inside Molab got the
        sandbox reset, and Molab is meant for interactive use. The GitHub
        Actions workflow `.github/workflows/actquant-build.yml` compiles it for
        this GPU in a container matching Molab (Debian 13, Python 3.13). It
        smoke-tests it on CPU and publishes it as a release. This notebook
        downloads the release pinned in `requirements/actquant-prebuilt.json`,
        checks its SHA-256, and runs it against a pinned CUDA 13.0 runtime.
        That runtime is installed from pip wheels into a private folder;
        PyTorch's environment is not touched. Every departure from ActQuant's
        documented setup, including those made at build time, is listed as a
        deviation in the report.

        The smoke test uses a synthetic image and a fixed prompt. It shows that
        the runtime executes on this GPU and produces finite actions; it says
        nothing about task success. No LIBERO rollout is run here.

        **Before running**

        1. Attach the GPU and open the **Server** preview.
        2. From the **Packages** panel install `huggingface_hub`, pinned in
           `requirements/actquant-build.txt`. Do not install CUDA packages.

        No token is needed: the package, ActQuant and the checkpoint are public.
        The whole run takes a few minutes. The package, toolkit and checkpoint
        stay in a work folder for the session, so single stages can be rerun.
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
        """Download the build files at one resolved commit when Molab syncs only the notebook."""
        import json as _json
        import tempfile as _tempfile
        import urllib.error as _urlerror
        import urllib.request as _urlrequest

        token = os.environ.get("GITHUB_TOKEN")
        headers = {"User-Agent": "eaq-molab-actquant-build", "X-GitHub-Api-Version": "2022-11-28"}
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
        "scripts/actquant_build.py",
        "requirements/actquant-build.txt",
        "requirements/actquant-cuda-toolkit.txt",
        "requirements/actquant-prebuilt.json",
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
    build_script = repository_root / "scripts" / "actquant_build.py" if repository_root else None
    build_requirements = repository_root / "requirements" / "actquant-build.txt" if repository_root else None
    toolkit_requirements = (
        repository_root / "requirements" / "actquant-cuda-toolkit.txt" if repository_root else None
    )
    prebuilt_pin = repository_root / "requirements" / "actquant-prebuilt.json" if repository_root else None
    return (
        build_requirements,
        build_script,
        eaq_commit,
        fetch_error,
        prebuilt_pin,
        repository_root,
        repository_source,
        toolkit_requirements,
    )


@app.cell
def _(
    Path,
    build_requirements,
    eaq_commit,
    fetch_error,
    importlib_metadata,
    json,
    mo,
    os,
    prebuilt_pin,
    repository_root,
    repository_source,
    shutil,
    subprocess,
    tempfile,
    toolkit_requirements,
):
    work_dir = Path(os.environ.get("EAQ_WORK_DIR", Path(tempfile.gettempdir()) / "eaq-actquant"))

    def _version(name):
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return "not installed"

    _pins = {}
    if build_requirements is not None:
        for _line in build_requirements.read_text(encoding="utf-8").splitlines():
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
        _anchor = work_dir if work_dir.exists() else work_dir.parent
        _disk = shutil.disk_usage(_anchor)
        _disk_text = (f"{_disk.free / 2**30:.1f} GiB free" if _disk.total <= 2**50
                      else "not measurable (sandbox reports a placeholder size)")
    except OSError as _exc:
        _disk_text = f"unavailable ({type(_exc).__name__})"

    _existing = [name for name in ("cuda-toolkit", "prebuilt", "checkpoints") if (work_dir / name).exists()]
    try:
        _pin = json.loads(prebuilt_pin.read_text(encoding="utf-8")) if prebuilt_pin is not None else {}
    except (OSError, ValueError):
        _pin = {}
    _pin_text = (f"`{_pin['tag']}` of `{_pin['repository']}`, SHA-256 `{_pin['sha256'][:12]}…`"
                 if _pin.get("tag") and _pin.get("sha256") else
                 "**none yet**: run the *ActQuant build* workflow and copy the pin from its summary")

    _lines = [
        "## Environment",
        "",
        f"- Build files: "
        + (f"`{repository_root}` ({repository_source})" if repository_root is not None
           else f"not found — {fetch_error or 'no fetch attempted'}"),
        f"- EAQ commit: `{eaq_commit}`",
        "- CUDA toolkit pins (installed by the `toolkit` stage): "
        + (", ".join(f"`{_l.strip()}`" for _l in toolkit_requirements.read_text(encoding="utf-8").splitlines()
                     if "==" in _l and not _l.lstrip().startswith("#"))
           if toolkit_requirements is not None else "not found"),
        f"- Runtime package: {_pin_text}",
        f"- GPU: `{_gpu}`",
        f"- Work folder: `{work_dir}`; disk {_disk_text}; "
        + (f"already contains {', '.join(f'`{item}`' for item in _existing)}" if _existing else "empty"),
        "",
        "| Package | Installed | Pinned |",
        "| --- | --- | --- |",
        *(f"| `{name}` | `{_version(name)}` | `{_pins.get(name, '—')}` |"
          for name in ("torch", "huggingface_hub")),
    ]
    mo.md("\n".join(_lines))
    return (work_dir,)


@app.cell
def _(mo):
    # Compiling (source/configure/build/package) happens in GitHub Actions, not here.
    stage_picker = mo.ui.multiselect(
        options=["toolkit", "fetch", "download", "infer"],
        value=["toolkit", "fetch", "download", "infer"],
        label="Stages (always run in this order)",
    )
    cpu_check = mo.ui.checkbox(label="Also run the CLI on CPU and compare with CUDA (slower)")
    run_build = mo.ui.run_button(
        label="Run ActQuant",
        kind="success",
        tooltip="Starts scripts/actquant_build.py with the selected stages in a new run folder.",
    )
    mo.vstack([stage_picker, cpu_check, run_build])
    return cpu_check, run_build, stage_picker


@app.cell
def _(
    Path,
    build_script,
    cpu_check,
    datetime,
    eaq_commit,
    fetch_error,
    json,
    os,
    prebuilt_pin,
    repository_source,
    run_build,
    stage_picker,
    subprocess,
    sys,
    timezone,
    toolkit_requirements,
    work_dir,
):
    # The script runs detached with its output in a log file, so a disconnect does not stop it.
    # last_run.json lets the status cell (and a new session) find it.
    last_run_file = work_dir / "last_run.json"

    def read_json(path):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def build_alive(pid):
        # A finished child of this kernel is a zombie with an empty cmdline, so it counts as ended.
        try:
            return b"actquant_build" in Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False

    launch = {"notice": None, "process": None}
    if run_build.value:
        _last = read_json(last_run_file)
        if build_script is None:
            launch["notice"] = f"**Could not start:** build script not available: {fetch_error}"
        elif not stage_picker.value:
            launch["notice"] = "**Could not start:** select at least one stage."
        elif _last and build_alive(_last["pid"]):
            launch["notice"] = f"Run `{_last['run_id']}` is still running, so no new run was started."
        else:
            _run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + os.urandom(3).hex()
            _run_dir = work_dir / "runs" / _run_id
            _run_dir.mkdir(parents=True, exist_ok=False)
            _command = [sys.executable, str(build_script), "--output", str(_run_dir),
                        "--work-dir", str(work_dir), "--toolkit-requirements", str(toolkit_requirements),
                        "--prebuilt-pin", str(prebuilt_pin), "--stages", *stage_picker.value]
            if cpu_check.value:
                _command.append("--cpu-check")
            (_run_dir / "manifest.json").write_text(json.dumps({
                "schema_version": 2,
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
    return build_alive, last_run_file, launch, read_json


@app.cell
def _(mo):
    # Re-runs the status cell on a timer instead of blocking the kernel while the build runs.
    status_refresh = mo.ui.refresh(options=["5s", "15s", "60s"], default_interval="5s",
                                   label="Build status refresh")
    status_refresh
    return (status_refresh,)


@app.cell
def _(
    Path,
    ZIP_DEFLATED,
    ZipFile,
    build_alive,
    datetime,
    io,
    json,
    last_run_file,
    launch,
    mo,
    os,
    read_json,
    status_refresh,
    timezone,
):
    status_refresh.value  # dependency: re-run on every refresh tick

    def _log_tail(path, lines=40):
        try:
            with open(path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 65536))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _archive(run_dir):
        # Logs, report and manifest only: build trees and the checkpoint stay in the work folder.
        buffer = io.BytesIO()
        with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
            for file in sorted(run_dir.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(run_dir))
        return buffer.getvalue()

    _items = [mo.md(launch["notice"])] if launch["notice"] else []
    _last = read_json(last_run_file)
    if _last is None:
        _items.append(mo.md("No run yet. Choose stages and click **Run ActQuant**."))
    else:
        _run_dir = Path(_last["run_dir"])
        _process = launch["process"]
        _running = (_process.poll() is None if _process is not None and _process.pid == _last["pid"]
                    else build_alive(_last["pid"]))
        _report = read_json(_run_dir / "report.json")
        _manifest = read_json(_run_dir / "manifest.json") or {}
        # Built lazily, only when clicked: the snapshot is current at download time.
        _download = mo.download(
            data=lambda run_dir=_run_dir: _archive(run_dir),
            filename=f"eaq-actquant-build-{_last['run_id']}.zip",
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
            _items.append(mo.md(
                f"**Running** run `{_last['run_id']}`: stage `{_stage}`, started {_elapsed_text} ago."
                f"\n\n```text\n{_log_tail(_run_dir / 'notebook.log')}\n```"
            ))
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
                    _note = _entry.get("error") or _entry.get("reason") or _entry.get("note") or ""
                    _entry_status = "interrupted" if _entry["status"] == "running" else _entry["status"]
                    _rows.append(f"| `{_name}` | **{_entry_status}** | {_entry.get('seconds', '')} s | "
                                 f"{str(_note).replace('|', '/')} |")
                _items.append(mo.md(
                    f"## Result: `{_status}`\n\n"
                    f"Run folder: `{_run_dir}`\n\n" + "\n".join(_rows)
                ))
                if _status == "interrupted":
                    _items.append(mo.md(
                        "The run was killed without finishing; the table shows the stage it was in. "
                        "Rerun it; stages that finished are reused if the work folder survived."
                        "\n\n```text\n" + _log_tail(_run_dir / "notebook.log") + "\n```"
                    ))

                _infer = _report.get("stages", {}).get("infer", {})
                _cuda = _infer.get("cli_cuda")
                if _cuda:
                    _summary = [
                        "### Inference",
                        "",
                        f"- CLI on `{_cuda.get('backend')}`: exit `{_cuda.get('returncode')}`, "
                        f"{_cuda.get('inference_ms')} ms inference (includes first-call warm-up)",
                        f"- Actions: {_cuda.get('action_dim')} dim × {_cuda.get('action_horizon')} horizon; "
                        f"stats `{_cuda.get('stats')}`",
                        f"- First timestep: `{_cuda.get('first_actions')}`",
                    ]
                    _binding = _infer.get("binding") or {}
                    if _binding.get("status") == "passed":
                        _summary.append(
                            f"- `pi05.so` binding: load {_binding['load_seconds']:.1f} s, runs "
                            f"{', '.join(f'{t:.2f}' for t in _binding['run_seconds'])} s, "
                            f"repeat max diff `{_binding['repeat_max_abs_diff']:.2e}`, "
                            f"max diff vs CLI `{_infer.get('binding_vs_cli_max_abs_diff')}`"
                        )
                    elif _binding:
                        _summary.append(f"- `pi05.so` binding: {_binding.get('status')}")
                    if "cpu_vs_cuda_max_abs_diff" in _infer:
                        _summary.append(f"- CPU vs CUDA max diff (first timestep): "
                                        f"`{_infer['cpu_vs_cuda_max_abs_diff']:.4f}`")
                    _items.append(mo.md("\n".join(_summary)))

                _deviations = "\n".join(f"- {item}" for item in _report.get("deviations", [])) or "- none"
                _items.append(mo.md(f"### Deviations from ActQuant's documented setup\n\n{_deviations}"))

                _failed = _report.get("failed_stage")
                if _failed:
                    _tail = _report["stages"][_failed].get("tail") or _log_tail(_run_dir / "notebook.log")
                    _items.append(mo.md(f"### `{_failed}` output tail\n\n```text\n{_tail}\n```"))
                _items.append(mo.accordion({
                    "report.json": mo.md("```json\n" + json.dumps(_report, indent=2) + "\n```")
                }))
            _items.append(_download)
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
