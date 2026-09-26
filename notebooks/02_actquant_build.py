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
    import time
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
        time,
        timezone,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # EAQ — ActQuant build on Molab

        Builds [ActQuant](https://github.com/arashakb/ActQuant)'s Pi 0.5 runtime
        at its pinned commit for the attached GPU, downloads the released 3-bit
        checkpoint (`ActQuant-Pi05-LIBERO-3bpw`, about 2.4 GB), and runs **one**
        inference on CUDA. It applies the recipe from the preflight notebook,
        with one change: PyTorch's pip CUDA toolkit mixes nvcc 13.3 with 13.0
        runtime headers, which CUB rejects, so the build uses its own pinned
        CUDA 13.0 toolkit. Device code is built for this GPU only, with
        unversioned library links and a toolkit RPATH.
        Every departure from ActQuant's documented setup is listed as a
        deviation in the report.

        The smoke test uses a synthetic image and a fixed prompt. It shows that
        the runtime executes on this GPU and produces finite actions; it says
        nothing about task success. No LIBERO rollout is run here.

        **Before running**

        1. Attach the GPU and open the **Server** preview.
        2. From the **Packages** panel install `cmake`, `ninja`,
           `huggingface_hub`, and optionally `pybind11` (for the `pi05.so`
           binding), pinned in `requirements/actquant-build.txt`.
        3. Do not install CUDA packages. The `toolkit` stage installs a pinned,
           version-consistent CUDA 13.0 toolkit (about 0.5 GB of wheels) into
           the work folder; PyTorch's environment is not touched.

        No token is needed: ActQuant and the checkpoint are public.

        Sources, build trees and the checkpoint are kept in a work folder for
        the session, so you can rerun single stages and the build continues
        where it stopped. The first build compiles ggml's CUDA kernels and can
        take a long time. The build runs as a separate process, so it keeps
        going if the notebook disconnects: reopen the notebook (or rerun the
        cell below the button) to reattach to it and see its result. Download
        the run artifacts before the session ends.
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
    return (
        build_requirements,
        build_script,
        eaq_commit,
        fetch_error,
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
    mo,
    os,
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

    _existing = [name for name in ("ActQuant", "checkpoints") if (work_dir / name).exists()]
    _existing += sorted(path.name for path in work_dir.glob("build-*")) if work_dir.exists() else []

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
        f"- GPU: `{_gpu}`",
        f"- Work folder: `{work_dir}`; disk {_disk_text}; "
        + (f"already contains {', '.join(f'`{item}`' for item in _existing)}" if _existing else "empty"),
        "",
        "| Package | Installed | Pinned |",
        "| --- | --- | --- |",
        *(f"| `{name}` | `{_version(name)}` | `{_pins.get(name, '—')}` |"
          for name in ("torch", "cmake", "ninja", "huggingface_hub", "pybind11")),
    ]
    mo.md("\n".join(_lines))
    return (work_dir,)


@app.cell
def _(mo):
    stage_picker = mo.ui.multiselect(
        options=["toolkit", "source", "configure", "build", "download", "infer"],
        value=["toolkit", "source", "configure", "build", "download", "infer"],
        label="Stages (always run in this order)",
    )
    no_vmm = mo.ui.checkbox(
        label="Build without CUDA virtual memory management (GGML_CUDA_NO_VMM; separate build tree)"
    )
    cpu_check = mo.ui.checkbox(label="Also run the CLI on CPU and compare with CUDA (slower)")
    build_jobs = mo.ui.dropdown(
        options=["auto", "4", "8", "12", "16", "20"],
        value="auto",
        label="Parallel build jobs (auto: up to 8, limited by memory)",
    )
    run_build = mo.ui.run_button(
        label="Run ActQuant build",
        kind="success",
        tooltip="Runs scripts/actquant_build.py with the selected stages in a new run folder.",
    )
    mo.vstack([stage_picker, no_vmm, cpu_check, build_jobs, run_build])
    return build_jobs, cpu_check, no_vmm, run_build, stage_picker


@app.cell
def _(
    Path,
    build_jobs,
    build_script,
    cpu_check,
    datetime,
    eaq_commit,
    fetch_error,
    json,
    mo,
    no_vmm,
    os,
    repository_source,
    run_build,
    stage_picker,
    subprocess,
    sys,
    time,
    timezone,
    toolkit_requirements,
    work_dir,
):
    # The script runs detached, writing to a log file: a disconnected notebook no longer kills
    # the build through a broken output pipe. last_run.json lets a new session reattach.
    _last_run_file = work_dir / "last_run.json"

    def _read_json(path):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _alive(pid):
        # A finished child of this kernel is a zombie with an empty cmdline, so it counts as ended.
        try:
            return b"actquant_build" in Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False

    def _log_tail(path, lines=40):
        try:
            with open(path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 65536))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    build_result = None
    _notice = None
    _process = None
    _last = _read_json(_last_run_file)
    _last_alive = bool(_last) and _alive(_last["pid"])
    if run_build.value:
        if build_script is None:
            build_result = {"error": f"Build script not available: {fetch_error}"}
        elif not stage_picker.value:
            build_result = {"error": "Select at least one stage."}
        elif _last_alive:
            _notice = (f"Run `{_last['run_id']}` is still running, so no new run was started. "
                       "Showing that run instead.")
        else:
            _run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + os.urandom(3).hex()
            _run_dir = work_dir / "runs" / _run_id
            _run_dir.mkdir(parents=True, exist_ok=False)
            _command = [sys.executable, str(build_script), "--output", str(_run_dir),
                        "--work-dir", str(work_dir), "--toolkit-requirements", str(toolkit_requirements),
                        "--stages", *stage_picker.value]
            if no_vmm.value:
                _command.append("--no-vmm")
            if cpu_check.value:
                _command.append("--cpu-check")
            if build_jobs.value != "auto":
                _command += ["--jobs", build_jobs.value]
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
                _process = subprocess.Popen(_command, stdin=subprocess.DEVNULL, stdout=_log_handle,
                                            stderr=subprocess.STDOUT, env=_env, start_new_session=True)
            _last = {"run_id": _run_id, "run_dir": str(_run_dir), "pid": _process.pid}
            _last_run_file.write_text(json.dumps(_last) + "\n", encoding="utf-8")
    elif _last is not None:
        _notice = (f"Reattached to run `{_last['run_id']}`, which is still running."
                   if _last_alive else f"Showing the last run, `{_last['run_id']}`.")

    if build_result is None and _last is not None:
        _run_dir = Path(_last["run_dir"])
        _started = time.monotonic()
        while (_process.poll() is None) if _process is not None else _alive(_last["pid"]):
            _elapsed = int(time.monotonic() - _started)
            mo.output.replace(mo.md(
                (f"{_notice}\n\n" if _notice else "")
                + f"**Running** (watched for {_elapsed // 60} min {_elapsed % 60} s) — `{_run_dir}`\n\n"
                "```text\n" + _log_tail(_run_dir / "notebook.log") + "\n```"
            ))
            time.sleep(3)
        mo.output.clear()

        _code_text = (_run_dir / "exit_code").read_text(encoding="utf-8").strip() \
            if (_run_dir / "exit_code").is_file() else ""
        _exit_code = _process.returncode if _process is not None else (
            int(_code_text) if _code_text.lstrip("-").isdigit() else None)
        _manifest = _read_json(_run_dir / "manifest.json") or {}
        if "finished_utc" not in _manifest:
            _manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _manifest["exit_code"] = _exit_code
            (_run_dir / "manifest.json").write_text(json.dumps(_manifest, indent=2) + "\n", encoding="utf-8")
        build_result = {"run_id": _last["run_id"], "run_dir": _run_dir, "exit_code": _exit_code,
                        "report": _read_json(_run_dir / "report.json"), "notice": _notice,
                        "log_tail": _log_tail(_run_dir / "notebook.log")}
    return (build_result,)


@app.cell
def _(ZIP_DEFLATED, ZipFile, build_result, io, json, mo):
    _items = []
    if build_result is None:
        _items.append(mo.md("Choose stages and click **Run ActQuant build**."))
    elif "error" in build_result:
        _items.append(mo.md(f"**Could not start:** {build_result['error']}"))
    else:
        if build_result.get("notice"):
            _items.append(mo.md(build_result["notice"]))
        _report = build_result["report"]
        if _report is None:
            _items.append(mo.md(
                f"**No report was written** (exit code `{build_result['exit_code']}`).\n\n"
                "```text\n" + build_result["log_tail"] + "\n```"
            ))
        else:
            # A report still saying "running" means the process was killed mid-stage.
            _status = "interrupted" if _report.get("status") == "running" else _report.get("status")
            _rows = ["| Stage | Status | Time | Note |", "| --- | --- | --- | --- |"]
            for _stage, _entry in _report.get("stages", {}).items():
                _note = _entry.get("error") or _entry.get("reason") or _entry.get("note") or ""
                _entry_status = "interrupted" if _entry["status"] == "running" else _entry["status"]
                _rows.append(f"| `{_stage}` | **{_entry_status}** | {_entry.get('seconds', '')} s | "
                             f"{str(_note).replace('|', '/')} |")
            _items.append(mo.md(
                f"## Result: `{_status}`\n\n"
                f"Run folder: `{build_result['run_dir']}`\n\n" + "\n".join(_rows)
            ))
            if _status == "interrupted":
                _items.append(mo.md(
                    "The run was killed without finishing; the table shows the stage it was in. "
                    "`memory.log` in the ZIP shows whether memory ran low. Rerun to continue: "
                    "the build resumes where it stopped if the work folder survived.\n\n"
                    "```text\n" + build_result["log_tail"] + "\n```"
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
                _entry = _report["stages"][_failed]
                _tail = _entry.get("tail") or build_result["log_tail"]
                _items.append(mo.md(f"### `{_failed}` output tail\n\n```text\n{_tail}\n```"))
            _items.append(mo.accordion({
                "report.json": mo.md("```json\n" + json.dumps(_report, indent=2) + "\n```")
            }))

        # Logs, report and manifest only: build trees and the checkpoint stay in the work folder.
        _buffer = io.BytesIO()
        with ZipFile(_buffer, mode="w", compression=ZIP_DEFLATED) as _archive:
            for _file in sorted(build_result["run_dir"].rglob("*")):
                if _file.is_file():
                    _archive.write(_file, _file.relative_to(build_result["run_dir"]))
        _items.append(mo.download(
            data=_buffer.getvalue(),
            filename=f"eaq-actquant-build-{build_result['run_id']}.zip",
            mimetype="application/zip",
            label="Download run artifacts (.zip)",
        ))
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
