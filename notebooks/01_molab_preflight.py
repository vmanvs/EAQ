import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import importlib
    import importlib.metadata as importlib_metadata
    import io
    import json
    import os
    import platform
    import shutil
    import subprocess
    import sys
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path
    from zipfile import ZIP_DEFLATED, ZipFile

    import marimo as mo

    return (
        Path,
        datetime,
        importlib,
        importlib_metadata,
        io,
        json,
        mo,
        os,
        platform,
        shutil,
        subprocess,
        sys,
        tempfile,
        timezone,
        ZIP_DEFLATED,
        ZipFile,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # EAQ — Molab GPU preflight

        This bounded infrastructure check inspects the current runtime, tests a
        small FP16/BF16 attention calculation, renders a simple MuJoCo scene, and
        checks access to pinned model configuration metadata. It does **not** load
        LaWAM, download full checkpoints or datasets, or run LIBERO.

        The runtime and package inventory below do not install packages or fetch
        data. Click **Run bounded preflight** to run the repository script. That
        script may fetch small pinned configuration files from Hugging Face; it
        does not fetch model weights. Any `HF_TOKEN` provided by Molab Secrets or
        the notebook `.env` is inherited by the child process and is never shown.

        If small preflight dependencies are missing, install only the needed
        entries from `requirements/preflight.txt` with Molab's package manager.
        Inspect Molab's preinstalled PyTorch and CUDA versions first; do not
        replace PyTorch blindly or install LaWAM's full pinned requirements.

        After the run, export the run folder before the Molab session ends. Scratch
        files can be removed when the session is torn down.
        """
    )
    return


@app.cell
def _(Path, mo, subprocess):
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

        visited = set()
        for start in starts:
            for candidate in (start, *start.parents):
                candidate = candidate.resolve()
                if candidate in visited:
                    continue
                visited.add(candidate)
                script = candidate / "scripts" / "cloud_preflight.py"
                requirements = candidate / "requirements" / "preflight.txt"
                if script.is_file() and requirements.is_file():
                    return candidate, script, requirements
        return None, None, None

    repository_root, preflight_script, requirements_file = locate_repository()
    eaq_commit = "unknown"
    if repository_root is not None:
        try:
            git_root_result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(repository_root),
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            git_root = Path(git_root_result.stdout.strip()).resolve()
            if git_root_result.returncode == 0 and git_root == repository_root.resolve():
                git_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(repository_root),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                )
                resolved_commit = git_result.stdout.strip()
                if git_result.returncode == 0 and resolved_commit:
                    eaq_commit = resolved_commit
        except (OSError, subprocess.SubprocessError):
            pass

    return (
        eaq_commit,
        preflight_script,
        repository_root,
        requirements_file,
    )


@app.cell
def _(
    Path,
    eaq_commit,
    importlib,
    importlib_metadata,
    mo,
    platform,
    repository_root,
    requirements_file,
    shutil,
    subprocess,
    tempfile,
):
    package_names = [
        "marimo",
        "torch",
        "torchvision",
        "mujoco",
        "Pillow",
        "psutil",
        "huggingface-hub",
    ]
    if requirements_file is not None:
        for line in requirements_file.read_text(encoding="utf-8").splitlines():
            requirement = line.partition("#")[0].strip()
            if requirement:
                name = requirement.split("==", maxsplit=1)[0].strip()
                if name:
                    package_names.append(name)

    packages = {}
    for name in dict.fromkeys(package_names):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "not installed"
        except Exception as exc:
            packages[name] = f"unavailable ({type(exc).__name__})"

    scratch_parent = (
        repository_root / "artifacts" / "molab-preflight"
        if repository_root is not None
        else Path(tempfile.gettempdir()) / "eaq-molab-preflight"
    )
    disk_anchor = repository_root or Path(tempfile.gettempdir())
    try:
        disk = shutil.disk_usage(disk_anchor)
        scratch_state = (
            f"{disk.free / 2**30:.2f} GiB free at `{disk_anchor}`; "
            f"run folders will be placed under `{scratch_parent}`"
        )
    except OSError as exc:
        scratch_state = f"unavailable ({type(exc).__name__})"

    memory_state = "unavailable (psutil is not installed)"
    try:
        psutil = importlib.import_module("psutil")
        memory = psutil.virtual_memory()
        memory_state = (
            f"{memory.available / 2**30:.2f} GiB available / "
            f"{memory.total / 2**30:.2f} GiB total"
        )
    except Exception as exc:
        memory_state = f"unavailable ({type(exc).__name__})"

    gpu_state = "PyTorch unavailable"
    try:
        torch = importlib.import_module("torch")
        cuda_available = torch.cuda.is_available()
        gpu_lines = [
            f"PyTorch `{torch.__version__}`; CUDA runtime `{torch.version.cuda}`; "
            f"CUDA available: `{cuda_available}`"
        ]
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                capability = torch.cuda.get_device_capability(index)
                gpu_lines.append(
                    f"GPU {index}: {properties.name}, "
                    f"{properties.total_memory / 2**30:.2f} GiB VRAM, "
                    f"compute capability {capability[0]}.{capability[1]}"
                )
        gpu_state = "  \n".join(gpu_lines)
    except Exception as exc:
        gpu_state = f"unavailable ({type(exc).__name__})"

    driver_state = "not available (`nvidia-smi` not found or failed)"
    try:
        driver_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if driver_result.returncode == 0 and driver_result.stdout.strip():
            versions = sorted(set(driver_result.stdout.strip().splitlines()))
            driver_state = ", ".join(f"`{version.strip()}`" for version in versions)
    except (OSError, subprocess.SubprocessError):
        pass

    repository_state = (
        f"`{repository_root}`" if repository_root is not None else "not found"
    )
    requirements_state = (
        f"`{requirements_file}`" if requirements_file is not None else "not found"
    )
    package_rows = "\n".join(
        f"| `{name}` | `{version}` |" for name, version in packages.items()
    )
    mo.md(
        f"""
        ## Runtime and repository

        - Python: `{platform.python_version()}` on `{platform.platform()}`
        - Current directory: `{Path.cwd()}`
        - EAQ repository root: {repository_state}
        - Preflight script: `{repository_root / 'scripts' / 'cloud_preflight.py' if repository_root else 'not found'}`
        - Requirements file: {requirements_state}
        - EAQ Git commit: `{eaq_commit if repository_root is not None else 'unknown'}`
        - Scratch: {scratch_state}
        - RAM: {memory_state}
        - GPU: {gpu_state}
        - NVIDIA driver: {driver_state}

        ## Installed package versions

        These are inspection results only; this notebook does not install packages.

        | Package | Installed version |
        | --- | --- |
        {package_rows}
        """
    )
    return


@app.cell
def _(mo):
    run_preflight = mo.ui.run_button(
        label="Run bounded preflight",
        kind="primary",
        tooltip="Runs scripts/cloud_preflight.py in a new scratch folder.",
    )
    run_preflight
    return (run_preflight,)


@app.cell
def _(
    Path,
    datetime,
    eaq_commit,
    json,
    mo,
    os,
    preflight_script,
    repository_root,
    run_preflight,
    subprocess,
    sys,
    timezone,
):
    run_result = None
    if run_preflight.value:
        if repository_root is None or preflight_script is None:
            run_result = {
                "error": (
                    "Could not locate scripts/cloud_preflight.py and "
                    "requirements/preflight.txt in the mirrored repository."
                )
            }
        else:
            run_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                + "_"
                + os.urandom(4).hex()
            )
            run_dir = repository_root / "artifacts" / "molab-preflight" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            started_utc = datetime.now(timezone.utc).isoformat()
            manifest = {
                "schema_version": 1,
                "scope": "bounded infrastructure preflight; no policy inference",
                "run_id": run_id,
                "started_utc": started_utc,
                "eaq_commit": eaq_commit,
                "python": sys.version.split()[0],
                "preflight_script": str(preflight_script),
                "requirements_file": str(
                    repository_root / "requirements" / "preflight.txt"
                ),
            }

            command = [
                sys.executable,
                str(preflight_script),
                "--output",
                str(run_dir),
            ]
            process = None
            execution_error = None
            child_env = os.environ.copy()
            child_env["MUJOCO_GL"] = "egl"
            try:
                # Inherit Molab's environment so HF_TOKEN remains in the child
                # environment without being read or printed by this notebook.
                process = subprocess.run(
                    command,
                    cwd=str(repository_root),
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=15 * 60,
                )
                (run_dir / "preflight.log").write_text(
                    process.stdout or "", encoding="utf-8"
                )
            except subprocess.TimeoutExpired as exc:
                partial_output = exc.stdout or ""
                if isinstance(partial_output, bytes):
                    partial_output = partial_output.decode("utf-8", errors="replace")
                execution_error = (
                    "Preflight timed out after 15 minutes; the process was stopped. "
                    "Any partial report.json has been retained."
                )
                (run_dir / "preflight.log").write_text(
                    partial_output + "\n" + execution_error + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                execution_error = f"{type(exc).__name__}: {exc}"
                (run_dir / "preflight.log").write_text(
                    execution_error + "\n", encoding="utf-8"
                )

            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["exit_code"] = (
                process.returncode if process is not None else None
            )
            if execution_error:
                manifest["execution_error"] = execution_error
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

            report = None
            report_path = run_dir / "report.json"
            report_error = None
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    report_error = f"{type(exc).__name__}: {exc}"

            run_result = {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "return_code": process.returncode if process is not None else None,
                "execution_error": execution_error,
                "report_error": report_error,
                "report": report,
                "log": (run_dir / "preflight.log").read_text(encoding="utf-8"),
            }

    return (run_result,)


@app.cell
def _(io, json, mo, Path, run_result, ZIP_DEFLATED, ZipFile):
    output_items = []
    if run_result is None:
        output_items.append(
            mo.md("Click **Run bounded preflight** above to start the checks.")
        )
    elif "error" in run_result:
        output_items.append(
            mo.md(f"**Preflight could not start:** {run_result['error']}")
        )
    else:
        output_items.append(mo.md(
            f"""
            ## Preflight results

            Run folder: `{run_result['run_dir']}`  
            Exit code: `{run_result['return_code']}`

            Export this run folder from Molab before the session ends. It contains
            the report, manifest, package inventory, log, and rendered frame when
            available.
            """
        ))

        if run_result["execution_error"]:
            output_items.append(
                mo.md(f"**Execution error:** `{run_result['execution_error']}`")
            )
        if run_result["report_error"]:
            output_items.append(
                mo.md(f"**Report read error:** `{run_result['report_error']}`")
            )

        report = run_result["report"]
        if report is not None:
            failures = [
                (name, details)
                for name, details in report.get("checks", {}).items()
                if details.get("status") == "failed"
            ]
            if failures:
                failure_details = "\n".join(
                    f"- **{name}:** {details.get('error', 'failed without details')}"
                    for name, details in failures
                )
                output_items.append(mo.md(f"### Failed checks\n\n{failure_details}"))
            else:
                output_items.append(mo.md("No failed checks are listed in the report."))

            output_items.append(
                mo.md(
                    "### `report.json`\n\n```json\n"
                    + json.dumps(report, indent=2)
                    + "\n```"
                )
            )
        else:
            log_tail = run_result["log"][-6000:] or "No script output was captured."
            output_items.append(
                mo.md(f"### Failure details\n\n```text\n{log_tail}\n```")
            )

        render_path = Path(run_result["run_dir"]) / "render.png"
        if render_path.is_file():
            output_items.append(
                mo.image(
                    src=str(render_path),
                    alt="Simple MuJoCo headless-rendering preflight frame",
                )
            )

        run_dir = Path(run_result["run_dir"])
        archive_buffer = io.BytesIO()
        with ZipFile(archive_buffer, mode="w", compression=ZIP_DEFLATED) as archive:
            for artifact in sorted(run_dir.rglob("*")):
                if artifact.is_file():
                    archive.write(artifact, artifact.relative_to(run_dir))
        output_items.append(
            mo.download(
                data=archive_buffer.getvalue(),
                filename=f"eaq-molab-preflight-{run_result['run_id']}.zip",
                mimetype="application/zip",
                label="Download preflight artifacts (.zip)",
            )
        )
    mo.vstack(output_items)
    return


if __name__ == "__main__":
    app.run()
