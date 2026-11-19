"""Install and run Nova's private, managed ComfyUI instance."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import webbrowser
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener, urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
COMFY_REV = "995456d0188ffb51032c28f41527cfa82d87d8a9"
ADAPTER_REV = "a0f451a5113cf9becb0847b92884cb10cbdec0ef"
BLOCK_BYTES = 4 * 1024 * 1024
LOCAL_HTTP = build_opener(ProxyHandler({}))
TORCH_VERSION = "2.10.0"
VISION_VERSION = "0.25.0"


@dataclass(frozen=True)
class Download:
    name: str
    url: str
    sha256: str
    size: int | None = None


COMFY = Download(
    f"ComfyUI-{COMFY_REV}.zip",
    f"https://codeload.github.com/Comfy-Org/ComfyUI/zip/{COMFY_REV}",
    "6b791de98dad072c19f678ab73d95e81b42bb5bdd5f5b8f0cf55c288c983f913",
)
ADAPTER = Download(
    f"ComfyUI_IPAdapter_plus-{ADAPTER_REV}.zip",
    f"https://codeload.github.com/cubiq/ComfyUI_IPAdapter_plus/zip/{ADAPTER_REV}",
    "c6c49c82aa65cb96b93bdf9f9b547f9c95310a2668a7a9aaa0285cccf4590347",
)
HF = "https://huggingface.co"
IP_ADAPTER = f"{HF}/h94/IP-Adapter/resolve/018e402774aeeddd60609b4ecdb7e298259dc729"
MODELS = (
    ("checkpoints", Download(
        "v1-5-pruned-emaonly.safetensors",
        f"{HF}/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/"
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14/v1-5-pruned-emaonly.safetensors",
        "6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa", 4265146304,
    )),
    ("clip_vision", Download(
        "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors", f"{IP_ADAPTER}/models/image_encoder/model.safetensors",
        "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030", 2528373448,
    )),
    ("ipadapter", Download(
        "ip-adapter-plus_sd15.safetensors", f"{IP_ADAPTER}/models/ip-adapter-plus_sd15.safetensors",
        "a1c250be40455cc61a43da1201ec3f1edaea71214865fb47f57927e06cbe4996", 98183288,
    )),
    ("controlnet", Download(
        "control_v11p_sd15_openpose_fp16.safetensors",
        f"{HF}/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/"
        "ab830a51c5c573a5b85bfdbaa3ae0ab7e1baf5f7/control_v11p_sd15_openpose_fp16.safetensors",
        "4003c1da17b0e4ba444e02140e1c0d83bb24b79e4dcfd613c3a554d38f0f89c7", 722601100,
    )),
)


def say(message):
    print(message, flush=True)


@contextmanager
def instance_lock(home):
    with (home / "launcher.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another Nova launcher is already running.") from exc
        yield


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def stamp(path, asset):
    stat = path.stat()
    return {"sha256": asset.sha256, "size": stat.st_size, "mtime": stat.st_mtime_ns}


def download(asset, directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / asset.name
    partial = target.with_suffix(target.suffix + ".part")
    marker = target.with_suffix(target.suffix + ".verified.json")
    if target.is_file():
        expected = stamp(target, asset)
        if ((asset.size is None or expected["size"] == asset.size)
                and (read_json(marker) == expected or checksum(target) == asset.sha256)):
            write_json(marker, expected)
            return target

    for attempt in range(1, 4):
        current = partial.stat().st_size if partial.is_file() else 0
        if asset.size is not None and current > asset.size:
            partial.unlink()
            current = 0
        try:
            if asset.size is None or current != asset.size:
                # A unique range query avoids reusing a cached full-response redirect.
                url = asset.url
                headers = {}
                if current and asset.size is not None:
                    url += f"?nova_resume={current}"
                    headers["Range"] = f"bytes={current}-"
                say(f"Downloading {asset.name}" + (f" ({current // 2**20} MiB saved)" if headers else ""))
                with urlopen(Request(url, headers=headers), timeout=60) as response:
                    append = response.status == 206 and bool(headers)
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {current}-"):
                        partial.unlink(missing_ok=True)
                        raise RuntimeError(f"Unexpected byte range for {asset.name}.")
                    written = current if append else 0
                    last_report = time.monotonic()
                    with partial.open("ab" if append else "wb") as output:
                        while chunk := response.read(BLOCK_BYTES):
                            output.write(chunk)
                            written += len(chunk)
                            if time.monotonic() - last_report >= 5:
                                total = f" / {asset.size // 2**20} MiB" if asset.size else ""
                                say(f"  {written // 2**20} MiB{total}")
                                last_report = time.monotonic()
            if asset.size is not None and partial.stat().st_size != asset.size:
                raise RuntimeError(f"Incomplete download for {asset.name}.")
            if checksum(partial) != asset.sha256:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"Checksum mismatch for {asset.name}.")
            partial.replace(target)
            write_json(marker, stamp(target, asset))
            return target
        except (OSError, RuntimeError, HTTPException) as exc:
            if isinstance(exc, HTTPError) and exc.code == 416:
                partial.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Could not download {asset.name}: {exc}") from exc
            say(f"Retrying {asset.name} ({exc})")
            time.sleep(attempt)


def extract_source(archive, destination):
    if destination.is_dir() and any(destination.iterdir()):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging, ZipFile(archive) as source:
        root = Path(staging)
        source.extractall(root)
        folders = list(root.iterdir())
        if len(folders) != 1 or not folders[0].is_dir():
            raise RuntimeError(f"Unexpected source archive layout in {archive.name}.")
        if destination.is_dir():
            destination.rmdir()
        folders[0].replace(destination)


def venv_python(path):
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(*args):
    subprocess.run(args, check=True)


def gpu_wheel():
    if platform.system() == "Darwin":
        return "native"
    if platform.machine().lower() not in {"amd64", "x86_64"} or not shutil.which("nvidia-smi"):
        return "cpu"
    try:
        status = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        driver = int(status.stdout.strip().splitlines()[0].split(".")[0])
        if driver >= 570:
            return "cu128"
        if driver >= (560 if os.name == "nt" else 550):
            return "cu126"
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return "cpu"


def environment(home, comfy, wheel):
    directory = home / "environments" / f"py{sys.version_info.major}.{sys.version_info.minor}-{wheel}"
    python = venv_python(directory)
    if not python.is_file():
        run(sys.executable, "-m", "venv", directory)
    comfy_requirements = comfy / "requirements.txt"
    nova_requirements = ROOT / "requirements.txt"
    marker = directory / "nova-install.json"
    expected = {
        "comfy": checksum(comfy_requirements), "nova": checksum(nova_requirements),
        "torch": TORCH_VERSION, "vision": VISION_VERSION, "wheel": wheel,
    }
    if read_json(marker) != expected:
        say(f"Installing PyTorch ({wheel}), ComfyUI and Nova dependencies…")
        pip = (python, "-m", "pip", "--disable-pip-version-check", "--no-cache-dir")
        index = () if wheel == "native" else ("--index-url", f"https://download.pytorch.org/whl/{wheel}")
        run(*pip, "install", *index, f"torch=={TORCH_VERSION}", f"torchvision=={VISION_VERSION}")
        run(*pip, "install", "-r", comfy_requirements, "-r", nova_requirements)
        write_json(marker, expected)
    return python


def prepare(home, force_cpu):
    downloads = home / "downloads"
    comfy = home / "engine" / f"ComfyUI-{COMFY_REV}"
    adapter = comfy / "custom_nodes" / "ComfyUI_IPAdapter_plus"
    if os.name == "nt" and platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("The managed PyTorch build requires 64-bit x86 Windows.")
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        raise RuntimeError("The managed PyTorch build requires an Apple Silicon Mac.")
    wheel = "cpu" if force_cpu and platform.system() != "Darwin" else gpu_wheel()
    missing = 0
    for folder, model in MODELS:
        target = comfy / "models" / folder / model.name
        partial = target.with_suffix(target.suffix + ".part")
        if not target.is_file() or target.stat().st_size != model.size:
            saved = partial.stat().st_size if partial.is_file() else 0
            missing += max(0, model.size - saved)
    dependencies_needed = not venv_python(
        home / "environments" / f"py{sys.version_info.major}.{sys.version_info.minor}-{wheel}"
    ).is_file()
    reserve = (10 if wheel.startswith("cu") else 4) * 2**30 if dependencies_needed else 0
    if shutil.disk_usage(home).free < missing + reserve:
        raise RuntimeError(f"More disk space is needed at {home}. Free about "
                           f"{(missing + reserve) / 2**30:.1f} GiB there to continue.")

    if not (comfy / "main.py").is_file():
        extract_source(download(COMFY, downloads), comfy)
    if not (adapter / "IPAdapterPlus.py").is_file():
        extract_source(download(ADAPTER, downloads), adapter)
    if not (comfy / "main.py").is_file() or not (adapter / "IPAdapterPlus.py").is_file():
        raise RuntimeError(f"The managed source tree is incomplete at {comfy}. "
                           "Rename that directory and rerun the launcher.")
    for folder, model in MODELS:
        download(model, comfy / "models" / folder)
    return comfy, environment(home, comfy, wheel), wheel


def comfy_device(python, wheel, force_cpu):
    if force_cpu or wheel == "cpu":
        return ["--cpu"]
    device = "mps" if wheel == "native" else "cuda"
    probe = subprocess.run(
        [python, "-c", f"import torch; x = torch.ones(1, device={device!r}); print((x + 1).item())"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if probe.returncode:
        say(f"{device.upper()} is unavailable to PyTorch. Falling back to CPU.")
        return ["--cpu"]
    return []


def local_port(preferred, exclude=()):
    # ComfyUI and Uvicorn bind independently. Prefer familiar ports, then an OS-assigned port.
    for port in (preferred, 0):
        for _ in range(10 if port == 0 else 1):
            with socket.socket() as listener:
                try:
                    listener.bind(("127.0.0.1", port))
                except OSError:
                    continue
                selected = listener.getsockname()[1]
                if selected not in exclude:
                    return selected
    raise RuntimeError("Could not find a free loopback port.")


@contextmanager
def service(command, cwd, log_path, env=None):
    with log_path.open("wb") as output:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, creationflags=flags)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def wait_for_service(process, url, log, valid):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"A service stopped during startup. See {log}.")
        try:
            with LOCAL_HTTP.open(url, timeout=3) as response:
                if valid(response):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError(f"A service did not become ready at {url}. See {log}.")


def engine_ready(response):
    return "system" in json.load(response)


def studio_ready(response):
    models = json.load(response)
    return all(models.get(name) for name in ("checkpoint", "adapter", "vision", "controlnet"))


def launch(home, comfy, python, wheel, options):
    comfy_port = local_port(8188)
    studio_port = local_port(8000, (comfy_port,))
    comfy_url, studio_url = (f"http://127.0.0.1:{port}" for port in (comfy_port, studio_port))
    logs = home / "logs"
    logs.mkdir(exist_ok=True)
    comfy_log, studio_log = logs / "comfyui.log", logs / "studio.log"
    private = home / "private"
    io_flags = []
    for name in ("input", "output", "temp"):
        folder = private / name
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        io_flags.extend([f"--{name}-directory", str(folder)])
    env = {**os.environ, "NOVA_COMFY_URL": comfy_url, "NOVA_STARTER_MODELS": "1"}
    env.setdefault("NOVA_DATA_DIR", str(home))
    say(f"Starting ComfyUI. Logs: {logs}")
    device_flags = comfy_device(python, wheel, options.cpu)
    with ExitStack() as stack:
        engine = stack.enter_context(service(
            [python, "-u", "main.py", "--listen", "127.0.0.1", "--port", str(comfy_port),
             "--disable-metadata", *io_flags, *device_flags], comfy, comfy_log,
        ))
        wait_for_service(engine, f"{comfy_url}/system_stats", comfy_log, engine_ready)
        studio = stack.enter_context(service(
            [python, "-m", "uvicorn", "nova.app:app", "--host", "127.0.0.1",
             "--port", str(studio_port)], ROOT, studio_log, env,
        ))
        wait_for_service(studio, f"{studio_url}/api/models", studio_log, studio_ready)
        say(f"Nova Stasis is ready at {studio_url}. Press Ctrl+C to stop both services.")
        if not options.no_browser:
            webbrowser.open(studio_url)
        while True:
            for process, log in ((engine, comfy_log), (studio, studio_log)):
                if process.poll() is not None:
                    raise RuntimeError(f"A service stopped. See {log}.")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true", help="Install without starting the studio")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    options = parser.parse_args()
    try:
        home = Path(os.environ.get("NOVA_LAUNCHER_HOME", "~/.nova-stasis")).expanduser().resolve()
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        with instance_lock(home):
            comfy, python, wheel = prepare(home, options.cpu)
            if options.setup_only:
                say(f"Nova Stasis is installed at {home}.")
            else:
                launch(home, comfy, python, wheel, options)
    except KeyboardInterrupt:
        say("Nova Stasis stopped.")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Nova Stasis: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
