from __future__ import annotations

import shlex


LLAMA_DIR = "/content/llama.cpp"
SERVER_PID = "/tmp/collama-llama-server.pid"
SERVER_LOG = "/tmp/collama-llama-server.log"


def q(value: str | int | None) -> str:
    return shlex.quote("" if value is None else str(value))


def install_script(*, ref: str = "master", jobs: int | None = None) -> str:
    jobs_expr = str(jobs) if jobs else "$(nproc)"
    return f"""#!/usr/bin/env bash
set -euxo pipefail

step() {{
  printf '\\n[collama:install] %s\\n' "$*"
}}

export DEBIAN_FRONTEND=noninteractive

step "Installing build dependencies"
apt-get update || true
apt-get install -y build-essential ca-certificates cmake curl git libcurl4-openssl-dev pkg-config python3 python3-pip python3-venv

step "Checking GPU and CUDA toolchain"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi is not available; CUDA build may fail." >&2
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is not available. Use a GPU Colab runtime with CUDA before running install." >&2
  exit 1
fi

nvcc --version

step "Preparing llama.cpp source at {LLAMA_DIR}"
if [ ! -d {q(LLAMA_DIR)}/.git ]; then
  git clone https://github.com/ggml-org/llama.cpp {q(LLAMA_DIR)}
fi

cd {q(LLAMA_DIR)}
git fetch --depth 1 origin {q(ref)}
git checkout FETCH_HEAD

step "Configuring llama.cpp CUDA build"
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON

step "Building llama-server, llama-cli, and llama-bench"
cmake --build build --config Release -j {jobs_expr} --target llama-server llama-cli llama-bench

step "Installed llama-server version"
build/bin/llama-server --version || true
"""


def tailscale_script(
    *,
    authkey: str | None,
    hostname: str,
    login_server: str | None = None,
    advertise_tags: str | None = None,
) -> str:
    if not authkey:
        return f"""#!/usr/bin/env bash
set -euo pipefail

echo 'error: TAILSCALE_AUTHKEY was not provided.' >&2
echo 'Generate an auth key at https://login.tailscale.com/admin/settings/keys' >&2
echo 'Then run: TAILSCALE_AUTHKEY=tskey-auth-... collama tailscale' >&2
echo 'Or use: collama tailscale --authkey tskey-auth-...' >&2
exit 1
"""

    up_args = ["--hostname", hostname]
    up_args.extend(["--authkey", authkey])
    if login_server:
        up_args.extend(["--login-server", login_server])
    if advertise_tags:
        up_args.extend(["--advertise-tags", advertise_tags])
    up_line = "tailscale up " + " ".join(q(arg) for arg in up_args)
    return f"""#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl iproute2

if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
  systemctl enable --now tailscaled || systemctl restart tailscaled
else
  if ! pgrep -x tailscaled >/dev/null 2>&1; then
    nohup tailscaled --tun=userspace-networking --socks5-server=localhost:1055 > /tmp/tailscaled.log 2>&1 &
  fi
fi

{up_line}
tailscale status
tailscale ip -4
"""


def serve_script(
    *,
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    ctx_size: int | None = None,
    gpu_layers: int | None = None,
    extra_args: list[str] | None = None,
) -> str:
    args = [
        f"{LLAMA_DIR}/build/bin/llama-server",
        "--hf-repo",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if ctx_size:
        args.extend(["--ctx-size", str(ctx_size)])
    if gpu_layers is not None:
        args.extend(["--n-gpu-layers", str(gpu_layers)])
    if extra_args:
        args.extend(extra_args)
    cmd = " ".join(q(arg) for arg in args)
    return f"""#!/usr/bin/env bash
set -euo pipefail

if [ ! -x {q(LLAMA_DIR)}/build/bin/llama-server ]; then
  echo "llama-server is missing. Run install first." >&2
  exit 1
fi

if [ -f {q(SERVER_PID)} ] && kill -0 "$(cat {q(SERVER_PID)})" >/dev/null 2>&1; then
  echo "llama-server is already running with PID $(cat {q(SERVER_PID)})." >&2
  exit 1
fi

nohup {cmd} > {q(SERVER_LOG)} 2>&1 &
pid=$!
echo "$pid" > {q(SERVER_PID)}
echo "Started llama-server PID $pid"
echo "Log: {SERVER_LOG}"
sleep 2
if ! kill -0 "$pid" >/dev/null 2>&1; then
  echo "llama-server exited early:" >&2
  tail -n 80 {q(SERVER_LOG)} >&2 || true
  exit 1
fi
curl -fsS "http://127.0.0.1:{port}/health" || true
"""


def stop_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

if [ ! -f {q(SERVER_PID)} ]; then
  echo "No llama-server PID file found at {SERVER_PID}." >&2
  exit 1
fi

pid="$(cat {q(SERVER_PID)})"
if ! kill -0 "$pid" >/dev/null 2>&1; then
  rm -f {q(SERVER_PID)}
  echo "No running llama-server process found for PID $pid." >&2
  exit 1
fi

kill -TERM "$pid"
for _ in $(seq 1 20); do
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    rm -f {q(SERVER_PID)}
    echo "Stopped llama-server PID $pid"
    exit 0
  fi
  sleep 1
done

echo "llama-server PID $pid did not stop after 20 seconds." >&2
exit 1
"""


def status_script() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

echo "=== llama-server executable check ==="
if [ -x {q(LLAMA_DIR)}/build/bin/llama-server ]; then
  echo "llama-server is executable at {LLAMA_DIR}/build/bin/llama-server"
  {q(LLAMA_DIR)}/build/bin/llama-server --version 2>/dev/null || true
else
  echo "llama-server is NOT found or not executable at {LLAMA_DIR}/build/bin/llama-server"
fi

echo
echo "=== llama-server running check ==="
if [ -f {q(SERVER_PID)} ]; then
  pid="$(cat {q(SERVER_PID)})"
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "llama-server is running with PID $pid (from PID file)"
  else
    echo "llama-server PID file exists (PID $pid) but process is NOT running"
  fi
else
  echo "No llama-server PID file at {SERVER_PID}"
fi

# Also check for any llama-server processes not managed by PID file
running_pids=$(pgrep -f 'llama-server' 2>/dev/null || true)
if [ -n "$running_pids" ]; then
  echo "Found running llama-server process(es): $running_pids"
  ps -p $running_pids -o pid,ppid,cmd 2>/dev/null || true
else
  echo "No llama-server processes found"
fi
"""


def host_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

if ! command -v tailscale >/dev/null 2>&1; then
  echo "tailscale is not installed on the Colab runtime." >&2
  exit 1
fi

echo "Tailscale IPv4:"
tailscale ip -4
echo
echo "Tailscale status:"
tailscale status
"""


def benchy_script(
    *,
    base_url: str,
    model: str,
    pp: int,
    tg: int,
    runs: int,
    concurrency: int,
    extra_args: list[str] | None = None,
) -> str:
    args = [
        "uvx",
        "--from",
        "git+https://github.com/eugr/llama-benchy",
        "llama-benchy",
        "--base-url",
        base_url,
        "--model",
        model,
        "--pp",
        str(pp),
        "--tg",
        str(tg),
        "--runs",
        str(runs),
        "--concurrency",
        str(concurrency),
    ]
    if extra_args:
        args.extend(extra_args)
    cmd = " ".join(q(arg) for arg in args)
    return f"""#!/usr/bin/env bash
set -euo pipefail

if ! command -v uvx >/dev/null 2>&1; then
  python3 -m pip install -U uv
fi

{cmd}
"""
