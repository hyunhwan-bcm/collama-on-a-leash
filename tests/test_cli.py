from __future__ import annotations

import subprocess
from unittest.mock import patch

from colla import cli


class Calls:
    def __init__(self) -> None:
        self.items: list[tuple[list[str], str | None]] = []

    def run(self, cmd, input=None, text=None, stdout=None, stderr=None):
        self.items.append((list(cmd), input))
        if cmd[-3:] == ["status", "-s", "collama"]:
            return subprocess.CompletedProcess(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)


def run_cli(argv: list[str], calls: Calls) -> int:
    with patch("shutil.which", return_value="/bin/colab"), patch("subprocess.run", calls.run):
        return cli.main(argv)


def payload(calls: Calls) -> str:
    assert calls.items[-1][1]
    return calls.items[-1][1] or ""


def test_install_creates_gpu_session_and_builds_cuda_llama_cpp() -> None:
    calls = Calls()
    assert run_cli(["install", "--ref", "master", "--jobs", "2"], calls) == 0
    assert calls.items[0][0] == ["colab", "status", "-s", "collama"]
    assert calls.items[1][0] == ["colab", "new", "-s", "collama", "--gpu", "G4"]
    body = payload(calls)
    assert "git clone https://github.com/ggml-org/llama.cpp" in body
    assert "-DGGML_CUDA=ON" in body
    assert "--target llama-server llama-cli llama-bench" in body


def test_tailscale_uses_authkey_hostname_and_starts_daemon() -> None:
    calls = Calls()
    assert run_cli(["tailscale", "--authkey", "tskey-test", "--hostname", "colab-node"], calls) == 0
    body = payload(calls)
    assert "curl -fsSL https://tailscale.com/install.sh | sh" in body
    assert "tailscale up --hostname colab-node --authkey tskey-test" in body
    assert "tailscaled --tun=userspace-networking" in body


def test_tailscale_without_authkey_prints_manual_command() -> None:
    calls = Calls()
    assert run_cli(["tailscale", "--hostname", "manual-node"], calls) == 0
    body = payload(calls)
    assert "TAILSCALE_AUTHKEY was not provided" in body
    assert "tailscale up --hostname manual-node" in body


def test_healthcheck_prints_json() -> None:
    with patch("colla.cli.check_health", return_value={"base_url": "http://100.1.2.3:8000"}):
        assert cli.main(["healthcheck", "100.1.2.3"]) == 0


def test_serve_starts_llama_server_with_defaults_and_model() -> None:
    calls = Calls()
    assert run_cli(["serve", "unsloth/Qwen3-GGUF"], calls) == 0
    body = payload(calls)
    assert "llama-server" in body
    assert "--hf-repo unsloth/Qwen3-GGUF --host 0.0.0.0 --port 8000" in body
    assert "/tmp/collama-llama-server.pid" in body


def test_stop_does_not_create_session_and_errors_when_no_remote_pid() -> None:
    calls = Calls()
    assert run_cli(["stop"], calls) == 0
    assert calls.items[0][0] == ["colab", "exec", "-s", "collama"]
    body = payload(calls)
    assert "No llama-server PID file found" in body
    assert "exit 1" in body


def test_benchy_runs_llama_benchy_against_v1_endpoint() -> None:
    calls = Calls()
    assert run_cli(["benchy", "--address", "100.1.2.3", "--port", "8000", "--model", "repo/model"], calls) == 0
    body = payload(calls)
    assert "git+https://github.com/eugr/llama-benchy" in body
    assert "--base-url http://100.1.2.3:8000/v1 --model repo/model" in body


def test_host_prints_tailscale_ip_and_status_without_creating_session() -> None:
    calls = Calls()
    assert run_cli(["host"], calls) == 0
    assert calls.items[0][0] == ["colab", "exec", "-s", "collama"]
    body = payload(calls)
    assert "tailscale ip -4" in body
    assert "tailscale status" in body
