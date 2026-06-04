from __future__ import annotations

import subprocess
from io import StringIO
from unittest.mock import patch

from colla import cli


class Calls:
    def __init__(self) -> None:
        self.items: list[tuple[list[str], str | None]] = []
        self.popen_inputs: list[str] = []
        self.last_popen: FakePopen | None = None

    def run(self, cmd, input=None, text=None, stdout=None, stderr=None, capture_output=None):
        self.items.append((list(cmd), input))
        if cmd[-3:] == ["status", "-s", "collama"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="[colab] Session 'collama' not found.\n", stderr="")
        if input and "__COLLAMA_REMOTE_EXIT_CODE__" in input:
            return subprocess.CompletedProcess(cmd, 0, stdout="__COLLAMA_REMOTE_EXIT_CODE__=0\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0)

    def popen(self, cmd, stdin=None, stdout=None, stderr=None, text=None, bufsize=None):
        self.items.append((list(cmd), None))
        self.last_popen = FakePopen(
            "[collama] remote script exit code: 0\n", returncode=0, inputs=self.popen_inputs
        )
        return self.last_popen


class FakePopen:
    def __init__(self, output: str, *, returncode: int, inputs: list[str]) -> None:
        self.stdin = CapturingStdin(inputs)
        self.stdout = StringIO(output)
        self._returncode = returncode
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:
        return self._returncode


class CapturingStdin(StringIO):
    def __init__(self, inputs: list[str]) -> None:
        super().__init__()
        self._inputs = inputs

    def close(self) -> None:
        self._inputs.append(self.getvalue())
        super().close()


def run_cli(argv: list[str], calls: Calls) -> int:
    with (
        patch("shutil.which", return_value="/bin/colab"),
        patch("subprocess.run", calls.run),
        patch("subprocess.Popen", calls.popen),
    ):
        return cli.main(argv)


def payload(calls: Calls) -> str:
    assert calls.items[-1][1]
    return calls.items[-1][1] or ""


def remote_script_payload(calls: Calls) -> str:
    for item_input in calls.popen_inputs:
        if "/tmp/collama-on-a-leash.sh" in item_input:
            return item_input
    for _, item_input in calls.items:
        if item_input and "/tmp/collama-on-a-leash.sh" in item_input:
            return item_input
    raise AssertionError("remote script payload was not executed")


def test_install_creates_gpu_session_and_builds_cuda_llama_cpp() -> None:
    calls = Calls()
    assert run_cli(["install", "--ref", "master", "--jobs", "2"], calls) == 0
    assert calls.items[0][0] == ["colab", "status", "-s", "collama"]
    assert calls.items[1][0] == ["colab", "new", "-s", "collama", "--gpu", "G4"]
    body = remote_script_payload(calls)
    assert "set -euxo pipefail" in body
    assert "[collama:install]" in body
    assert "Installing build dependencies" in body
    assert "Checking GPU and CUDA toolchain" in body
    assert "Building llama-server, llama-cli, and llama-bench" in body
    assert "git clone https://github.com/ggml-org/llama.cpp" in body
    assert "nvidia-cuda-toolkit" not in body
    assert "command -v nvcc" in body
    assert "-DGGML_CUDA=ON" in body
    assert "--target llama-server llama-cli llama-bench" in body
    assert "raise SystemExit" not in body
    assert "raise RuntimeError" not in body
    assert "subprocess.Popen" in body
    assert "stderr=subprocess.STDOUT" in body
    assert "flush=True" in body


def test_install_reuses_existing_session() -> None:
    calls = Calls()

    def run(cmd, input=None, text=None, capture_output=None):
        calls.items.append((list(cmd), input))
        if cmd[-3:] == ["status", "-s", "collama"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="Name: collama | Hardware: G4 | Status: IDLE\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0)

    with (
        patch("shutil.which", return_value="/bin/colab"),
        patch("subprocess.run", run),
        patch("subprocess.Popen", calls.popen),
    ):
        assert cli.main(["install"]) == 0

    assert calls.items[0][0] == ["colab", "status", "-s", "collama"]
    assert calls.items[1][0] == ["colab", "exec", "-s", "collama"]


def test_remote_script_failure_is_reported_locally_without_remote_traceback() -> None:
    calls = Calls()

    def run(cmd, input=None, text=None, stdout=None, stderr=None, capture_output=None):
        calls.items.append((list(cmd), input))
        if cmd[-3:] == ["status", "-s", "collama"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="Name: collama | Hardware: G4 | Status: IDLE\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0)

    def popen(cmd, stdin=None, stdout=None, stderr=None, text=None, bufsize=None):
        calls.items.append((list(cmd), None))
        return FakePopen("[collama] remote script exit code: 1\n", returncode=0, inputs=calls.popen_inputs)

    with (
        patch("shutil.which", return_value="/bin/colab"),
        patch("subprocess.run", run),
        patch("subprocess.Popen", popen),
    ):
        assert cli.main(["install"]) == 1

    body = remote_script_payload(calls)
    assert "raise RuntimeError" not in body
    assert "remote script exit code" in body


def test_colab_cli_timeout_after_remote_success_is_ignored() -> None:
    calls = Calls()

    def run(cmd, input=None, text=None, stdout=None, stderr=None, capture_output=None):
        calls.items.append((list(cmd), input))
        if cmd[-3:] == ["status", "-s", "collama"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="Name: collama | Hardware: G4 | Status: IDLE\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0)

    def popen(cmd, stdin=None, stdout=None, stderr=None, text=None, bufsize=None):
        calls.items.append((list(cmd), None))
        calls.last_popen = FakePopen(
            "[collama:install] Installed llama-server version\n"
            "[collama] remote script exit code: 0\n"
            "Traceback (most recent call last):\n"
            "TimeoutError: Timeout waiting for reply\n",
            returncode=1,
            inputs=calls.popen_inputs,
        )
        return calls.last_popen

    with (
        patch("shutil.which", return_value="/bin/colab"),
        patch("subprocess.run", run),
        patch("subprocess.Popen", popen),
    ):
        assert cli.main(["install"]) == 0

    assert calls.last_popen is not None
    assert calls.last_popen.terminated


def test_colab_cli_traceback_after_remote_success_is_suppressed(capsys) -> None:
    calls = Calls()

    def run(cmd, input=None, text=None, stdout=None, stderr=None, capture_output=None):
        calls.items.append((list(cmd), input))
        if cmd[-3:] == ["status", "-s", "collama"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="Name: collama | Hardware: G4 | Status: IDLE\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0)

    def popen(cmd, stdin=None, stdout=None, stderr=None, text=None, bufsize=None):
        calls.items.append((list(cmd), None))
        return FakePopen(
            "[collama:install] Installed llama-server version\n"
            "[collama] remote script exit code: 0\n"
            "Traceback (most recent call last):\n"
            "TimeoutError: Timeout waiting for reply\n",
            returncode=1,
            inputs=calls.popen_inputs,
        )

    with (
        patch("shutil.which", return_value="/bin/colab"),
        patch("subprocess.run", run),
        patch("subprocess.Popen", popen),
    ):
        assert cli.main(["install"]) == 0

    output = capsys.readouterr().out
    assert "[collama] remote script exit code: 0" in output
    assert "Traceback" not in output
    assert "TimeoutError" not in output


def test_new_creates_session_without_running_install_script() -> None:
    calls = Calls()
    assert run_cli(["new"], calls) == 0
    assert calls.items == [(["colab", "new", "-s", "collama", "--gpu", "G4"], None)]


def test_tailscale_uses_authkey_hostname_and_starts_daemon() -> None:
    calls = Calls()
    assert run_cli(["tailscale", "--authkey", "tskey-test", "--hostname", "colab-node"], calls) == 0
    body = remote_script_payload(calls)
    assert "curl -fsSL https://tailscale.com/install.sh | sh" in body
    assert "tailscale up --hostname colab-node --authkey tskey-test" in body
    assert "tailscaled --tun=userspace-networking" in body


def test_tailscale_without_authkey_prints_manual_command() -> None:
    calls = Calls()
    assert run_cli(["tailscale", "--hostname", "manual-node"], calls) == 0
    body = remote_script_payload(calls)
    assert "TAILSCALE_AUTHKEY was not provided" in body
    assert "tailscale up --hostname manual-node" in body


def test_healthcheck_prints_json() -> None:
    with patch("colla.cli.check_health", return_value={"base_url": "http://100.1.2.3:8000"}):
        assert cli.main(["healthcheck", "100.1.2.3"]) == 0


def test_serve_starts_llama_server_with_defaults_and_model() -> None:
    calls = Calls()
    assert run_cli(["serve", "unsloth/Qwen3-GGUF"], calls) == 0
    body = remote_script_payload(calls)
    assert "llama-server" in body
    assert "--hf-repo unsloth/Qwen3-GGUF --host 0.0.0.0 --port 8000" in body
    assert "/tmp/collama-llama-server.pid" in body


def test_stop_does_not_create_session_and_errors_when_no_remote_pid() -> None:
    calls = Calls()
    assert run_cli(["stop"], calls) == 0
    assert calls.items[0][0] == ["colab", "exec", "-s", "collama"]
    body = remote_script_payload(calls)
    assert "No llama-server PID file found" in body
    assert "exit 1" in body


def test_benchy_runs_llama_benchy_against_v1_endpoint() -> None:
    calls = Calls()
    assert run_cli(["benchy", "--address", "100.1.2.3", "--port", "8000", "--model", "repo/model"], calls) == 0
    body = remote_script_payload(calls)
    assert "git+https://github.com/eugr/llama-benchy" in body
    assert "--base-url http://100.1.2.3:8000/v1 --model repo/model" in body


def test_host_prints_tailscale_ip_and_status_without_creating_session() -> None:
    calls = Calls()
    assert run_cli(["host"], calls) == 0
    assert calls.items[0][0] == ["colab", "exec", "-s", "collama"]
    body = remote_script_payload(calls)
    assert "tailscale ip -4" in body
    assert "tailscale status" in body
