from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


REMOTE_EXIT_CODE_PATH = "/tmp/collama-on-a-leash.exit"


@dataclass(frozen=True)
class ColabOptions:
    session: str
    gpu: str | None = "G4"
    auth: str | None = None
    config: str | None = None
    colab_bin: str = "colab"


class ColabError(RuntimeError):
    pass


class ColabRunner:
    def __init__(self, options: ColabOptions, *, dry_run: bool = False) -> None:
        self.options = options
        self.dry_run = dry_run

    def require_cli(self) -> None:
        if self.dry_run:
            return
        if shutil.which(self.options.colab_bin) is None:
            raise ColabError(
                "google-colab-cli is required. Install it with "
                "`uv tool install -U google-colab-cli` or "
                "`pip install -U google-colab-cli`."
            )

    def ensure_session(self) -> None:
        if self._status_ok():
            self._log(f"Using existing Colab session '{self.options.session}'.")
            return
        self._log(
            f"Colab session '{self.options.session}' was not found; creating it with GPU {self.options.gpu}."
        )
        self.create_session()

    def create_session(self) -> None:
        self.require_cli()
        self._log(f"Creating Colab session '{self.options.session}' with GPU {self.options.gpu}.")
        self._run(self.create_session_cmd())

    def create_session_cmd(self) -> list[str]:
        cmd = self._base_cmd(["new", "-s", self.options.session])
        if self.options.gpu:
            cmd.extend(["--gpu", self.options.gpu])
        return cmd

    def stop_session(self) -> None:
        self._run(self._base_cmd(["stop", "-s", self.options.session]))

    def run_bash(self, script: str, *, create: bool = True) -> None:
        self.require_cli()
        if create:
            self.ensure_session()
        payload = _python_payload(script)
        self._log(f"Executing remote script in Colab session '{self.options.session}'.")
        self._run(self._base_cmd(["exec", "-s", self.options.session]), input_text=payload)
        exit_code = self._read_remote_exit_code()
        if exit_code != 0:
            raise ColabError(f"Remote script failed with exit code {exit_code}.")

    def _status_ok(self) -> bool:
        self.require_cli()
        cmd = self._base_cmd(["status", "-s", self.options.session])
        if self.dry_run:
            self._print_command(cmd)
            return False
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            return False
        output = f"{proc.stdout}\n{proc.stderr}".lower()
        return "not found" not in output and "no active sessions" not in output

    def _base_cmd(self, args: list[str]) -> list[str]:
        cmd = [self.options.colab_bin]
        if self.options.auth:
            cmd.extend(["--auth", self.options.auth])
        if self.options.config:
            cmd.extend(["--config", self.options.config])
        cmd.extend(args)
        return cmd

    def _run(
        self,
        cmd: list[str],
        *,
        input_text: str | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str] | None:
        if self.dry_run:
            self._print_command(cmd)
            if input_text:
                print(input_text)
            return
        proc = subprocess.run(cmd, input=input_text, text=True, capture_output=capture_output)
        if proc.returncode != 0:
            raise ColabError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
        return proc

    def _read_remote_exit_code(self) -> int:
        payload = (
            "import pathlib\n"
            f"path = pathlib.Path({REMOTE_EXIT_CODE_PATH!r})\n"
            "value = path.read_text().strip() if path.exists() else 'missing'\n"
            "print(f'__COLLAMA_REMOTE_EXIT_CODE__={value}')\n"
        )
        proc = self._run(
            self._base_cmd(["exec", "-s", self.options.session]),
            input_text=payload,
            capture_output=True,
        )
        if proc is None:
            return 0
        marker = "__COLLAMA_REMOTE_EXIT_CODE__="
        for line in proc.stdout.splitlines():
            if marker in line:
                value = line.rsplit(marker, 1)[1].strip()
                if value.isdigit():
                    return int(value)
                raise ColabError(f"Remote script exit code was not recorded: {value}")
        raise ColabError("Could not read remote script exit code from Colab output.")

    @staticmethod
    def _print_command(cmd: list[str]) -> None:
        print("+ " + " ".join(cmd), file=sys.stderr)

    @staticmethod
    def _log(message: str) -> None:
        print(f"[collama] {message}", file=sys.stderr)


def _python_payload(script: str) -> str:
    return (
        "import pathlib, subprocess, sys\n"
        "script = r'''\n"
        f"{script}\n"
        "'''\n"
        "path = pathlib.Path('/tmp/collama-on-a-leash.sh')\n"
        f"exit_path = pathlib.Path({REMOTE_EXIT_CODE_PATH!r})\n"
        "exit_path.write_text('127')\n"
        "path.write_text(script)\n"
        "path.chmod(0o755)\n"
        "proc = subprocess.Popen(\n"
        "    ['bash', str(path)],\n"
        "    stdout=subprocess.PIPE,\n"
        "    stderr=subprocess.STDOUT,\n"
        "    text=True,\n"
        "    bufsize=1,\n"
        ")\n"
        "assert proc.stdout is not None\n"
        "for line in proc.stdout:\n"
        "    print(line, end='', flush=True)\n"
        "exit_code = proc.wait()\n"
        "exit_path.write_text(str(exit_code))\n"
        "print(f'[collama] remote script exit code: {exit_code}', flush=True)\n"
        "sys.stdout.flush()\n"
    )
