from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


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
            return
        self.create_session()

    def create_session(self) -> None:
        self.require_cli()
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
        self._run(self._base_cmd(["exec", "-s", self.options.session]), input_text=payload)

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

    def _run(self, cmd: list[str], *, input_text: str | None = None) -> None:
        if self.dry_run:
            self._print_command(cmd)
            if input_text:
                print(input_text)
            return
        proc = subprocess.run(cmd, input=input_text, text=True)
        if proc.returncode != 0:
            raise ColabError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")

    @staticmethod
    def _print_command(cmd: list[str]) -> None:
        print("+ " + " ".join(cmd), file=sys.stderr)


def _python_payload(script: str) -> str:
    return (
        "import pathlib, subprocess, textwrap\n"
        "script = r'''\n"
        f"{script}\n"
        "'''\n"
        "path = pathlib.Path('/tmp/collama-on-a-leash.sh')\n"
        "path.write_text(script)\n"
        "path.chmod(0o755)\n"
        "raise SystemExit(subprocess.call(['bash', str(path)]))\n"
    )
