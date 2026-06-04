from __future__ import annotations

import argparse
import json
import os
import socket
import sys

from .colab import ColabError, ColabOptions, ColabRunner
from .health import check_health
from .scripts import benchy_script, host_script, install_script, serve_script, stop_script, tailscale_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collama",
        description="Run llama.cpp on Google Colab through google-colab-cli.",
    )
    parser.add_argument("-s", "--session", default="collama", help="Colab session name.")
    parser.add_argument("--gpu", default="L4", choices=["T4", "L4", "A100", "H100"], help="GPU for new sessions.")
    parser.add_argument("--auth", choices=["oauth2", "adc"], help="Authentication mode passed to colab.")
    parser.add_argument("--config", help="Session config path passed to colab.")
    parser.add_argument("--colab-bin", default="colab", help="google-colab-cli executable.")
    parser.add_argument("--dry-run", action="store_true", help="Print Colab commands and generated scripts.")

    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="Build llama.cpp with CUDA on the Colab runtime.")
    install.add_argument("--ref", default="master", help="llama.cpp git ref to build.")
    install.add_argument("--jobs", type=int, help="Parallel build jobs; default is nproc on Colab.")

    tailscale = sub.add_parser("tailscale", help="Install/start tailscaled and join the local Tailnet.")
    tailscale.add_argument("--authkey", default=os.getenv("TAILSCALE_AUTHKEY"), help="Tailscale auth key or TAILSCALE_AUTHKEY.")
    tailscale.add_argument("--hostname", default=_default_hostname(), help="Tailscale hostname for the Colab runtime.")
    tailscale.add_argument("--login-server", default=os.getenv("TAILSCALE_LOGIN_SERVER"), help="Optional custom coordination server.")
    tailscale.add_argument("--advertise-tags", default=os.getenv("TAILSCALE_ADVERTISE_TAGS"), help="Optional Tailscale tags.")

    health = sub.add_parser("healthcheck", help="Check llama-server through a Tailscale address and port.")
    health.add_argument("address", help="Tailscale IP or DNS name.")
    health.add_argument("-p", "--port", type=int, default=8000, help="llama-server port.")
    health.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")

    serve = sub.add_parser("serve", help="Start llama-server on the Colab runtime.")
    serve.add_argument("model", help="Hugging Face GGUF repo, for example unsloth/Qwen3-GGUF.")
    serve.add_argument("--host", default="0.0.0.0", help="llama-server host.")
    serve.add_argument("-p", "--port", type=int, default=8000, help="llama-server port.")
    serve.add_argument("--ctx-size", type=int, help="Context size passed to llama-server.")
    serve.add_argument("--gpu-layers", type=int, help="GPU layers passed to llama-server.")
    serve.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra llama-server args after --.")

    sub.add_parser("stop", help="Stop the remote llama-server process.")

    benchy = sub.add_parser("benchy", help="Run llama-benchy against an OpenAI-compatible endpoint.")
    benchy.add_argument("--address", default="127.0.0.1", help="Endpoint address from the Colab runtime.")
    benchy.add_argument("-p", "--port", type=int, default=8000, help="Endpoint port.")
    benchy.add_argument("--model", required=True, help="Model name to pass to llama-benchy.")
    benchy.add_argument("--pp", type=int, default=2048, help="Prompt processing tokens.")
    benchy.add_argument("--tg", type=int, default=256, help="Generated tokens.")
    benchy.add_argument("--runs", type=int, default=3, help="Benchmark runs.")
    benchy.add_argument("--concurrency", type=int, default=1, help="Concurrent requests.")
    benchy.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra llama-benchy args after --.")

    sub.add_parser("host", help="Show the Colab runtime's Tailscale address.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (ColabError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "healthcheck":
        result = check_health(args.address, args.port, timeout=args.timeout)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    runner = ColabRunner(
        ColabOptions(
            session=args.session,
            gpu=args.gpu,
            auth=args.auth,
            config=args.config,
            colab_bin=args.colab_bin,
        ),
        dry_run=args.dry_run,
    )

    if args.command == "install":
        runner.run_bash(install_script(ref=args.ref, jobs=args.jobs))
    elif args.command == "tailscale":
        runner.run_bash(
            tailscale_script(
                authkey=args.authkey,
                hostname=args.hostname,
                login_server=args.login_server,
                advertise_tags=args.advertise_tags,
            )
        )
    elif args.command == "serve":
        runner.run_bash(
            serve_script(
                model=args.model,
                host=args.host,
                port=args.port,
                ctx_size=args.ctx_size,
                gpu_layers=args.gpu_layers,
                extra_args=_trim_remainder(args.extra_args),
            )
        )
    elif args.command == "stop":
        runner.run_bash(stop_script(), create=False)
    elif args.command == "benchy":
        base_url = f"http://{args.address}:{args.port}/v1"
        runner.run_bash(
            benchy_script(
                base_url=base_url,
                model=args.model,
                pp=args.pp,
                tg=args.tg,
                runs=args.runs,
                concurrency=args.concurrency,
                extra_args=_trim_remainder(args.extra_args),
            )
        )
    elif args.command == "host":
        runner.run_bash(host_script(), create=False)
    else:
        raise RuntimeError(f"unknown command: {args.command}")
    return 0


def _trim_remainder(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _default_hostname() -> str:
    name = socket.gethostname().split(".")[0] or "local"
    safe = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in name.lower()).strip("-")
    return f"collama-{safe or 'local'}"
