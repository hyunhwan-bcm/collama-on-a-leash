# collama-on-a-leash

Colab orchestration for running `llama.cpp` with CUDA through
[`google-colab-cli`](https://github.com/googlecolab/google-colab-cli).

This replaces the old Ollama bootstrap. The local `collama` CLI creates or
reuses a Colab session, runs setup commands remotely, exposes the server with
Tailscale, and can benchmark the OpenAI-compatible `llama-server` endpoint.

## Install locally

Install the Colab CLI first:

```bash
uv tool install -U google-colab-cli
```

Install this project in editable mode:

```bash
python3 -m pip install -e .
```

You can also run from the checkout:

```bash
./install.sh --help
python3 -m colla --help
```

## Commands

All remote commands use a Colab session named `collama` by default. Override it
with `-s NAME`. New sessions request a G4 GPU by default; use `--gpu T4`,
`--gpu L4`, `--gpu A100`, or `--gpu H100` when creating a session.

### `new`

Create the Colab session without installing anything:

```bash
collama new
```

This is optional. `install`, `tailscale`, `serve`, and `benchy` create the
session automatically when it is missing.

### `install`

Build `llama.cpp` from source with CUDA enabled:

```bash
collama install
```

If the `collama` Colab session does not exist yet, `install` creates it first.

The remote build performs:

```bash
git clone https://github.com/ggml-org/llama.cpp /content/llama.cpp
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build --config Release -j "$(nproc)" --target llama-server llama-cli llama-bench
```

Options:

```bash
collama install --ref master --jobs 4
```

### `tailscale`

Install/start `tailscaled` on the Colab runtime and join your Tailnet:

```bash
TAILSCALE_AUTHKEY=tskey-auth-... collama tailscale
```

Without `TAILSCALE_AUTHKEY`, the command installs Tailscale and prints the
manual `tailscale up` command to run inside Colab. A remote Colab VM needs an
auth key or an interactive login step; local machine status is not enough to
authenticate the remote machine.

Options:

```bash
collama tailscale \
  --authkey tskey-auth-... \
  --hostname collama-colab \
  --advertise-tags tag:colab
```

### `serve`

Start `llama-server` with a Hugging Face GGUF repo:

```bash
collama serve unsloth/Qwen3-GGUF
```

Defaults are `--host 0.0.0.0` and `--port 8000`.

Pass through server options after `--`:

```bash
collama serve unsloth/Qwen3-GGUF --ctx-size 32768 --gpu-layers 99 -- --jinja
```

The remote PID and log are stored at:

- `/tmp/collama-llama-server.pid`
- `/tmp/collama-llama-server.log`

### `healthcheck`

Check the server from your local machine through its Tailscale address:

```bash
collama healthcheck 100.x.y.z --port 8000
```

This probes:

- `http://ADDRESS:PORT/health`
- `http://ADDRESS:PORT/v1/models`

### `stop`

Stop the remote `llama-server` process:

```bash
collama stop
```

This fails if no PID file exists or if the process is already gone.

### `benchy`

Run [`llama-benchy`](https://github.com/eugr/llama-benchy) from the Colab
runtime against the OpenAI-compatible endpoint:

```bash
collama benchy --address 127.0.0.1 --port 8000 --model unsloth/Qwen3-GGUF
```

For a Tailscale endpoint:

```bash
collama benchy --address 100.x.y.z --port 8000 --model unsloth/Qwen3-GGUF
```

Extra `llama-benchy` arguments can be passed after `--`:

```bash
collama benchy --model unsloth/Qwen3-GGUF -- --depth 0 4096 8192 --no-cache
```

### `host`

Print the remote Tailscale IPv4 address and status:

```bash
collama host
```

## Typical flow

```bash
collama install
TAILSCALE_AUTHKEY=tskey-auth-... collama tailscale
collama serve unsloth/Qwen3-GGUF --ctx-size 32768 --gpu-layers 99
collama host
collama healthcheck 100.x.y.z --port 8000
collama benchy --address 100.x.y.z --port 8000 --model unsloth/Qwen3-GGUF
collama stop
```

## Testing

The test suite mocks `google-colab-cli` and validates command construction plus
all generated remote scripts:

```bash
pytest
```
