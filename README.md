# collama-on-a-leash

One-stop Colab bootstrap for Ollama with optional SSH, Tailscale, and `llama.cpp` CUDA build.

## Quick start

Run directly from GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/hyunhwan-bcm/collama-on-a-leash/main/install.sh | sh
```

Run locally:

```bash
chmod +x install.sh
./install.sh
```

## Default behavior

The installer configures:

- `OLLAMA_HOST=0.0.0.0:11434`
- `OLLAMA_CONTEXT_LENGTH=40000`
- `OLLAMA_KEEP_ALIVE=5m`
- `OLLAMA_FLASH_ATTENTION=1`
- `OLLAMA_NUM_PARALLEL=4`
- SSH enabled with root login + password authentication
- Root password set from `ROOT_PASSWORD` (default: `root`) when `SET_ROOT_PASSWORD=1`
- `ollama serve` started in background (unless already running)

Defaults only apply if you did not already export those environment variables.

## Configuration

Set variables before running installer.

```bash
OLLAMA_HOST=0.0.0.0:11434 \
OLLAMA_CONTEXT_LENGTH=40000 \
OLLAMA_KEEP_ALIVE=5m \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_NUM_PARALLEL=4 \
ENABLE_SSH=1 \
SET_ROOT_PASSWORD=1 \
ROOT_PASSWORD='change-me' \
INSTALL_TAILSCALE=0 \
INSTALL_LLAMA_CPP=0 \
START_OLLAMA_SERVER=1 \
OLLAMA_MODEL='qwen3:14b' \
curl -fsSL https://raw.githubusercontent.com/hyunhwan-bcm/collama-on-a-leash/main/install.sh | sh
```

### Variables

- `OLLAMA_HOST` default: `0.0.0.0:11434`
- `OLLAMA_CONTEXT_LENGTH` default: `40000`
- `OLLAMA_KEEP_ALIVE` default: `5m`
- `OLLAMA_FLASH_ATTENTION` default: `1`
- `OLLAMA_NUM_PARALLEL` default: `4`
- `ENABLE_SSH` default: `1`
- `SET_ROOT_PASSWORD` default: `1`
- `ROOT_PASSWORD` default: `root`
- `INSTALL_TAILSCALE` default: `0`
- `INSTALL_LLAMA_CPP` default: `0`
- `START_OLLAMA_SERVER` default: `1`
- `OLLAMA_MODEL` default: empty (no auto-pull)
- `TAILSCALE_AUTHKEY` optional if `INSTALL_TAILSCALE=1`

## Example profiles

Minimal Ollama server only:

```bash
ENABLE_SSH=0 \
SET_ROOT_PASSWORD=0 \
INSTALL_TAILSCALE=0 \
INSTALL_LLAMA_CPP=0 \
OLLAMA_MODEL='qwen3:0.6b' \
curl -fsSL https://raw.githubusercontent.com/hyunhwan-bcm/collama-on-a-leash/main/install.sh | sh
```

Colab remote access with custom password:

```bash
ENABLE_SSH=1 \
SET_ROOT_PASSWORD=1 \
ROOT_PASSWORD='strong-password-here' \
INSTALL_TAILSCALE=1 \
TAILSCALE_AUTHKEY='tskey-xxxxx' \
curl -fsSL https://raw.githubusercontent.com/hyunhwan-bcm/collama-on-a-leash/main/install.sh | sh
```

## Verify after install

```bash
ps aux | grep '[o]llama serve'
curl http://127.0.0.1:11434/api/tags
ollama run qwen3:14b --verbose
```

## Notes

- SSH password setup is intentionally isolated in the script so you can disable it with `SET_ROOT_PASSWORD=0`.
- Ollama env defaults are persisted to `/etc/profile.d/ollama-env.sh`.
