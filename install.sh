#!/usr/bin/env sh
set -eu

log() {
  printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf "[%s] WARN: %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*" >&2
}

SUDO=""

run_root() {
  if [ -n "$SUDO" ]; then
    "$SUDO" "$@"
  else
    "$@"
  fi
}

apt_install() {
  if [ -n "$SUDO" ]; then
    DEBIAN_FRONTEND=noninteractive "$SUDO" apt-get install -y "$@"
  else
    DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
  fi
}

require_root_or_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
    return
  fi
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
    return
  fi
  echo "This script must run as root or with sudo available." >&2
  exit 1
}

sshd_set_option() {
  key="$1"
  value="$2"
  cfg="/etc/ssh/sshd_config"

  if [ ! -f "$cfg" ]; then
    warn "Skipping sshd config update because $cfg does not exist."
    return
  fi

  if grep -Eq "^[#[:space:]]*${key}[[:space:]]+" "$cfg"; then
    run_root sed -i -E "s|^[#[:space:]]*${key}[[:space:]]+.*|${key} ${value}|g" "$cfg"
  else
    printf "%s %s\n" "$key" "$value" | run_root tee -a "$cfg" >/dev/null
  fi
}

start_service_if_possible() {
  service_name="$1"
  if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    run_root systemctl restart "$service_name" 2>/dev/null || run_root systemctl start "$service_name" 2>/dev/null || true
  elif command -v service >/dev/null 2>&1; then
    run_root service "$service_name" restart 2>/dev/null || run_root service "$service_name" start 2>/dev/null || true
  fi
}

require_root_or_sudo

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This installer supports Debian/Ubuntu environments with apt-get." >&2
  exit 1
fi

# Defaults (only used if not already exported by caller)
: "${OLLAMA_HOST:=0.0.0.0:11434}"
: "${OLLAMA_CONTEXT_LENGTH:=40000}"
: "${OLLAMA_KEEP_ALIVE:=5m}"
: "${OLLAMA_FLASH_ATTENTION:=1}"
: "${OLLAMA_NUM_PARALLEL:=4}"

# Optional flags
: "${ENABLE_SSH:=1}"
: "${SET_ROOT_PASSWORD:=1}"
: "${ROOT_PASSWORD:=root}"
: "${INSTALL_TAILSCALE:=0}"
: "${START_OLLAMA_SERVER:=1}"
: "${OLLAMA_MODEL:=}"

export OLLAMA_HOST
export OLLAMA_CONTEXT_LENGTH
export OLLAMA_KEEP_ALIVE
export OLLAMA_FLASH_ATTENTION
export OLLAMA_NUM_PARALLEL

log "Collecting quick system info..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  warn "nvidia-smi not found."
fi
df -h || true
ping -c 1 google.com >/dev/null 2>&1 || warn "Ping to google.com failed."

log "Installing base packages..."
run_root apt-get update
apt_install \
  ca-certificates \
  cmake \
  curl \
  git \
  iputils-ping \
  lshw \
  openssh-server \
  pciutils \
  tmux \
  zstd

if [ "$INSTALL_TAILSCALE" = "1" ]; then
  log "Installing Tailscale..."
  curl -fsSL https://tailscale.com/install.sh | sh

  if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    run_root systemctl enable --now tailscaled || true
  else
    run_root nohup tailscaled --tun=userspace-networking >/tmp/tailscaled.log 2>&1 &
  fi

  if [ -n "${TAILSCALE_AUTHKEY:-}" ]; then
    run_root tailscale up --authkey "$TAILSCALE_AUTHKEY"
  else
    warn "Tailscale installed. Run 'sudo tailscale up' (or set TAILSCALE_AUTHKEY) to connect."
  fi
fi

if [ "$ENABLE_SSH" = "1" ]; then
  log "Configuring SSH..."
  sshd_set_option "PermitRootLogin" "yes"
  sshd_set_option "PasswordAuthentication" "yes"

  # Password section (kept isolated intentionally)
  if [ "$SET_ROOT_PASSWORD" = "1" ]; then
    printf "root:%s\n" "$ROOT_PASSWORD" | run_root chpasswd
    log "Root password was set from ROOT_PASSWORD (default: root)."
  else
    warn "Skipping root password setup because SET_ROOT_PASSWORD=$SET_ROOT_PASSWORD."
  fi

  start_service_if_possible "ssh"
  start_service_if_possible "sshd"
fi

log "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

log "Persisting Ollama environment defaults..."
run_root mkdir -p /etc/profile.d
run_root tee /etc/profile.d/ollama-env.sh >/dev/null <<EOF
export OLLAMA_HOST="${OLLAMA_HOST}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}"
EOF
run_root chmod 644 /etc/profile.d/ollama-env.sh

if [ "$START_OLLAMA_SERVER" = "1" ]; then
  if pgrep -f "ollama serve" >/dev/null 2>&1; then
    warn "An Ollama server process is already running. Leaving it unchanged."
  else
    log "Starting Ollama server on http://${OLLAMA_HOST} ..."
    nohup env \
      OLLAMA_HOST="${OLLAMA_HOST}" \
      OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH}" \
      OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE}" \
      OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION}" \
      OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}" \
      ollama serve >/tmp/ollama-server.log 2>&1 &
  fi
fi

if [ -n "$OLLAMA_MODEL" ]; then
  log "Pulling model: $OLLAMA_MODEL"
  ollama pull "$OLLAMA_MODEL"
fi

log "Done."
log "Ollama target: http://${OLLAMA_HOST}"
log "Context length: ${OLLAMA_CONTEXT_LENGTH}"
log "Parallel requests: ${OLLAMA_NUM_PARALLEL}"
