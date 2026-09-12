#!/bin/sh
# sprout-local one-line install script.
#
#   curl -fsSL https://raw.githubusercontent.com/sprout-foundry/sprout-local/main/scripts/install.sh | sh
#
# Downloads the latest release tarball for this platform, verifies it,
# and installs to ~/.local/bin (or /usr/local/bin as a fallback).
# Pin a version with SPROUT_LOCAL_VERSION=vX.Y.Z.
set -eu

# Colors with fallback for non-tty
if [ -t 1 ]; then
    GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; BLUE=''; NC=''
fi
log_info()    { printf '%b[INFO]%b %s\n' "$BLUE" "$NC" "$1"; }
log_success() { printf '%b[SUCCESS]%b %s\n' "$GREEN" "$NC" "$1"; }
log_error()   { printf '%b[ERROR]%b %s\n' "$RED" "$NC" "$1" >&2; }

REPO="sprout-foundry/sprout-local"
BINARY="sprout-local"
VERSION="${SPROUT_LOCAL_VERSION:-}"

cleanup() {
    if [ -n "${TEMP_DIR:-}" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}
TEMP_DIR=""
trap cleanup EXIT INT TERM

need() {
    for cmd in "$@"; do
        command -v "$cmd" >/dev/null 2>&1 || {
            log_error "$cmd is required but not installed"
            exit 1
        }
    done
}
need curl tar awk grep unzip

# ── platform ────────────────────────────────────────────────────────────
OS=$(uname -s)
ARCH=$(uname -m)
case "$OS" in
    Darwin) plat="darwin" ;;
    Linux)  plat="linux" ;;
    *) log_error "unsupported OS: $OS (the MLX backend is macOS-only; Linux needs the ggml tag and libggml)"; exit 1 ;;
esac
case "$ARCH" in
    arm64|aarch64) arch="arm64" ;;
    x86_64|amd64)  arch="amd64" ;;
    *) log_error "unsupported architecture: $ARCH"; exit 1 ;;
esac

# ── version: env pin or latest release ─────────────────────────────────
if [ -z "$VERSION" ]; then
    log_info "Resolving latest release…"
    response=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>&1) || {
        log_error "Failed to reach the GitHub API."
        log_error "Pin a version and re-run:  SPROUT_LOCAL_VERSION=vX.Y.Z curl -fsSL … | sh"
        exit 1
    }
    VERSION=$(echo "$response" | awk -F'"' '/"tag_name":/ {print $4; exit}')
    [ -n "$VERSION" ] || { log_error "GitHub API returned no tag_name."; exit 1; }
fi
log_info "Installing $BINARY $VERSION for $plat/$arch…"

# ── download + verify ──────────────────────────────────────────────────
ASSET="${BINARY}-${plat}-${arch}.tar.gz"
URL="https://github.com/$REPO/releases/download/${VERSION}/${ASSET}"
TEMP_DIR=$(mktemp -d)

log_info "Downloading $ASSET…"
curl -fsSL --progress-bar -o "$TEMP_DIR/$ASSET" "$URL"

if command -v shasum >/dev/null 2>&1 || command -v sha256sum >/dev/null 2>&1; then
    SUMS_URL="https://github.com/$REPO/releases/download/${VERSION}/SHA256SUMS"
    if curl -fsSL -o "$TEMP_DIR/SHA256SUMS" "$SUMS_URL" 2>/dev/null; then
        expected=$(grep "$ASSET" "$TEMP_DIR/SHA256SUMS" | awk '{print $1}')
        if [ -n "$expected" ]; then
            if command -v shasum >/dev/null 2>&1; then
                actual=$(shasum -a 256 "$TEMP_DIR/$ASSET" | awk '{print $1}')
            else
                actual=$(sha256sum "$TEMP_DIR/$ASSET" | awk '{print $1}')
            fi
            if [ "$expected" != "$actual" ]; then
                log_error "checksum mismatch for $ASSET"
                exit 1
            fi
            log_info "Checksum verified."
        fi
    fi
fi

# ── extract + install ──────────────────────────────────────────────────
tar -xzf "$TEMP_DIR/$ASSET" -C "$TEMP_DIR"

# Install target: ~/.local/bin unless root, then /usr/local/bin.
if [ "$(id -u)" = "0" ]; then
    INSTALL_DIR="/usr/local/bin"
else
    INSTALL_DIR="$HOME/.local/bin"
fi
mkdir -p "$INSTALL_DIR"
mv "$TEMP_DIR/${BINARY}-${plat}-${arch}" "$INSTALL_DIR/$BINARY"
chmod +x "$INSTALL_DIR/$BINARY"

log_success "Installed $BINARY $VERSION → $INSTALL_DIR/$BINARY"
case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *) log_warn "$INSTALL_DIR is not in your PATH — add it to your shell profile." ;;
esac

cat <<'NEXT'

Next steps:
  sprout-local -pull        list downloadable models (RAM-tier annotated)
  sprout-local -pull <name> download one, then chat
  sprout-local              interactive REPL (models: ~/.sprout-local/models)
  sprout-local -serve       web UI + OpenAI-style /v1 API on :8321

NEXT
