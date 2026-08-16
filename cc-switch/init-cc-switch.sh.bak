#!/usr/bin/env bash
set -euo pipefail

# 将脚本路径解析为真实物理路径，避免 /workspace/persistent 符号链接。
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
ROOT_DIR="$(dirname -- "${SCRIPT_PATH}")"

CONFIG_DIR="${ROOT_DIR}/.cc-switch"
BIN="${ROOT_DIR}/bin/cc-switch"
HOME_DIR="${HOME:-/root}"

mkdir -p "${CONFIG_DIR}"

# 首次运行时，从旧的临时配置目录迁移数据。
if [ -d "/root/.cc-switch" ] &&
   [ ! -f "${CONFIG_DIR}/cc-switch.db" ]; then
  cp -a /root/.cc-switch/. "${CONFIG_DIR}/" 2>/dev/null || true
fi

export CC_SWITCH_CONFIG_DIR="${CONFIG_DIR}"

if [ ! -x "${BIN}" ]; then
  echo "cc-switch binary not found: ${BIN}" >&2
  exit 1
fi

init_live_client_configs() {
  mkdir -p \
    "${HOME_DIR}/.claude" \
    "${HOME_DIR}/.codex"

  chmod 700 \
    "${HOME_DIR}/.claude" \
    "${HOME_DIR}/.codex"

  if [ ! -e "${HOME_DIR}/.claude/settings.json" ]; then
    (
      umask 077
      printf '{}\n' > "${HOME_DIR}/.claude/settings.json"
    )
  fi

  if [ ! -e "${HOME_DIR}/.codex/config.toml" ]; then
    (
      umask 077
      touch "${HOME_DIR}/.codex/config.toml"
    )
  fi
}

# 从持久化数据库读出当前 provider，
# 并重新生成 /root/.claude 和 /root/.codex 下的活配置。
restore_current_providers() {
  init_live_client_configs

  local app output id

  for app in claude codex; do
    output="$(
      "${BIN}" --app "${app}" provider current 2>/dev/null || true
    )"

    # 兼容类似以下输出：
    # ID: provider-id
    # Current: provider-id
    # → Current: provider-id
    id="$(
      printf '%s\n' "${output}" |
        awk -F':' '
          /ID:/ || /Current:/ {
            value = $NF
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            if (value != "") {
              print value
              exit
            }
          }
        '
    )"

    if [ -n "${id}" ]; then
      echo "→ 恢复 ${app}: ${id}"

      if ! "${BIN}" --app "${app}" use "${id}"; then
        echo "  警告：${app} 恢复失败" >&2
      fi
    else
      echo "→ ${app}: 没有当前 provider，跳过" >&2
    fi
  done
}

case "${1:-}" in
  ""|restore|init)
    restore_current_providers
    exit 0
    ;;

  use|start)
    init_live_client_configs
    ;;

  provider)
    if [ "${2:-}" = "switch" ]; then
      init_live_client_configs
    fi
    ;;
esac

exec "${BIN}" "$@"