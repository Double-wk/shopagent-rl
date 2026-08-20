#!/bin/bash
set -e

BASE=/workspace/openssh
ROOT="$BASE/root"
ETC="$BASE/etc"
PKG="$BASE/packages"

SSHD="$ROOT/usr/sbin/sshd"
CONFIG="$ETC/sshd_config"
AUTH_KEYS="$ETC/authorized_keys"
HOST_KEY="$ETC/ssh_host_ed25519_key"

PIDFILE="$BASE/sshd.pid"
LOGFILE="$BASE/sshd.log"

LIB1="$ROOT/usr/lib/x86_64-linux-gnu"
LIB2="$ROOT/lib/x86_64-linux-gnu"

export LD_LIBRARY_PATH="$LIB1:$LIB2${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "======================================"
echo " Persistent SSH bootstrap / start"
echo "======================================"

mkdir -p "$ROOT" "$ETC" "$PKG"

# ==================================================
# 1. sshd 不存在时自动下载安装到 /workspace
# ==================================================

if [ ! -x "$SSHD" ]; then
    echo "[+] sshd missing, bootstrapping into /workspace"

    cd "$PKG"

    if ! ls openssh-server_*.deb >/dev/null 2>&1; then
        echo "[+] Downloading openssh-server..."
        apt-get update || true
        apt-get download openssh-server
    fi

    if ! ls libwrap0_*.deb >/dev/null 2>&1; then
        echo "[+] Downloading libwrap0..."
        apt-get download libwrap0
    fi

    echo "[+] Extracting packages..."

    OPENSSH_DEB=$(ls -t openssh-server_*.deb | head -1)
    LIBWRAP_DEB=$(ls -t libwrap0_*.deb | head -1)

    dpkg-deb -x "$OPENSSH_DEB" "$ROOT"
    dpkg-deb -x "$LIBWRAP_DEB" "$ROOT"

    if [ ! -x "$SSHD" ]; then
        echo "ERROR: failed to create $SSHD"
        exit 1
    fi

    echo "[+] sshd installed into /workspace"
fi


# ==================================================
# 2. 检查动态库
# ==================================================

export LD_LIBRARY_PATH="$LIB1:$LIB2${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MISSING_LIBS=$(ldd "$SSHD" 2>/dev/null | grep "not found" || true)

if [ -n "$MISSING_LIBS" ]; then
    echo "ERROR: sshd still has missing libraries:"
    echo "$MISSING_LIBS"
    exit 1
fi

echo "[+] Dependencies OK"


# ==================================================
# 3. 持久化 Host Key
# ==================================================

if [ ! -f "$HOST_KEY" ]; then
    echo "[+] Generating persistent SSH host key"

    ssh-keygen \
        -t ed25519 \
        -f "$HOST_KEY" \
        -N ""
fi

chmod 600 "$HOST_KEY"
chown root:root "$HOST_KEY"


# ==================================================
# 4. 持久化 + 自动同步 authorized_keys
# ==================================================

touch "$AUTH_KEYS"

# 每次运行都把 AMD 当前实例中的公钥合并进 workspace
if [ -f /root/.ssh/authorized_keys ]; then
    echo "[+] Syncing /root/.ssh/authorized_keys -> workspace"

    cat /root/.ssh/authorized_keys >> "$AUTH_KEYS"
fi

# 如果最后还是空的，说明没有任何可登录公钥
if [ ! -s "$AUTH_KEYS" ]; then
    echo "ERROR: no authorized SSH keys found"
    echo "Please add your public key to:"
    echo "  /root/.ssh/authorized_keys"
    echo "or:"
    echo "  $AUTH_KEYS"
    exit 1
fi

# 去重
sort -u "$AUTH_KEYS" -o "$AUTH_KEYS"

chmod 600 "$AUTH_KEYS"
chown root:root "$AUTH_KEYS"

echo "[+] Authorized keys:"
ssh-keygen -lf "$AUTH_KEYS" || true


# ==================================================
# 5. 自动生成 sshd_config
# ==================================================

if [ ! -f "$CONFIG" ]; then
    echo "[+] Creating sshd_config"

    {
        echo "Port 22"
        echo "ListenAddress 0.0.0.0"
        echo ""
        echo "HostKey $HOST_KEY"
        echo ""
        echo "PermitRootLogin yes"
        echo "PubkeyAuthentication yes"
        echo "PasswordAuthentication no"
        echo "KbdInteractiveAuthentication no"
        echo ""
        echo "AuthorizedKeysFile $AUTH_KEYS"
        echo ""
        echo "UsePAM no"
        echo "PidFile $PIDFILE"
        echo ""
        echo "Subsystem sftp internal-sftp"
        echo ""
        echo "LogLevel INFO"
    } > "$CONFIG"
fi


# ==================================================
# 6. 创建运行目录
# ==================================================

mkdir -p /run/sshd
chmod 755 /run/sshd


# ==================================================
# 7. 创建 sshd 系统用户
# ==================================================

if ! getent passwd sshd >/dev/null 2>&1; then
    echo "[+] Creating sshd system user"

    NOLOGIN=/usr/sbin/nologin

    if [ ! -x "$NOLOGIN" ]; then
        NOLOGIN=/bin/false
    fi

    useradd \
        -r \
        -M \
        -d /run/sshd \
        -s "$NOLOGIN" \
        sshd
fi


# ==================================================
# 8. AMD 新容器可能重新锁定 root
# ==================================================

ROOT_STATUS=$(passwd -S root 2>/dev/null | awk '{print $2}')

if [ "$ROOT_STATUS" = "L" ]; then
    echo "[+] Root is locked, enabling public-key SSH"

    RANDOM_PASS=$(openssl rand -base64 48)

    echo "root:$RANDOM_PASS" | chpasswd

    unset RANDOM_PASS
fi

ROOT_STATUS=$(passwd -S root 2>/dev/null | awk '{print $2}')

echo "[+] Root status: $ROOT_STATUS"


# ==================================================
# 9. 检查 sshd 配置
# ==================================================

"$SSHD" -t -f "$CONFIG"

echo "[+] Config OK"


# ==================================================
# 10. 检查 workspace sshd 是否已经运行
# ==================================================

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE" 2>/dev/null || true)

    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "[+] sshd already running"
        echo "[+] PID=$PID"
        exit 0
    fi

    echo "[+] Removing stale PID file"
    rm -f "$PIDFILE"
fi


# ==================================================
# 11. 启动 workspace sshd
# ==================================================

echo "[+] Starting sshd from /workspace"

"$SSHD" \
    -f "$CONFIG" \
    -E "$LOGFILE"

sleep 1


# ==================================================
# 12. 验证
# ==================================================

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")

    if kill -0 "$PID" 2>/dev/null; then
        echo ""
        echo "======================================"
        echo " SSH STARTED SUCCESSFULLY"
        echo "======================================"
        echo "PID:    $PID"
        echo "SSHD:   $SSHD"
        echo "Config: $CONFIG"
        echo "Log:    $LOGFILE"
        echo ""

        tail -5 "$LOGFILE"

        exit 0
    fi
fi

echo "ERROR: sshd failed to start"

tail -50 "$LOGFILE" 2>/dev/null || true

exit 1
