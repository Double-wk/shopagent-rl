#!/bin/bash
set -e

ROOT=/workspace/openssh/root
SSHD="$ROOT/usr/sbin/sshd"
CONFIG=/workspace/openssh/etc/sshd_config
AUTH_KEYS=/workspace/openssh/etc/authorized_keys
PIDFILE=/workspace/openssh/sshd.pid
LOGFILE=/workspace/openssh/sshd.log

export LD_LIBRARY_PATH="$ROOT/usr/lib/x86_64-linux-gnu:$ROOT/lib/x86_64-linux-gnu"

echo "=== Starting persistent SSH ==="

# sshd binary
if [ ! -x "$SSHD" ]; then
    echo "ERROR: sshd not found at $SSHD"
    exit 1
fi

# Runtime directory
mkdir -p /run/sshd
chmod 755 /run/sshd

# sshd system user may disappear after container recreation
if ! getent passwd sshd >/dev/null 2>&1; then
    echo "[+] Creating sshd user"
    useradd -r -M -d /run/sshd -s /usr/sbin/nologin sshd
fi

# Root may become locked after container recreation
ROOT_STATUS=$(passwd -S root 2>/dev/null | awk '{print $2}')

if [ "$ROOT_STATUS" = "L" ]; then
    echo "[+] Root account locked, unlocking for public-key SSH"

    RANDOM_PASS=$(openssl rand -base64 48)
    echo "root:$RANDOM_PASS" | chpasswd
    unset RANDOM_PASS
fi

echo "[+] Root status: $(passwd -S root | awk '{print $2}')"

# Authorized keys
if [ ! -f "$AUTH_KEYS" ]; then
    echo "ERROR: authorized_keys missing"
    exit 1
fi

sort -u "$AUTH_KEYS" -o "$AUTH_KEYS"
chmod 600 "$AUTH_KEYS"
chown root:root "$AUTH_KEYS"

# Validate config
"$SSHD" -t -f "$CONFIG"
echo "[+] Config OK"

# Check our own sshd
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE" 2>/dev/null || true)

    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "[+] sshd already running, PID=$PID"
        exit 0
    fi

    rm -f "$PIDFILE"
fi

# Start
"$SSHD" \
  -f "$CONFIG" \
  -E "$LOGFILE"

sleep 1

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")

    if kill -0 "$PID" 2>/dev/null; then
        echo "[+] SSH started successfully, PID=$PID"
        tail -5 "$LOGFILE"
        exit 0
    fi
fi

echo "ERROR: sshd failed to start"
tail -50 "$LOGFILE"
exit 1
