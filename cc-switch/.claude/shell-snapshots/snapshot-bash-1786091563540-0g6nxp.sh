# Snapshot file
# Unset all aliases to avoid conflicts with functions
unalias -a 2>/dev/null || true
# Functions
eval "$(echo 'X19jb25kYV9hY3RpdmF0ZSAoKSAKeyAKICAgIGlmIFsgLW4gIiR7Q09OREFfUFMxX0JBQ0tVUDor
eH0iIF07IHRoZW4KICAgICAgICBQUzE9IiRDT05EQV9QUzFfQkFDS1VQIjsKICAgICAgICBcdW5z
ZXQgQ09OREFfUFMxX0JBQ0tVUDsKICAgIGZpOwogICAgXGxvY2FsIGFza19jb25kYTsKICAgIGFz
a19jb25kYT0iJChQUzE9IiR7UFMxOi19IiBfX2NvbmRhX2V4ZSBzaGVsbC5wb3NpeCAiJEAiKSIg
fHwgXHJldHVybjsKICAgIFxldmFsICIkYXNrX2NvbmRhIjsKICAgIF9fY29uZGFfaGFzaHIKfQo=' | base64 -d)" > /dev/null 2>&1
eval "$(echo 'X19jb25kYV9leGUgKCkgCnsgCiAgICAoIGlmIFsgLW4gIiR7X0NFX006K3h9IiBdICYmIFsgLW4g
IiR7X0NFX0NPTkRBOit4fSIgXTsgdGhlbgogICAgICAgICIkQ09OREFfRVhFIiAkX0NFX00gJF9D
RV9DT05EQSAiJEAiOwogICAgZWxzZQogICAgICAgICIkQ09OREFfRVhFIiAiJEAiOwogICAgZmkg
KQp9Cg==' | base64 -d)" > /dev/null 2>&1
eval "$(echo 'X19jb25kYV9oYXNociAoKSAKeyAKICAgIGlmIFsgLW4gIiR7WlNIX1ZFUlNJT046K3h9IiBdOyB0
aGVuCiAgICAgICAgXHJlaGFzaDsKICAgIGVsc2UKICAgICAgICBpZiBbIC1uICIke1BPU0hfVkVS
U0lPTjoreH0iIF07IHRoZW4KICAgICAgICAgICAgOjsKICAgICAgICBlbHNlCiAgICAgICAgICAg
IFxoYXNoIC1yOwogICAgICAgIGZpOwogICAgZmkKfQo=' | base64 -d)" > /dev/null 2>&1
eval "$(echo 'X19jb25kYV9yZWFjdGl2YXRlICgpIAp7IAogICAgZWNobyAiJ19fY29uZGFfcmVhY3RpdmF0ZScg
aXMgZGVwcmVjYXRlZCBhbmQgd2lsbCBiZSByZW1vdmVkIGluIDI1LjkuIFVzZSAnX19jb25kYV9h
Y3RpdmF0ZSByZWFjdGl2YXRlJyBpbnN0ZWFkLiIgMT4mMjsKICAgIF9fY29uZGFfYWN0aXZhdGUg
cmVhY3RpdmF0ZQp9Cg==' | base64 -d)" > /dev/null 2>&1
eval "$(echo 'Y29uZGEgKCkgCnsgCiAgICBcbG9jYWwgY21kPSIkezEtX19taXNzaW5nX199IjsKICAgIGNhc2Ug
IiRjbWQiIGluIAogICAgICAgIGFjdGl2YXRlIHwgZGVhY3RpdmF0ZSkKICAgICAgICAgICAgX19j
b25kYV9hY3RpdmF0ZSAiJEAiCiAgICAgICAgOzsKICAgICAgICBpbnN0YWxsIHwgdXBkYXRlIHwg
dXBncmFkZSB8IHJlbW92ZSB8IHVuaW5zdGFsbCkKICAgICAgICAgICAgX19jb25kYV9leGUgIiRA
IiB8fCBccmV0dXJuOwogICAgICAgICAgICBfX2NvbmRhX2FjdGl2YXRlIHJlYWN0aXZhdGUKICAg
ICAgICA7OwogICAgICAgICopCiAgICAgICAgICAgIF9fY29uZGFfZXhlICIkQCIKICAgICAgICA7
OwogICAgZXNhYwp9Cg==' | base64 -d)" > /dev/null 2>&1
# Shell Options
shopt -u autocd
shopt -u assoc_expand_once
shopt -u cdable_vars
shopt -u cdspell
shopt -u checkhash
shopt -u checkjobs
shopt -s checkwinsize
shopt -s cmdhist
shopt -u compat31
shopt -u compat32
shopt -u compat40
shopt -u compat41
shopt -u compat42
shopt -u compat43
shopt -u compat44
shopt -s complete_fullquote
shopt -u direxpand
shopt -u dirspell
shopt -u dotglob
shopt -u execfail
shopt -u expand_aliases
shopt -u extdebug
shopt -u extglob
shopt -s extquote
shopt -u failglob
shopt -s force_fignore
shopt -s globasciiranges
shopt -s globskipdots
shopt -u globstar
shopt -u gnu_errfmt
shopt -u histappend
shopt -u histreedit
shopt -u histverify
shopt -s hostcomplete
shopt -u huponexit
shopt -u inherit_errexit
shopt -s interactive_comments
shopt -u lastpipe
shopt -u lithist
shopt -u localvar_inherit
shopt -u localvar_unset
shopt -s login_shell
shopt -u mailwarn
shopt -u no_empty_cmd_completion
shopt -u nocaseglob
shopt -u nocasematch
shopt -u noexpand_translation
shopt -u nullglob
shopt -s patsub_replacement
shopt -s progcomp
shopt -u progcomp_alias
shopt -s promptvars
shopt -u restricted_shell
shopt -u shift_verbose
shopt -s sourcepath
shopt -u varredir_close
shopt -u xpg_echo
set -o braceexpand
set -o hashall
set -o interactive-comments
set -o monitor
set -o onecmd
shopt -s expand_aliases
# Aliases
# Check for rg availability
if ! (unalias rg 2>/dev/null; command -v rg) >/dev/null 2>&1; then
  function rg {
  local _cc_bin="${CLAUDE_CODE_EXECPATH:-}"
  [[ -x $_cc_bin ]] || _cc_bin=/workspace/cc-switch/.local/bin/claude
  if [[ ! -x $_cc_bin ]]; then command rg ${1+"$@"}; return; fi
  if [[ -n ${ZSH_VERSION:-} ]]; then
    ARGV0=rg "$_cc_bin" ${1+"$@"}
  elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    ARGV0=rg "$_cc_bin" ${1+"$@"}
  else
    (exec -a rg "$_cc_bin" ${1+"$@"})
  fi
}
fi
# Shadow find/grep with embedded bfs/ugrep
unalias find 2>/dev/null || true
unalias grep 2>/dev/null || true
function find {
  local _cc_bin="${CLAUDE_CODE_EXECPATH:-}"
  [[ -x $_cc_bin ]] || _cc_bin=/workspace/cc-switch/.local/bin/claude
  if [[ ! -x $_cc_bin ]]; then command find ${1+"$@"}; return; fi
  if [[ -n ${ZSH_VERSION:-} ]]; then
    ARGV0=bfs "$_cc_bin" -S dfs -regextype findutils-default ${1+"$@"}
  elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    ARGV0=bfs "$_cc_bin" -S dfs -regextype findutils-default ${1+"$@"}
  else
    (exec -a bfs "$_cc_bin" -S dfs -regextype findutils-default ${1+"$@"})
  fi
}
function grep {
  local _cc_a
  for _cc_a in ${1+"$@"}; do
    case "$_cc_a" in -*-filter*|-*-pager*|-*-view*|-*-format-open*|-*-config*|---*|-@*|-*-save-config*|-[Zz]*|-[!-]*[Zz]*|--null|--null-data) command grep ${1+"$@"}; return ;; esac
  done
  local _cc_bin="${CLAUDE_CODE_EXECPATH:-}"
  [[ -x $_cc_bin ]] || _cc_bin=/workspace/cc-switch/.local/bin/claude
  if [[ ! -x $_cc_bin ]]; then command grep ${1+"$@"}; return; fi
  if [[ -n ${ZSH_VERSION:-} ]]; then
    ARGV0=ugrep "$_cc_bin" -G --ignore-files --hidden -I --exclude-dir=.git --exclude-dir=.svn --exclude-dir=.hg --exclude-dir=.bzr --exclude-dir=.jj --exclude-dir=.sl ${1+"$@"}
  elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    ARGV0=ugrep "$_cc_bin" -G --ignore-files --hidden -I --exclude-dir=.git --exclude-dir=.svn --exclude-dir=.hg --exclude-dir=.bzr --exclude-dir=.jj --exclude-dir=.sl ${1+"$@"}
  else
    (exec -a ugrep "$_cc_bin" -G --ignore-files --hidden -I --exclude-dir=.git --exclude-dir=.svn --exclude-dir=.hg --exclude-dir=.bzr --exclude-dir=.jj --exclude-dir=.sl ${1+"$@"})
  fi
}
# Shadow pkill to refuse patterns matching the CLI process
unalias pkill 2>/dev/null || true
function pkill {
  if [ -n "${CLAUDE_PID:-}" ] && [ -r "/proc/${CLAUDE_PID}/comm" ]; then
    local _cc_skip="" _cc_a
    local -a _cc_probe=()
    for _cc_a in ${1+"$@"}; do
      if [ -n "$_cc_skip" ]; then _cc_skip=""; continue; fi
      case "$_cc_a" in
        --signal) _cc_skip=1 ;;
        --signal=*|-e|--echo) ;;
        -[0-9]*) ;;
        -[PUGOF]?*) _cc_probe+=("$_cc_a") ;;
        -[ABCDEFGHIJKLMNOPQRSTUVWXYZ][ABCDEFGHIJKLMNOPQRSTUVWXYZ0-9]*) ;;
        *) _cc_probe+=("$_cc_a") ;;
      esac
    done
    if command pgrep ${_cc_probe[@]+"${_cc_probe[@]}"} 2>/dev/null | command grep -qx "${CLAUDE_PID}"; then
      printf 'pkill: refusing to run — this pattern matches the Claude CLI process (PID %s). Narrow the pattern, or target your own children with `pkill -P $$ ...`.\n' "${CLAUDE_PID}" >&2
      return 1
    fi
  fi
  command pkill ${1+"$@"}
}
export PATH=/root/.opencode/bin:/workspace/miniconda3/bin:/workspace/miniconda3/condabin:/root/.opencode/bin:/root/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/root/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli:/root/.vscode-server/cli/servers/Stable-1b6a188127eeaf9194f945eb6eb89a657e93c54c/server/bin/remote-cli:/root/.opencode/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
