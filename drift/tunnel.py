"""No-account TCP tunnel (bore.pub) so a node behind NAT is reachable.

DRIFT's head *dials* its nodes, but a node on a home network or a Colab/VM has no
inbound reachability. `drift node --tunnel` exposes the node's port to a public
`bore.pub:PORT` — no account, no token — and the head connects with
`drift run --nodes bore.pub:PORT`. bore (github.com/ekzhang/bore) is a single
static binary, downloaded once per platform and cached under ~/.cache/drift.
"""

from __future__ import annotations

import io
import os
import platform
import re
import subprocess
import tarfile
import time
import urllib.request
import zipfile

_BORE_VERSION = "v0.6.0"

# (system, machine) -> release asset suffix
_ASSETS = {
    ("darwin", "arm64"): "aarch64-apple-darwin.tar.gz",
    ("darwin", "aarch64"): "aarch64-apple-darwin.tar.gz",
    ("darwin", "x86_64"): "x86_64-apple-darwin.tar.gz",
    ("linux", "x86_64"): "x86_64-unknown-linux-musl.tar.gz",
    ("linux", "amd64"): "x86_64-unknown-linux-musl.tar.gz",
    ("linux", "aarch64"): "aarch64-unknown-linux-musl.tar.gz",
    ("linux", "arm64"): "aarch64-unknown-linux-musl.tar.gz",
    ("windows", "amd64"): "x86_64-pc-windows-msvc.zip",
    ("windows", "x86_64"): "x86_64-pc-windows-msvc.zip",
}


def _cache_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), ".cache", "drift")
    os.makedirs(d, exist_ok=True)
    return d


def ensure_bore() -> str:
    """Path to the bore binary for this platform (downloads + caches once)."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    asset = _ASSETS.get((system, machine))
    if asset is None:
        raise RuntimeError(
            f"no prebuilt bore for {system}/{machine}; install bore yourself "
            "(github.com/ekzhang/bore) and expose the node port manually"
        )
    exe = "bore.exe" if system == "windows" else "bore"
    dst = os.path.join(_cache_dir(), exe)
    if os.path.exists(dst):
        return dst

    url = (f"https://github.com/ekzhang/bore/releases/download/"
           f"{_BORE_VERSION}/bore-{_BORE_VERSION}-{asset}")
    req = urllib.request.Request(url, headers={"User-Agent": "drift-tunnel"})
    blob = urllib.request.urlopen(req, timeout=60).read()  # bounded: never hang
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            name = next(n for n in z.namelist() if os.path.basename(n) == exe)
            data = z.read(name)
    else:
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            member = next(m for m in t.getmembers() if os.path.basename(m.name) == exe)
            data = t.extractfile(member).read()
    with open(dst, "wb") as f:
        f.write(data)
    os.chmod(dst, 0o755)
    return dst


def open_bore(port: int, timeout: float = 40.0):
    """Expose localhost:port via bore.pub.

    Returns (addr, proc): addr is 'bore.pub:PORT' (or None on failure) and proc
    is the bore subprocess — keep it alive for the node's lifetime, terminate to
    tear the tunnel down.
    """
    bore = ensure_bore()
    logf = os.path.join(_cache_dir(), f"bore-{port}.log")
    proc = subprocess.Popen(
        [bore, "local", str(port), "--to", "bore.pub"],
        stdout=open(logf, "w"), stderr=subprocess.STDOUT, start_new_session=True,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1)
        try:
            log = open(logf).read()
        except FileNotFoundError:
            log = ""
        m = re.search(r"bore\.pub:(\d+)", log) or re.search(r"remote_port[=:\s]+(\d+)", log)
        if m:
            return f"bore.pub:{m.group(1)}", proc
    return None, proc
