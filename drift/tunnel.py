"""No-account TCP tunnel (bore.pub) so a node behind NAT is reachable.

DRIFT's head *dials* its nodes, but a node on a home network or a Colab/VM has no
inbound reachability. `drift node --tunnel` exposes the node's port to a public
`bore.pub:PORT` — no account, no token — and the head connects with
`drift run --nodes bore.pub:PORT`. bore (github.com/ekzhang/bore) is a single
static binary, downloaded once per platform and cached under ~/.cache/drift.
"""

from __future__ import annotations

import hashlib
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

# SHA256 of each bore v0.6.0 release asset (computed from the published files).
# The download is verified against this before the binary is written + executed —
# a project that verifies its nodes should verify the code it fetches and runs.
_SHA256 = {
    "aarch64-apple-darwin.tar.gz":      "65f43a67b90874700538bdb6064c5e92276e64dfba24f5cd72ef24a035eec3bc",
    "x86_64-apple-darwin.tar.gz":       "206db723a382bbc18d2893fc8472868e0b5f41de35975269aab7891ecc8659cc",
    "x86_64-unknown-linux-musl.tar.gz": "e484d1e3acba77169b773f31a5bfb34192d4b660f44a094a658a2522cd2270f7",
    "aarch64-unknown-linux-musl.tar.gz":"ffc4515f3617420b243758cf36ed6a63208d7dba76b2ec3e90d1f476a9742951",
    "x86_64-pc-windows-msvc.zip":       "01709c64fe2787cdc9a21d7030b0f08ad72dff0c36b7ecb72f4f667a55a34b4f",
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
    expected = _SHA256.get(asset)
    if expected is not None:
        got = hashlib.sha256(blob).hexdigest()
        if got != expected:
            raise RuntimeError(
                f"bore {asset} checksum mismatch: expected {expected}, got {got} — "
                "refusing to run an unverified binary (tampered or corrupt download)")
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
