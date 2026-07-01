"""Zero-config LAN discovery so nodes and the head find each other by themselves.

A `drift node` advertises itself over mDNS (`_drift._tcp.local.`); `drift run`
(without `--nodes`) browses for them. Entirely optional — if `zeroconf` is not
installed, `advertise()` is a no-op and `discover()` returns [], and the user
falls back to the explicit `--nodes host:port,…` list.
"""

from __future__ import annotations

import socket
import time

from .common import lan_ip

SERVICE = "_drift._tcp.local."

try:
    from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
    HAVE_ZEROCONF = True
except Exception:  # zeroconf not installed
    HAVE_ZEROCONF = False


def advertise(port: int, device: str, name: str | None = None):
    """Announce this node on the LAN. Returns an opaque handle for unadvertise()
    (or None if zeroconf is unavailable)."""
    if not HAVE_ZEROCONF:
        return None
    ip = lan_ip()
    label = name or f"drift-{ip.replace('.', '-')}-{port}"
    info = ServiceInfo(
        SERVICE,
        f"{label}.{SERVICE}",
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties={"device": device, "v": "1"},
        server=f"{label}.local.",
    )
    zc = Zeroconf()
    zc.register_service(info)
    return (zc, info)


def unadvertise(handle) -> None:
    if not handle:
        return
    zc, info = handle
    try:
        zc.unregister_service(info)
    finally:
        zc.close()


def discover(timeout: float = 3.0) -> list[dict]:
    """Browse the LAN for `drift node`s. Returns [{name,host,port,device}]."""
    if not HAVE_ZEROCONF:
        return []
    zc = Zeroconf()
    found: dict[str, dict] = {}

    def _resolve(name: str) -> None:
        info = zc.get_service_info(SERVICE, name, timeout=1500)
        if info and info.addresses:
            host = socket.inet_ntoa(info.addresses[0])
            props = info.properties or {}
            device = (props.get(b"device") or b"").decode() or "?"
            found[name] = {"name": name.split(".")[0], "host": host,
                           "port": info.port, "device": device}

    class _Listener:
        def add_service(self, zc_, type_, name):
            _resolve(name)

        def update_service(self, zc_, type_, name):
            _resolve(name)

        def remove_service(self, zc_, type_, name):
            found.pop(name, None)

    ServiceBrowser(zc, SERVICE, _Listener())
    try:
        time.sleep(timeout)
    finally:
        zc.close()
    return list(found.values())
