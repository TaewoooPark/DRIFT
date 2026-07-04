"""`drift` — one command in front of the whole toolkit.

`drift <command>` dispatches to the module that implements it, so users type
`drift doctor` / `drift up 2` / `drift run` instead of remembering
`python -m drift.<module>` incantations. New commands are added as their modules land.
"""

from __future__ import annotations

import importlib
import sys

# command -> (module, function, args prepended before the user's)
_COMMANDS: dict[str, tuple[str, str, list[str]]] = {
    "doctor":    ("drift.doctor", "main", []),
    "up":        ("drift.run", "up_main", []),
    "run":       ("drift.run", "main", []),
    "node":      ("drift.node", "main", []),
    "keygen":    ("drift.crypto", "main", []),
    "reference": ("drift.reference", "main", []),
    "parity":    ("drift.parity_test", "main", []),
    "itest":     ("drift.itest", "main", []),
    "bench":     ("drift.bench", "main", []),
    "ping":      ("drift.orchestrator", "main", ["--ping"]),
}

_HELP = """drift — run one model split across your machines, no datacenter.

usage: drift <command> [options]

getting started:
  doctor        preflight environment & config check (run this first)
  up [N]        localhost: spawn N nodes, auto-split the model, and chat/generate
  node          run THIS machine as a worker (auto device/port, LAN-reachable)
  run           head: assign layers to running nodes and chat/generate
                  (--nodes host:port,… ; omit --prompt for an interactive chat)
  keygen        create/print the network key + node identity (encrypt the wire)

commands:
  reference     single-machine reference generation (the oracle)
  parity        bitwise parity gate   (--mode inprocess|socket, --selftest)
  itest         integration gate over real nodes (--nodes N [--chain])
  bench         benchmarks            (fidelity / footprint / wire / overhead)
  ping          health-check configured shards

run `drift <command> --help` for a command's options.
"""


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_HELP)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in _COMMANDS:
        print(f"unknown command: {cmd!r}\n")
        print(_HELP)
        return 2
    modname, fnname, prepend = _COMMANDS[cmd]
    fn = getattr(importlib.import_module(modname), fnname)
    return fn(prepend + rest)


if __name__ == "__main__":
    sys.exit(main())
