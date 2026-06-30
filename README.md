# DRIFT — Decentralized Routed Inference For Tokens

**DRIFT** is a server-less, peer-to-peer inference network: heterogeneous personal devices
(a Mac, a Windows PC, …) split **one** model by layer and run it **together**. Instead of
routing through a hyperscaler's datacenter, *your machine and someone else's* converge to
run a single AI.

The name is the system:

- **D — Decentralized:** no single controller, no single point of failure. Heterogeneous devices join as equal P2P nodes.
- **R — Routed:** an orchestrator *routes* hidden state through the nodes to carry inference forward (pipeline routing).
- **I — Inference:** the workload is LLM inference (extensible to training later).
- **For T — For Tokens:** the double meaning of "token." (1) the inference **token** — the atomic unit of machine thought/output; (2) the **token** of value — the economic unit you earn by contributing and spend on inference. DRIFT's vision is to make *the unit of thought and the unit of value one and the same*.

> *drift* also evokes computation **flowing** across scattered machines — thought that
> never pools in one place.

## Scope of this repository

This repo implements the **first slice — D·R·I, i.e. heterogeneous split inference — as a
working demo**, not the full vision. The "For Tokens" economic layer (trustless
verification, global P2P, token economy) is vision and *future work*, out of scope here.

A Mac (Apple GPU, PyTorch **MPS**, decoder layers `[0,k)`) and a Windows PC (NVIDIA GPU,
PyTorch **CUDA**, layers `[k,N)`) jointly run one LLM, exchanging hidden states over a
**framework-neutral byte protocol (TCP + msgpack)** — explicitly **not**
`torch.distributed`/NCCL.

## What makes it different

Exo binds node-to-node communication to MLX (`mx.distributed`), so it only works
*Apple-silicon-to-Apple-silicon* (Windows is "Longer term" on its roadmap). DRIFT lifts the
data plane into a **framework-neutral protocol** so that *different runtimes and different
GPU vendors* can run one model together. **A data plane bound to no framework is the core
contribution.**

## Documentation

- [`DRIFT-implementation-spec.md`](DRIFT-implementation-spec.md) — the authoritative technical blueprint (Korean): architecture, wire protocol, milestones M0–M6, debugging guide.
- [`docs/`](docs/) — the execution layer (English): phased implementation plan, build workflow, goal-execution plan, skills/MCP plan, parity-debugging playbook, M0 setup runbook. Start at [`docs/README.md`](docs/README.md).

## Status

Pre-implementation. The spec and execution docs are complete; the `drift/` code follows the
phased plan in [`docs/01-implementation-plan.md`](docs/01-implementation-plan.md), starting
with environment setup in [`docs/06-m0-setup-runbook.md`](docs/06-m0-setup-runbook.md).

## License

TBD
