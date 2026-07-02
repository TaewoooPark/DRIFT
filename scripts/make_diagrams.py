#!/usr/bin/env python3
"""Generate DRIFT's README diagrams as mono-black SVGs (wrapped in HTML for a
headless-Chrome PNG render). Hand-laid coordinates, one function per diagram.

Run:  python scripts/make_diagrams.py         # writes docs/img/_build/*.html + manifest
Then: scripts/render_diagrams.sh              # Chrome headless -> docs/img/*.png
"""
from __future__ import annotations

import html
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "img", "_build")

# ------------------------------------------------------------------ palette
BG      = "#000000"
CARD    = "#101013"
CARD_HI = "#16161a"
STROKE  = "#3a3a40"
INK     = "#f0f0f2"
BODY    = "#b6b6bc"
FAINT   = "#7c7c84"
WIRE    = "#9a9aa2"
ARROW   = "#84848c"
BUG_S, BUG_F, BUG_I = "#6e4a47", "#140f0e", "#c56a62"
FIX_S, FIX_F, FIX_I = "#48674f", "#0e1310", "#71b98d"

SANS = "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def txt(x, y, s, *, size=15, fill=BODY, font=SANS, weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')


def card(x, y, w, h, *, fill=CARD, stroke=STROKE, sw=1, rx=13):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def titled_card(x, y, w, h, title, lines, *, fill=CARD, stroke=STROKE,
                title_fill=INK, tsize=16):
    """lines: list of (text, kind) with kind in body|code|faint|strong."""
    out = [card(x, y, w, h, fill=fill, stroke=stroke)]
    out.append(txt(x + 18, y + 31, title, size=tsize, fill=title_fill, weight=600))
    ky = y + 31 + 26
    for s, kind in lines:
        if kind == "code":
            out.append(txt(x + 18, ky, s, size=13.5, fill=BODY, font=MONO))
        elif kind == "faint":
            out.append(txt(x + 18, ky, s, size=13, fill=FAINT))
        elif kind == "strong":
            out.append(txt(x + 18, ky, s, size=14.5, fill=INK, weight=600))
        else:
            out.append(txt(x + 18, ky, s, size=14, fill=BODY))
        ky += 23
    return "".join(out)


def arrow(x1, y1, x2, y2, *, color=ARROW, dashed=False, sw=1.6):
    dash = ' stroke-dasharray="5 5"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#ah)"{dash}/>')


def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">'
            f'<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7.5" refY="3" '
            f'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="{ARROW}"/></marker></defs>'
            f'<rect width="{w}" height="{h}" fill="{BG}"/>')


def label(x, y, s, *, size=12.5, fill=WIRE, anchor="middle", font=SANS):
    return txt(x, y, s, size=size, fill=fill, anchor=anchor, font=font)


# ------------------------------------------------------------------ D1 arch
def d1():
    W, H = 1180, 384
    s = [svg_open(W, H)]
    s.append(txt(40, 40, "Architecture — one model, split by layer", size=17, fill=INK, weight=600))
    s.append(txt(40, 62, "control / data / KV planes; only the residual stream crosses the boundary", size=13, fill=FAINT))

    # cards
    s.append(titled_card(40, 96, 300, 214, "Orchestrator · head", [
        ("tokenizer", "code"), ("embed_tokens", "code"),
        ("final norm  ·  lm_head", "code"), ("sampler / argmax", "code"),
        ("holds ~15% — embed + norm + head", "faint"),
    ]))
    for (bx, title, dev) in [(470, "ShardServer A · Mac", "device = mps"),
                             (840, "ShardServer B · Windows", "device = cuda")]:
        rng = "[0, 14)" if bx == 470 else "[14, 28)"
        s.append(titled_card(bx, 116, 300, 174, title, [
            (f"decoder layers {rng}", "code"), (dev, "code"),
            ("materializes only its slice", "strong"),
            ("per-session KV cache · self-RoPE", "faint"),
        ]))

    # forward arrows
    s.append(arrow(340, 150, 470, 150))
    s.append(label(405, 138, "hidden + pos + ids"))
    s.append(label(405, 168, "TCP · msgpack · fp16 · ~3 KB/tok", size=11, fill=FAINT))
    s.append(arrow(770, 203, 840, 203))
    s.append(label(805, 191, "hidden"))

    # return-to-head (star topology) — dashed sweep under the row
    s.append(f'<path d="M 985,290 C 985,352 720,356 190,356 L 190,314" fill="none" '
             f'stroke="{FAINT}" stroke-width="1.5" stroke-dasharray="5 5" marker-end="url(#ah)"/>')
    s.append(label(588, 375, "every hop returns to the head → norm · lm_head · argmax  (star topology)",
                   size=12, fill=FAINT))
    s.append("</svg>")
    return W, H, "".join(s)


# ------------------------------------------------------------------ D2 kv
def d2():
    W, H = 1180, 470
    s = [svg_open(W, H)]
    s.append(txt(40, 40, "The KV-cache indexing trap — and the fix", size=17, fill=INK, weight=600))
    s.append(txt(40, 62, "why a shard that keeps global layer_idx silently breaks after token 1", size=13, fill=FAINT))

    def chain(px, py, pw, header, hs, hf, hi, steps, last_strong):
        out = [card(px, py, pw, 372, fill="#0b0b0d", stroke=hs)]
        out.append(txt(px + 20, py + 34, header, size=15, fill=hi, weight=600))
        cy = py + 58
        cw = pw - 40
        for i, st in enumerate(steps):
            strong = last_strong and i == len(steps) - 1
            out.append(card(px + 20, cy, cw, 52, fill=hf, stroke=hs))
            out.append(txt(px + 38, cy + 31, st,
                           size=13.5, fill=(hi if strong else BODY),
                           font=MONO, weight=(600 if strong else 400)))
            if i < len(steps) - 1:
                out.append(arrow(px + pw / 2, cy + 52, px + pw / 2, cy + 70, color=hs, sw=1.8))
            cy += 70
        return "".join(out)

    s.append(chain(40, 84, 530, "✗  naïve — keep global layer_idx 14…27",
                   BUG_S, BUG_F, BUG_I, [
                       "cache slots 0…13  =  EMPTY",
                       "get_seq_length() reads slot 0 → 0",
                       "decode mask: 'no past tokens'",
                       "diverges after token 1",
                   ], last_strong=True))
    s.append(chain(610, 84, 530, "✓  DRIFT — re-index kept layers to local 0…13",
                   FIX_S, FIX_F, FIX_I, [
                       "self_attn.layer_idx : 14…27 → 0…13",
                       "cache slots 0…13 = this shard's KV",
                       "get_seq_length() → correct past",
                       "bitwise parity, prefill + decode",
                   ], last_strong=True))
    s.append("</svg>")
    return W, H, "".join(s)


# ------------------------------------------------------------------ D3 seq
def d3():
    W, H = 1180, 812
    lanes = {"O": 170, "A": 590, "B": 1000}
    s = [svg_open(W, H)]
    s.append(txt(40, 40, "The decode loop over an injectable transport", size=17, fill=INK, weight=600))
    s.append(txt(40, 62, "written once; only the transport (in-process / TCP) is swapped", size=13, fill=FAINT))

    heads = {"O": "Orchestrator", "A": "Shard A · mps · [0,14)", "B": "Shard B · cuda · [14,28)"}
    top = 88
    for k, cx in lanes.items():
        s.append(card(cx - 130, top, 260, 44))
        s.append(txt(cx, top + 28, heads[k], size=14, fill=INK, weight=600, anchor="middle"))
        s.append(f'<line x1="{cx}" y1="{top + 44}" x2="{cx}" y2="{H - 20}" '
                 f'stroke="{STROKE}" stroke-width="1"/>')

    def band(y, name):
        s.append(f'<rect x="30" y="{y}" width="{W - 60}" height="26" rx="6" '
                 f'fill="#0c0c0f" stroke="{STROKE}" stroke-width="1"/>')
        s.append(txt(44, y + 18, name, size=12.5, fill=FAINT, weight=600))

    def selfcall(k, y, text, side="right"):
        cx = lanes[k]
        s.append(f'<rect x="{cx - 4}" y="{y - 15}" width="8" height="8" rx="2" fill="{ARROW}"/>')
        if side == "left":
            s.append(txt(cx - 16, y - 8, text, size=12.5, fill=BODY, font=MONO, anchor="end"))
        else:
            s.append(txt(cx + 16, y - 8, text, size=12.5, fill=BODY, font=MONO))

    def msg(a, b, y, text, ret=False):
        x1, x2 = lanes[a], lanes[b]
        s.append(arrow(x1, y, x2, y, dashed=ret, color=(FAINT if ret else ARROW)))
        mid = (x1 + x2) / 2
        s.append(txt(mid, y - 8, text, size=12.5, fill=(FAINT if ret else WIRE),
                     font=MONO, anchor="middle"))

    band(140, "prefill  ·  whole prompt, positions 0…S-1")
    selfcall("O", 196, "h = embed_tokens(input_ids)")
    msg("O", "A", 230, 'forward(h, pos 0…S-1, "prefill")')
    selfcall("A", 264, "rotary(pos) · layers[0:14] · fill KV")
    msg("A", "O", 298, "h′", ret=True)
    msg("O", "B", 332, 'forward(h′, pos, "prefill")')
    selfcall("B", 366, "layers[14:28] · fill KV", side="left")
    msg("B", "O", 400, "h″", ret=True)
    selfcall("O", 434, "logits = lm_head(norm(h″[-1])) · argmax")

    band(462, "decode  ·  one token at a time, p = S, S+1, …   (loop until EOS / max_new_tokens)")
    selfcall("O", 518, "h = embed_tokens(next)")
    msg("O", "A", 552, 'forward(h, [p], "decode")')
    selfcall("A", 586, "rotary(p) · layers[0:14] · KV.append")
    msg("A", "O", 620, "h′", ret=True)
    msg("O", "B", 654, 'forward(h′, [p], "decode")')
    selfcall("B", 688, "layers[14:28] · KV.append", side="left")
    msg("B", "O", 722, "h″", ret=True)
    selfcall("O", 756, "next = argmax(lm_head(norm(h″))) · p += 1")
    s.append("</svg>")
    return W, H, "".join(s)


# ------------------------------------------------------------------ D4 gate
def d4():
    W, H = 1180, 430
    s = [svg_open(W, H)]
    s.append(txt(40, 40, "The parity gate — strict on one device, relaxed across two", size=17, fill=INK, weight=600))
    s.append(txt(40, 62, "correctness-first: the split must reproduce the single machine before any speed work", size=13, fill=FAINT))

    s.append(titled_card(40, 100, 300, 72, "M1 · reference oracle", [("full model · greedy", "faint")]))
    s.append(titled_card(40, 250, 300, 72, "split path · N shards", [("head + sliced nodes", "faint")]))

    # gate hexagon
    gx, gy, gw, gh = 430, 158, 210, 118
    cx, cy = gx + gw / 2, gy + gh / 2
    s.append(f'<path d="M {gx+34},{gy} L {gx+gw-34},{gy} L {gx+gw},{cy} '
             f'L {gx+gw-34},{gy+gh} L {gx+34},{gy+gh} L {gx},{cy} Z" '
             f'fill="{CARD_HI}" stroke="{STROKE}" stroke-width="1"/>')
    s.append(txt(cx, cy - 4, "compare", size=15, fill=INK, weight=600, anchor="middle"))
    s.append(txt(cx, cy + 18, "token ids", size=13, fill=BODY, anchor="middle"))

    s.append(arrow(340, 136, gx + 30, gy + 24))
    s.append(arrow(340, 286, gx + 30, gy + gh - 24))
    s.append(label(388, 150, "ids", size=11, fill=FAINT))
    s.append(label(388, 300, "ids", size=11, fill=FAINT))

    # outcomes
    s.append(titled_card(760, 92, 380, 66, "✓  same device → strict bitwise",
                         [("50 / 50 token ids identical", "faint")],
                         stroke=FIX_S, title_fill=FIX_I))
    s.append(titled_card(760, 176, 380, 66, "✓  across GPU vendors →  --prefix-match K",
                         [("first K match · later fp16 drift ok", "faint")],
                         stroke=FIX_S, title_fill=FIX_I))
    s.append(titled_card(760, 276, 380, 74, "✗  diverges early → bisect",
                         [("fp32 max-abs-diff + layer bisection", "faint")],
                         stroke=BUG_S, title_fill=BUG_I))

    s.append(arrow(640, cy - 34, 760, 125, color=FIX_S))
    s.append(arrow(640, cy, 760, 209, color=FIX_S))
    s.append(arrow(640, cy + 34, 760, 300, color=BUG_S))
    # bisect loops back
    s.append(f'<path d="M 950,350 C 950,395 190,398 190,322" fill="none" stroke="{BUG_S}" '
             f'stroke-width="1.5" stroke-dasharray="5 5" marker-end="url(#ah)"/>')
    s.append(label(560, 414, "localize the broken boundary, then re-run", size=12, fill=FAINT))
    s.append("</svg>")
    return W, H, "".join(s)


# ------------------------------------------------------------------ emit
def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for name, fn in [("arch", d1), ("kv-reindex", d2), ("decode-loop", d3), ("parity-gate", d4)]:
        w, h, body = fn()
        page = (f'<!doctype html><html><head><meta charset="utf-8">'
                f'<style>html,body{{margin:0;padding:0;background:{BG}}}</style></head>'
                f'<body>{body}</body></html>')
        with open(os.path.join(OUT, name + ".html"), "w") as f:
            f.write(page)
        manifest.append({"name": name, "w": w, "h": h})
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", len(manifest), "diagrams to", os.path.normpath(OUT))
    for m in manifest:
        print(f"  {m['name']}  {m['w']}x{m['h']}")


if __name__ == "__main__":
    main()
