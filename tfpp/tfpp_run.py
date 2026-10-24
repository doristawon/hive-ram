# SPDX-License-Identifier: Apache-2.0
"""Run Flash Next on one GPU (TensorFold's path) or as one rank of a tfpp pipeline, and record the tokens.

  local : python tfpp_run.py --role local  --model DIR --out ref.json
  last  : python tfpp_run.py --role last   --model DIR --layers 32:48 --listen 5201
  leader: python tfpp_run.py --role leader --model DIR --layers 0:32 --peer 10.77.0.1:5201 --out pp.json

A middle rank takes --layers, --listen (for the rank before it) and --peer (the rank after it). The ranks may start
in any order: a rank links to the rank after it first (retrying until it answers, --connect-timeout) and only then
listens for the rank before it; the leader then checks the whole chain (contiguous layer ranges from 0 to the last
layer, the same checkpoint, capacity and windows; with --depth the last rank needs --depth too) before the first
prompt. Give every rank the same --depth and --capacity.

The rank link is meant for a private point-to-point link (see tfpp/wire.py): followers listen on --bind (default
TFPP_BIND or 0.0.0.0, which a WSL2-NAT node behind its Windows-side relay needs; elsewhere pass the link address).
Set the same TFPP_SECRET on every node to make both ends of every link authenticate. TFPP_IO_TIMEOUT (default 600 s,
0: none) bounds a wait inside a request.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from tfpp import proto

PROMPTS = [
    "請用繁體中文寫一段約 300 字的短文，描述颱風夜在台北公寓裡停電的情景。",
    "Write a Python function that parses an ISO 8601 duration string like 'P3DT4H5M' into seconds, with tests.",
    "用三點說明為什麼兩台筆電用 Thunderbolt 線串聯跑大型語言模型時，延遲比頻寬更重要。",
]


def chat(tok, text: str) -> list[int]:
    s = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tok.encode(s, add_special_tokens=False).ids


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Flash Next on one GPU, or one rank of a tfpp pipeline")
    ap.add_argument("--role", choices=["local", "leader", "middle", "last"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default=None, help="FIRST:END, this rank's layers (leader, middle, last)")
    ap.add_argument("--listen", type=int, default=5201, help="port a middle or last rank listens on")
    ap.add_argument("--bind", default=os.environ.get("TFPP_BIND", "0.0.0.0"),
                    help="address a middle or last rank listens on (default TFPP_BIND or 0.0.0.0)")
    ap.add_argument("--peer", default="", help="HOST:PORT of the next rank (leader, middle)")
    ap.add_argument("--connect-timeout", type=float, default=3600.0,
                    help="seconds to keep retrying the link to the next rank (and to wait for the chain's HELLO)")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--capacity", type=int, default=16384)
    ap.add_argument("--graphs", action="store_true", help="decode graphs (TensorFold's own on local, per-rank graphs in a pipeline)")
    ap.add_argument("--ple-on-ssd", action=argparse.BooleanOptionalAction, default=None,
                    help="read the n-gram (PLE) tables from disk (default: on for MLX checkpoints, off for NVFP4)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default="")
    return ap


def check_args(ap: argparse.ArgumentParser, a) -> None:
    """What can be refused before torch loads or any weight is read (argparse exits with status 2)."""

    if a.role != "local":
        if a.layers is None:
            ap.error(f"--role {a.role} needs --layers FIRST:END")
        try:
            a.first, a.end = proto.parse_layers(a.layers)
        except ValueError as exc:
            ap.error(str(exc))
    else:
        a.first = a.end = None
    if a.role in ("leader", "middle"):
        if not a.peer:
            ap.error(f"--role {a.role} needs --peer HOST:PORT (the next rank)")
        try:
            proto.parse_peer(a.peer)
        except ValueError as exc:
            ap.error(str(exc))
    elif a.peer:
        ap.error(f"--peer is for the leader and middle ranks, not --role {a.role}")
    if not 0 < a.listen < 65536:
        ap.error(f"--listen {a.listen}: not a port")
    if a.count < 1:
        ap.error("--count must be at least 1")
    if a.depth < 0:
        ap.error("--depth must be 0 or more")
    if a.capacity < 1:
        ap.error("--capacity must be positive")
    if a.repeat < 0:
        ap.error("--repeat must be 0 or more")
    if not 0.0 <= a.conf <= 1.0:
        ap.error("--conf must be between 0 and 1")
    if a.connect_timeout <= 0:
        ap.error("--connect-timeout must be positive")


def save(path: str, results: list) -> None:
    """The results so far, as UTF-8 JSON (rewritten after every prompt, replaced atomically)."""

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp, path)


def main() -> None:
    ap = parser()
    a = ap.parse_args()
    check_args(ap, a)

    import torch
    from tokenizers import Tokenizer

    from tfpp.load import load_range, model_layers
    from tfpp.wire import Link
    from tfpp import driver

    if a.role != "local":
        try:
            proto.check_role_range(a.role, a.first, a.end, model_layers(a.model))
        except ValueError as exc:
            ap.error(str(exc))
    ple = a.ple_on_ssd
    if ple is None:
        try:
            from tensorfold.families.qwen4_exp.cuda.weight_types import Config

            ple = Config.read(a.model).quant != "modelopt"     # NVFP4 tables stay memory-mapped
        except Exception:  # noqa: BLE001 - the loader reports an unreadable checkpoint itself
            ple = True
    torch.cuda.set_device(0)
    t0 = time.perf_counter()
    if a.role == "local":
        from tensorfold.families.qwen4_exp.cuda.weights import load
        w = load(a.model, mtp=a.depth > 0, draft_vocab="default" if a.depth > 0 else None, ple_on_ssd=ple)
        a.first, a.end = 0, w.cfg.layers
    else:
        w = load_range(a.model, a.first, a.end, mtp=a.role == "last" and a.depth > 0, ple_on_ssd=ple,
                       draft_vocab="default")
    free, total = torch.cuda.mem_get_info()
    print(f"[tfpp] {a.role} layers {a.first}:{a.end} loaded in {time.perf_counter() - t0:.1f}s; "
          f"GPU free {free / 2**30:.1f} of {total / 2**30:.1f} GiB", flush=True)

    if a.role == "local":
        from tensorfold.families.qwen4_exp.cuda.decode import Engine
        e = Engine(w, capacity=a.capacity, max_rows=max(8, a.depth + 1), graphs=a.graphs, kv_dtype="bf16")
        if e.graphs is not None:
            e.graphs.warm(a.depth + 1)
    else:
        e = driver.engine(w, a.depth, a.capacity, graphs=a.graphs, first=a.role == "leader",
                          last=a.role == "last")

    if a.role in ("middle", "last"):
        down = None
        if a.role == "middle":                           # link towards the last rank first: the chain forms from
            host, port = proto.parse_peer(a.peer)        # its end, so the ranks may start in any order
            down = Link.connect(host, port, name="down", connect_timeout=a.connect_timeout)
        up = Link.listen(a.listen, a.bind, name="up")
        print(f"[tfpp] {a.role} following ({up.describe()})", flush=True)
        try:
            driver.follow(e, up, down, depth=a.depth)
        except proto.RemoteError as exc:
            print(f"[tfpp] {a.role} stopped: {exc}", flush=True)
            sys.exit(1)
        print(f"[tfpp] {a.role} done: sent {up.sent / 2**20:.1f} MiB, received {up.received / 2**20:.1f} MiB",
              flush=True)
        return

    tok = Tokenizer.from_file(f"{a.model}/tokenizer.json")
    leader = None
    if a.role == "leader":
        host, port = proto.parse_peer(a.peer)
        link = Link.connect(host, port, name="down", connect_timeout=a.connect_timeout)
        print(f"[tfpp] linked to {link.describe()}", flush=True)
        leader = driver.Leader(e, link, depth=a.depth, timeout=a.connect_timeout)
    results = []
    try:
        for rep in range(a.repeat + 1):                      # the first pass warms the kernels and is not kept
            for i, text in enumerate(PROMPTS):
                ids = chat(tok, text)
                if leader is not None:
                    got = leader.generate(ids, a.count, a.depth, a.conf)
                else:
                    got = driver.local_generate(e, ids, a.count, a.depth, a.conf)
                got["prompt"] = i
                got["text"] = tok.decode(got["tokens"])
                if rep > 0 or a.repeat == 0:
                    results.append(got)
                    if a.out:
                        save(a.out, results)
                print(f"[tfpp] pass {rep} prompt {i}: {len(got['tokens'])} tokens, prefill {got['prefill_tps']:.0f} tok/s, "
                      f"decode {got['decode_tps']:.1f} tok/s, rounds {got['rounds']}, accepted {got['accepted']}/"
                      f"{got['drafted']} | {got['text'][:60]!r}", flush=True)
                if got.get("ms_per_round"):
                    print(f"[tfpp]   ms/round here {got['ms_per_round']} | peer {got.get('peer_ms_per_round')}", flush=True)
    except proto.RemoteError as exc:
        print(f"[tfpp] stopped: {exc}", flush=True)
        sys.exit(1)
    finally:
        if leader is not None:
            leader.quit()


if __name__ == "__main__":
    main()
