#!/usr/bin/env python3
"""Assert that two embedding endpoints produce vectors in the same space.

Two independently-pooled or independently-quantized deployments of the same
model can differ enough that cosine similarity between them is meaningless --
and nothing errors when that happens, retrieval just quietly degrades. Run this
before trusting one endpoint to answer queries against an index built by another.

Usage:
  probe_compare.py <url_a> <url_b> [--key-a KEY] [--key-b KEY] [--threshold 0.999]

Both URLs are base URLs (no /v1/embeddings suffix). Exits non-zero if any probe
falls below the threshold.
"""
import argparse
import json
import math
import sys
import urllib.error
import urllib.request

# Deliberately mixed: short queries, long-ish passages, near-duplicates, an
# instruction-prefixed query, unicode/diacritics, and domain jargon. Pooling and
# quantization differences show up unevenly across input lengths, so a single
# short string is not enough to conclude two endpoints agree.
PREFIX = ("Instruct: Given a web search query, retrieve relevant passages that "
          "answer the query\nQuery: ")
PROBES = [
    "blueberry",
    "blueberry companions",
    "what should I plant near blueberries to deter pests",
    PREFIX + "what should I plant near blueberries to deter pests",
    PREFIX + "can I grow cucumbers near zucchini",
    "Blueberries demand strongly acidic soil (pH 4.5-5.5); ordinary garden soil "
    "will cause them to yellow and fail regardless of other care.",
    "Northern highbush blueberries (Vaccinium corymbosum) are the standard "
    "home-garden type, hardy to roughly zone 4, growing 4-6 feet tall.",
    "jalapeño peppers and piment d'Espelette",
    "mummy berry, spotted wing drosophila, Vaccinium virgatum, ericaceae",
    "Row covers are lightweight, spun-bonded polypropylene fabrics draped "
    "directly over crops or supported on hoops. They work by trapping heat "
    "radiating from the soil while allowing light and water to pass through. "
    "Medium weight (1.0-1.25 oz/yard) transmits about 70-85 percent of sunlight "
    "and provides 4-8 degrees of frost protection, the most popular weight for "
    "spring and fall season extension.",
    # A passage long enough to exercise a larger bucket / different code path.
    ("Tomatoes are heavy feeders requiring consistent moisture. " * 40),
]


def embed(base, texts, key=None):
    payload = json.dumps({"input": texts}).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base.rstrip("/") + "/v1/embeddings",
                                 data=payload, headers=headers)
    try:
        body = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        sys.exit(f"FAIL {base}: HTTP {e.code} {e.read()[:200]!r}")
    except Exception as e:
        sys.exit(f"FAIL {base}: {e}")
    return [row["embedding"] for row in body["data"]]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url_a")
    ap.add_argument("url_b")
    ap.add_argument("--key-a")
    ap.add_argument("--key-b")
    ap.add_argument("--threshold", type=float, default=0.999)
    args = ap.parse_args()

    # One input per request: batching can select a different bucket than a
    # caller would use in production, which is itself a source of divergence.
    va = [embed(args.url_a, [p], args.key_a)[0] for p in PROBES]
    vb = [embed(args.url_b, [p], args.key_b)[0] for p in PROBES]

    if len(va[0]) != len(vb[0]):
        sys.exit(f"FAIL dimension mismatch: {len(va[0])} vs {len(vb[0])} "
                 f"-- these are definitively different vector spaces")

    print(f"A: {args.url_a}")
    print(f"B: {args.url_b}")
    print(f"dims: {len(va[0])}   probes: {len(PROBES)}   threshold: {args.threshold}\n")

    sims = []
    worst = None
    for probe, a, b in zip(PROBES, va, vb):
        c = cos(a, b)
        sims.append(c)
        flag = "" if c >= args.threshold else "   <-- BELOW THRESHOLD"
        label = probe.replace("\n", " ")
        print(f"  {c:.6f}  {label[:64]!r}{flag}")
        if worst is None or c < worst[0]:
            worst = (c, label)

    print(f"\n  min  {min(sims):.6f}")
    print(f"  mean {sum(sims)/len(sims):.6f}")

    if min(sims) < args.threshold:
        print(f"\nRESULT: FAIL -- worst probe {worst[0]:.6f} ({worst[1][:50]!r})")
        print("These endpoints do NOT share a vector space. Do not mix them "
              "against one index; embed documents and queries with the same one.")
        return 1

    print("\nRESULT: PASS -- endpoints agree within threshold; safe to use "
          "interchangeably against one index.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
