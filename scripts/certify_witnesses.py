"""Create or independently recheck exact-rational / Arb witness certificates."""

import argparse
import hashlib
import json
from pathlib import Path

import flint

from kuramoto_template.asymptotic_search.interval_certification import (
    attempt_original_finite, rational_finite_repair, rational_limiting_repair,
    verify_finite, verify_limiting,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("original-limit", "limiting", "research/joint-research-final/sweep/top-limit.json"),
    ("best-limit", "limiting", "research/s7-analysis/evidencia/fine-offset-1e-07--0.0001.json"),
    ("best-finite", "finite", "research/joint-research-final/finalist-refinement.json"),
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_certificate(path):
    certificate = read(path)
    if certificate["schema_version"] != 1:
        raise ValueError("Unknown certificate schema")
    source = ROOT / certificate["source_file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != certificate["source_sha256"]:
        raise ValueError(f"Changed original witness: {source}")
    bits = certificate["precision_bits"]
    if certificate["kind"] == "finite":
        result = verify_finite(certificate["exact_data"], precision_bits=bits,
                               box=certificate["result"]["proof"])
    elif certificate["kind"] == "limiting":
        result = verify_limiting(certificate["exact_data"], precision_bits=bits)
    else:
        raise ValueError("Unknown certificate kind")
    if result["density"] != certificate["result"]["density"]:
        raise ValueError("Certificate's claimed density differs from its exact data")
    return result


def create(output, bits):
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, kind, relative in CASES:
        source = ROOT / relative
        raw = read(source)
        if kind == "finite":
            saved = raw["best_finite"]
            original_attempt = attempt_original_finite(saved, precision_bits=bits)
            data, changes = rational_finite_repair(saved, precision_bits=bits)
            result = verify_finite(data, precision_bits=bits)
            source_mu = saved["mu"]
        else:
            saved = raw.get("result", raw)
            data, changes = rational_limiting_repair(saved)
            result = verify_limiting(data, precision_bits=bits)
            original_attempt = dict(certified=False,
                reason="Original float arrays have nonzero exact leading residuals; nearby rational repair is verified instead.")
            source_mu = saved["mu0"]
        certificate = dict(
            schema_version=1, name=name, kind=kind, precision_bits=bits,
            arithmetic=f"python-flint {flint.__version__} / FLINT-Arb",
            source_file=relative, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_reported_mu=source_mu, exact_data=data, changes=changes,
            original_snapshot_attempt=original_attempt, result=result,
            graph_realization_certified=False,
        )
        path = output / f"{name}.json"
        write(path, certificate)
        verify_certificate(path)
        results[name] = dict(density=result["density"], certified=True, scope=result["scope"],
                             original_snapshot_certified=False)
        print(name, result["density"]["lower"], result["scope"], flush=True)
    write(output / "summary.json", dict(results=results, precision_bits=bits,
                                        graph_realization_certified=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path, help="New directory for three certificates")
    group.add_argument("--verify", type=Path, help="Certificate JSON or a directory containing the three certificates")
    parser.add_argument("--precision-bits", type=int, default=256)
    args = parser.parse_args()
    if args.precision_bits < 192:
        parser.error("Use at least 192 bits for these fragile witnesses")
    if args.output:
        create(args.output, args.precision_bits)
    else:
        paths = [args.verify / f"{name}.json" for name, _, _ in CASES] if args.verify.is_dir() else [args.verify]
        for path in paths:
            result = verify_certificate(path)
            print(path.name, "verified", result["density"]["lower"])


if __name__ == "__main__":
    main()
