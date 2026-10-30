"""Certify an explicitly stored rational variant of any saved limiting result.

Use with new C4 discovery outputs. This never changes the source snapshot.
The existing verify command can independently recheck the resulting certificate.
"""

import argparse
import hashlib
import json
from pathlib import Path

import flint

from kuramoto_template.asymptotic_search.interval_certification import (
    rational_limiting_repair, verify_limiting,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory")
    args = parser.parse_args()
    source = args.input.resolve()
    relative = source.relative_to(ROOT).as_posix()
    raw = json.loads(source.read_text(encoding="utf-8"))
    saved = raw.get("result", raw)
    if saved.get("candidate") is None or saved.get("r") is None:
        parser.error("Input must contain a saved limiting candidate and correction")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(source_file=relative,
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   source_reported_mu=saved["mu0"], source_status=saved["status"],
                   original_snapshot_certified=False)
    try:
        data, changes = rational_limiting_repair(saved)
        checks = {str(bits): verify_limiting(data, precision_bits=bits) for bits in (192, 256, 320)}
        result = checks["256"]
        if any(check["density"] != result["density"] for check in checks.values()):
            raise ValueError("Inconsistent exact density across verification precisions")
        certificate = dict(schema_version=1, name="limiting-result", kind="limiting",
            precision_bits=256, arithmetic=f"python-flint {flint.__version__} / FLINT-Arb",
            source_file=relative, source_sha256=receipt["source_sha256"],
            source_reported_mu=saved["mu0"], exact_data=data, changes=changes,
            original_snapshot_attempt=dict(certified=False,
                reason="Only the explicitly stored nearby rational variant is certified."),
            result=result, graph_realization_certified=False)
        (args.output / "certificate.json").write_text(
            json.dumps(certificate, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        receipt.update(certified=True, density=result["density"], scope=result["scope"],
                       verified_bits=[192, 256, 320])
    except Exception as exc:
        receipt.update(certified=False, failure=f"{type(exc).__name__}: {exc}")
        if "data" in locals():
            (args.output / "failed-variant.json").write_text(
                json.dumps(dict(exact_data=data, changes=changes), indent=2, allow_nan=False) + "\n",
                encoding="utf-8")
        raise
    finally:
        (args.output / "summary.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, allow_nan=False))


if __name__ == "__main__":
    main()
