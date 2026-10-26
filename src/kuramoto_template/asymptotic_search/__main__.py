"""uv run python -m kuramoto_template.asymptotic_search --help"""

import argparse
import json

from .models import Numerics
from .search import TwoStageConfig, run_two_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--config",help="JSON TwoStageConfig, including optional numerics")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--proposals",type=int,default=30)
    args = parser.parse_args()
    if args.config:
        from pathlib import Path
        data = json.loads(Path(args.config).read_text(encoding="utf-8"))
        data["numerics"] = Numerics(**data.get("numerics",{}))
        config = TwoStageConfig(**data)
    elif args.resume:
        from pathlib import Path
        data = json.loads((Path(args.output)/"checkpoint.json").read_text(encoding="utf-8"))["identity"]["config"]
        data["numerics"] = Numerics(**data["numerics"])
        config = TwoStageConfig(**data)
    else:
        config = TwoStageConfig(rng_seed=args.seed,stage1_proposals=args.proposals)
    result = run_two_stage(config,output=args.output,resume=args.resume)
    print(json.dumps(dict(best_finite_numeric_mu=result.best_finite_numeric_mu,
                          calls=result.call_counts,stop_reason=result.stop_reason)))


if __name__ == "__main__":
    main()
