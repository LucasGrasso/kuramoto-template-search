"""Seeded random-policy smoke runner; no learned policy."""
import argparse
import json
from pathlib import Path
from . import KuramotoEnv, EnvironmentConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-sdp-calls", type=int, default=128)
    parser.add_argument("--max-clusters", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    config = EnvironmentConfig(max_sdp_calls=args.max_sdp_calls, max_clusters=args.max_clusters,
                               max_steps=10*args.max_sdp_calls)
    env = KuramotoEnv(config, output=args.output)
    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=args.seed+episode)
            reward_sum = 0.
            terminated = truncated = False
            while not (terminated or truncated):
                _, reward, terminated, truncated, info = env.step(env.sample_action())
                reward_sum += reward
            print(json.dumps(dict(episode=episode, reward=reward_sum, terminated=terminated,
                                  truncated=truncated, **info)), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
