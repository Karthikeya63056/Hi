"""Command line interface for Anvil PCAM Lab."""

from __future__ import annotations

import argparse
import json

import uvicorn

from anvil_pcam.core import PCAMEngine, benchmark_bank, evaluate_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="anvil-pcam",
        description="Anvil PCAM Lab: precision-controlled associative memory retrieval",
    )
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Launch the interactive Anvil PCAM dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8420)

    demo = sub.add_parser("demo", help="Run one noisy attractor retrieval trial")
    demo.add_argument("--pattern", default="A01", help="Stored attractor id")
    demo.add_argument("--sigma", type=float, default=0.58, help="Gaussian corruption level")
    demo.add_argument("--mask", type=float, default=0.28, help="Coordinate masking fraction")
    demo.add_argument("--seed", type=int, default=2404)
    demo.add_argument("--json", action="store_true", dest="json_output")

    bench = sub.add_parser("benchmark", help="Evaluate the attractor bank")
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--json", action="store_true", dest="json_output")

    args = parser.parse_args()
    engine = PCAMEngine()

    if args.command == "serve":
        uvicorn.run("anvil_pcam.web.app:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "demo":
        trial = evaluate_trial(engine, args.pattern, args.sigma, args.mask, args.seed)
        if args.json_output:
            print(json.dumps(trial, indent=2))
        else:
            _print_trial_summary(trial)
    elif args.command == "benchmark":
        result = benchmark_bank(engine, seed=args.seed)
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("Anvil PCAM attractor bank benchmark")
            print(f"identity Π accuracy : {result['baselineAccuracy']:.3f}")
            print(f"adaptive Π accuracy : {result['adaptiveAccuracy']:.3f}")
            print(f"identity Π mean cos : {result['meanBaselineScore']:.3f}")
            print(f"adaptive Π mean cos : {result['meanAdaptiveScore']:.3f}")
    else:
        parser.print_help()


def _print_trial_summary(trial: dict) -> None:
    target = trial["target"]
    metrics = trial["metrics"]
    print(f"Anvil PCAM noisy retrieval trial: {target['id']} / {target['label']}")
    print(f"corruption cosine to clean : {trial['noise']['cosineToClean']:.3f}")
    print(f"masked dimensions         : {len(trial['noise']['maskedDimensions'])}")
    print(f"identity Π final attractor: {trial['baseline']['finalId']} ({metrics['baseline']['finalTargetScore']:.3f})")
    print(f"adaptive Π final attractor: {trial['adaptive']['finalId']} ({metrics['adaptive']['finalTargetScore']:.3f})")
    print(f"Π anisotropy              : {metrics['adaptive']['anisotropy']:.3f}")


if __name__ == "__main__":
    main()
