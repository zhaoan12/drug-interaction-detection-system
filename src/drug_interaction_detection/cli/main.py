from __future__ import annotations

import argparse

from drug_interaction_detection.api.server import run_server
from drug_interaction_detection.config.settings import load_settings
from drug_interaction_detection.data.dataset import prepare_dataset, summarize_dataset
from drug_interaction_detection.evaluation.reporting import evaluate_bundle, render_markdown_report
from drug_interaction_detection.inference.service import InferenceEngine
from drug_interaction_detection.modeling.pipeline import train_models
from drug_interaction_detection.tracking.experiment import ExperimentRun, current_timestamp, make_run_name, record_run
from drug_interaction_detection.utils.io import read_json, write_json, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drug interaction detection system")
    parser.add_argument("--config", default="configs/default.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare-dataset")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--structure-only", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--structure-only", action="store_true")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--drug-a", required=True)
    predict_parser.add_argument("--drug-b", required=True)
    predict_parser.add_argument("--pair-id", default="cli-request")

    batch_parser = subparsers.add_parser("batch-predict")
    batch_parser.add_argument("--input", required=True)
    batch_parser.add_argument("--output", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--port", type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.command == "prepare-dataset":
        bundle = prepare_dataset(settings)
        write_json(settings.reports_dir / "dataset_summary.json", summarize_dataset(bundle))
        return
    if args.command == "train":
        bundle = prepare_dataset(settings)
        artifacts = train_models(settings, bundle, include_reference=not args.structure_only)
        result = evaluate_bundle(settings, bundle, artifacts, split="validation")
        record_run(
            settings.tracking_dir / "runs.jsonl",
            ExperimentRun(
                run_name=make_run_name("train"),
                started_at=current_timestamp(),
                params={"feature_mode": artifacts.feature_mode},
                metrics={
                    "interaction_accuracy": result.interaction_accuracy,
                    "severity_accuracy": result.severity_accuracy,
                },
                artifacts={"model": str(settings.artifacts_dir / f"model_{artifacts.feature_mode}.json")},
            ),
        )
        return
    if args.command == "evaluate":
        bundle = prepare_dataset(settings)
        full = train_models(settings, bundle, include_reference=not args.structure_only)
        result = evaluate_bundle(settings, bundle, full, split="test")
        ablation = None
        if not args.structure_only:
            ablation_artifacts = train_models(settings, bundle, include_reference=False)
            ablation = evaluate_bundle(settings, bundle, ablation_artifacts, split="test")
        report = render_markdown_report(result, ablation)
        report_path = settings.reports_dir / f"evaluation_{full.feature_mode}.md"
        report_path.write_text(report, encoding="utf-8")
        return
    if args.command == "predict":
        engine = InferenceEngine(settings)
        result = engine.predict(args.drug_a, args.drug_b, args.pair_id)
        print(result.model_dump_json(indent=2))
        return
    if args.command == "batch-predict":
        engine = InferenceEngine(settings)
        requests = read_json(args.input)
        outputs = [engine.predict(item["drug_a"], item["drug_b"], item.get("pair_id", "batch")).model_dump(mode="json") for item in requests]
        write_jsonl(settings.root / args.output, outputs)
        return
    if args.command == "serve":
        if args.port is not None:
            settings.settings.service.port = args.port
        run_server(settings)


if __name__ == "__main__":
    main()
