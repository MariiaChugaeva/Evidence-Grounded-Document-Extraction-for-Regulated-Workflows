"""Run the LLM diagnostic baseline.

Usage: python -m scripts.run_llm [--model qwen2.5:7b] [--layer native|ocr]
       [--layout rows|linear] [--split dev] [--limit N] [--dry-run DIR] [--no-cache]

Default backend is a local Ollama server (no API key). ``claude-*`` model
names still use the Anthropic SDK and ANTHROPIC_API_KEY. With --dry-run DIR
no request is made; the prompts are written to DIR for inspection. Every
real call is cached under data/llm_cache, so a second run costs nothing and
reproduces the same predictions.

The run reports twice. The prediction file holds the *repaired* output that
the evaluator scores, and data/results/<system>_raw_contract.csv plus the
printed table hold every contract breach the *raw* output committed - invalid
schema, a derived claim without its rule, an incomplete premise set, a quote
that is not on its page. A headline joint score is only meaningful next to
that table.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from sdsbench import annotations, textlayer
from sdsbench.extractors import llm
from sdsbench.schema import PredictionSet

OUTPUT_DIR = Path("data/predictions")
RESULTS_DIR = Path("data/results")


def write_contract_report(run: llm.LlmRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["document", "field", "code", "description", "detail"])
        for violation in run.violations:
            writer.writerow(
                [
                    violation.document,
                    violation.field,
                    violation.code,
                    llm.CONTRACT_DESCRIPTIONS.get(violation.code, ""),
                    violation.detail,
                ]
            )


def print_contract_report(run: llm.LlmRun, path: Path) -> None:
    counts = run.violation_counts()
    total = sum(counts.values())
    print()
    print("raw output vs the prediction contract (counted before repair)")
    print(f"{'code':34}{'claims':>8}  description")
    print("-" * 78)
    for code, count in counts.items():
        if count:
            print(f"{code:34}{count:>8}  {llm.CONTRACT_DESCRIPTIONS.get(code, '')}")
    print("-" * 78)
    print(
        f"{'total':34}{total:>8}  in {run.documents_with_violations()} of "
        f"{len(run.calls)} documents"
    )
    print(f"per-claim detail -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=llm.DEFAULT_MODEL)
    parser.add_argument("--layer", choices=[textlayer.NATIVE, textlayer.OCR], default=textlayer.NATIVE)
    parser.add_argument("--layout", choices=["rows", "linear"], default="rows")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--allow-locked", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", type=Path, default=None, help="write prompts here instead of calling the model")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--num-ctx", type=int, default=llm.DEFAULT_NUM_CTX)
    parser.add_argument("--ollama-host", default=None, help="Ollama base URL, default http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    from scripts.evaluate import guard_split

    gold = annotations.load_gold(split=args.split)
    guard_split(gold, args.allow_locked)
    documents = [item.document for item in gold][: args.limit]

    if args.dry_run is not None:
        args.dry_run.mkdir(parents=True, exist_ok=True)
        for document in documents:
            system, user = llm.build_messages(document, textlayer.load(document, args.layer), args.layout)
            with open(args.dry_run / f"{Path(document).stem}-{args.layer}-{args.layout}.txt", "w", encoding="utf-8") as handle:
                handle.write("=== SYSTEM ===\n" + system + "\n\n=== USER ===\n" + user + "\n")
        schema = llm.LlmOutput.model_json_schema()
        with open(args.dry_run / "output_schema.json", "w", encoding="utf-8") as handle:
            json.dump(schema, handle, indent=2)
        print(f"dry run: wrote {len(documents)} prompts and output_schema.json -> {args.dry_run}")
        return

    run = llm.LlmRun(model=args.model, layer=args.layer, layout=args.layout)
    for document in documents:
        prediction, record = llm.extract_document(
            document,
            args.model,
            args.layer,
            args.layout,
            use_cache=not args.no_cache,
            num_ctx=args.num_ctx,
            host=args.ollama_host,
        )
        run.documents.append(prediction)
        run.calls.append(record)
        status = "cached" if record.cached else f"{record.latency_seconds:.1f}s, ${record.cost_usd:.3f}"
        print(f"{document}: {status}{'  ERROR ' + record.error if record.error else ''}")

    safe_model = args.model.replace(":", "-").replace("/", "-")
    system = f"llm-{safe_model}-{args.layer}-{args.layout}"
    suffix = "" if args.split == "dev" else f"-{args.split}"
    prediction_set = PredictionSet(
        system=system,
        configuration={
            "layer": args.layer,
            "layout": args.layout,
            "extractor": f"llm:{args.model}",
            "backend": "anthropic" if llm.is_anthropic_model(args.model) else "ollama",
            "num_ctx": str(args.num_ctx),
            "split": args.split,
            "prompt_sha": llm.cache_key(args.model, llm.SYSTEM_PROMPT, "")[:12],
        },
        documents=run.documents,
    )
    path = args.output / f"{system}{suffix}.json"
    prediction_set.save(path)
    with open(args.output / f"{system}{suffix}.meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": args.model,
                "documents": len(run.documents),
                "total_cost_usd": round(run.total_cost_usd, 4),
                "prediction_id": prediction_set.prediction_id,
                "raw_contract_violations": run.violation_counts(),
                "documents_with_raw_violations": run.documents_with_violations(),
                # Per-claim violations live in the contract CSV, not in the cost log.
                "calls": [
                    {
                        key: value
                        for key, value in asdict(call).items()
                        if key != "violations"
                    }
                    for call in run.calls
                ],
            },
            handle,
            indent=2,
        )
    errors = sum(1 for call in run.calls if call.error)
    print(f"{system}: {len(run.documents)} documents, {errors} errors, "
          f"estimated cost ${run.total_cost_usd:.2f} -> {path}")

    report_path = RESULTS_DIR / f"{system}{suffix}_raw_contract.csv"
    write_contract_report(run, report_path)
    print_contract_report(run, report_path)


if __name__ == "__main__":
    main()
