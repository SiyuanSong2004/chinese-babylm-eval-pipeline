"""
Chinese BabyLM Evaluation Pipeline
===================================
Usage:
    python pipeline.py download [--eval_dir DIR] [--tasks TASK ...] [--force-download]
    python pipeline.py eval     [--config FILE] [--results_dir DIR] [--tasks TASK ...] [--force-redo]
    python pipeline.py gather   [--config FILE_OR_DIR] [--results_dir DIR]
"""

import argparse
import json
import pathlib
import subprocess
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Task constants
# ─────────────────────────────────────────────────────────────────────────────

ALL_TASKS = [
    "zhoblimp", "hanzi_structure", "hanzi_pinyin",
    "word_fmri", "fmri",
    "afqmc", "ocnli", "tnews", "cluewsc2020",
]

TASK_CATEGORY = {
    "zhoblimp":        "zero_shot",
    "hanzi_structure": "zero_shot",
    "hanzi_pinyin":    "zero_shot",
    "word_fmri":       "cogbench",
    "fmri":            "cogbench",
    "afqmc":           "finetune",
    "ocnli":           "finetune",
    "tnews":           "finetune",
    "cluewsc2020":     "finetune",
}

# Maps eval task name to the prepare function key.
# word_fmri and fmri both share the "cogbench" prepare function.
TASK_TO_PREPARE_NAME = {
    "zhoblimp":        "zhoblimp",
    "hanzi_structure": "hanzi_structure",
    "hanzi_pinyin":    "hanzi_pinyin",
    "word_fmri":       "cogbench",
    "fmri":            "cogbench",
    "afqmc":           "afqmc",
    "ocnli":           "ocnli",
    "tnews":           "tnews",
    "cluewsc2020":     "cluewsc2020",
}

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning task specs
# ─────────────────────────────────────────────────────────────────────────────

FINETUNE_SPECS = {
    "afqmc":       {"num_labels": 2,  "metrics": ["accuracy", "f1", "mcc"]},
    "ocnli":       {"num_labels": 3,  "metrics": ["accuracy"]},
    "tnews":       {"num_labels": 15, "metrics": ["accuracy"]},
    "cluewsc2020": {"num_labels": 2,  "metrics": ["accuracy", "f1", "mcc"], "use_wsc_epochs": True},
}

# Zero-shot task → data directory name
ZERO_SHOT_DATA_DIRS = {
    "zhoblimp":       "zhoblimp",
    "hanzi_structure": "hanzi_structure",
    "hanzi_pinyin":   "hanzi_pinyin",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data / result existence checks
# ─────────────────────────────────────────────────────────────────────────────

def _data_exists(eval_dir, task):
    """Return True if the evaluation data for *task* already exists."""
    eval_path = pathlib.Path(eval_dir)
    if task in ZERO_SHOT_DATA_DIRS:
        d = eval_path / "full_eval" / ZERO_SHOT_DATA_DIRS[task]
        return d.is_dir() and any(d.iterdir())
    elif task in ("word_fmri", "fmri"):
        d = eval_path / "cogbench-fmri-0415"
        return d.is_dir() and any(d.iterdir())
    elif task in FINETUNE_SPECS:
        f = eval_path / "full_eval" / "clue" / f"{task}.train.jsonl"
        return f.exists()
    return False


def _result_exists(results_dir, model_stem, backend, task):
    """Return True if a result file for *task* / *model_stem* already exists."""
    category = TASK_CATEGORY.get(task)
    if category == "zero_shot":
        report = (
            pathlib.Path(results_dir)
            / model_stem / "main" / "zero_shot" / backend / task / task
            / "best_temperature_report.txt"
        )
        return report.exists()
    elif category == "cogbench":
        report = (
            pathlib.Path(results_dir)
            / model_stem / "main" / "cogbench" / task
            / f"cogbench_{task}_{model_stem}_report.json"
        )
        return report.exists()
    elif category == "finetune":
        result_file = (
            pathlib.Path(results_dir)
            / model_stem / "main" / "finetune" / task / "results.txt"
        )
        return result_file.exists()
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: download
# ─────────────────────────────────────────────────────────────────────────────

def cmd_download(args):
    from prepare_chinese_data import (
        prepare_zhoblimp,
        prepare_hanzi_structure,
        prepare_hanzi_pinyin,
        prepare_cogbench,
        prepare_afqmc,
        prepare_ocnli,
        prepare_tnews,
        prepare_cluewsc2020,
    )

    output_dir = pathlib.Path(args.eval_dir)
    print(f"Downloading / preparing evaluation data into: {output_dir}\n")

    prepare_map = {
        "zhoblimp":        prepare_zhoblimp,
        "hanzi_structure": prepare_hanzi_structure,
        "hanzi_pinyin":    prepare_hanzi_pinyin,
        "cogbench":        prepare_cogbench,
        "afqmc":           prepare_afqmc,
        "ocnli":           prepare_ocnli,
        "tnews":           prepare_tnews,
        "cluewsc2020":     prepare_cluewsc2020,
    }

    if args.tasks:
        force = args.force_download
        seen_prepare = set()
        for task in args.tasks:
            prepare_name = TASK_TO_PREPARE_NAME[task]
            if prepare_name in seen_prepare:
                continue
            seen_prepare.add(prepare_name)
            if not force and _data_exists(output_dir, task):
                print(f"  Skipping {prepare_name} (data already exists). "
                      f"Use --force-download to re-download.")
                continue
            print(f"  Preparing {prepare_name} ...")
            prepare_map[prepare_name](output_dir=output_dir)
    else:
        all_prepare = [
            ("zhoblimp",        prepare_zhoblimp),
            ("hanzi_structure", prepare_hanzi_structure),
            ("hanzi_pinyin",    prepare_hanzi_pinyin),
            ("cogbench",        prepare_cogbench),
            ("afqmc",           prepare_afqmc),
            ("ocnli",           prepare_ocnli),
            ("tnews",           prepare_tnews),
            ("cluewsc2020",     prepare_cluewsc2020),
        ]
        for name, fn in all_prepare:
            print(f"  Preparing {name} ...")
            fn(output_dir=output_dir)

    print("\nAll datasets prepared.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build subprocess commands
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd, task_label):
    print(f"\n=== Evaluating {task_label} ===")
    print("Command:", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"WARNING: command for '{task_label}' exited with "
            f"returncode {result.returncode}",
            file=sys.stderr,
        )


def _build_zero_shot_cmd(model_path, backend, task, eval_dir, results_dir):
    data_dir = ZERO_SHOT_DATA_DIRS[task]
    return [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", model_path,
        "--backend", backend,
        "--task", task,
        "--data_path", str(pathlib.Path(eval_dir) / "full_eval" / data_dir),
        "--output_dir", results_dir,
        "--save_predictions",
    ]


def _build_cogbench_cmd(model_path, backend, task, eval_dir, results_dir):
    return [
        sys.executable, "-m", "evaluation_pipeline.cogbench.run",
        "--model_path_or_name", model_path,
        "--backend", backend,
        "--task", task,
        "--data_path", str(pathlib.Path(eval_dir) / "cogbench-fmri-0415"),
        "--output_dir", results_dir,
        "--save_predictions",
    ]


def _build_finetune_cmd(model_path, backend, task, eval_dir, results_dir, hparams):
    spec = FINETUNE_SPECS[task]
    clue_dir = pathlib.Path(eval_dir) / "full_eval" / "clue"
    epochs = hparams["wsc_epochs"] if spec.get("use_wsc_epochs") else hparams["max_epochs"]

    cmd = [
        sys.executable, "-m", "evaluation_pipeline.finetune.run",
        "--model_name_or_path", model_path,
        "--train_data",   str(clue_dir / f"{task}.train.jsonl"),
        "--valid_data",   str(clue_dir / f"{task}.valid.jsonl"),
        "--predict_data", str(clue_dir / f"{task}.valid.jsonl"),
        "--task", task,
        "--num_labels", str(spec["num_labels"]),
        "--batch_size", str(hparams["batch_size"]),
        "--learning_rate", str(hparams["lr"]),
        "--num_epochs", str(epochs),
        "--sequence_length", str(hparams["sequence_length"]),
        "--results_dir", results_dir,
        "--metrics", *spec["metrics"],
        "--metric_for_valid", "accuracy",
        "--seed", str(hparams["seed"]),
    ]
    if backend == "causal":
        cmd += ["--causal", "--take_final"]
    elif backend.startswith("enc_dec"):
        cmd += ["--enc_dec"]
    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: collect results
# ─────────────────────────────────────────────────────────────────────────────

def _collect_zero_shot(results_dir, model_stem, backend, task):
    report = (
        pathlib.Path(results_dir)
        / model_stem / "main" / "zero_shot" / backend / task / task
        / "best_temperature_report.txt"
    )
    if not report.exists():
        return None
    lines = report.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if "### AVERAGE ACCURACY" in line or "### AVERAGE SPEARMAN'S RHO" in line:
            # Value is on the next non-empty line
            for j in range(i + 1, len(lines)):
                candidate = lines[j].strip()
                if candidate:
                    try:
                        return float(candidate)
                    except ValueError:
                        return None
    return None


def _collect_finetune(results_dir, model_stem, task):
    result_file = (
        pathlib.Path(results_dir)
        / model_stem / "main" / "finetune" / task / "results.txt"
    )
    if not result_file.exists():
        return None
    for line in result_file.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            if key.strip() == "accuracy":
                try:
                    score = float(val.strip())
                    return score * 100 if score <= 1.0 else score
                except ValueError:
                    return None
    return None


def _collect_cogbench(results_dir, model_stem, task):
    report = (
        pathlib.Path(results_dir)
        / model_stem / "main" / "cogbench" / task
        / f"cogbench_{task}_{model_stem}_report.json"
    )
    if not report.exists():
        return None
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        score = float(data["mean"])
        # Cogbench branch uses percentage display in the summary table.
        score *= 100.0
        return score
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(all_tasks, scores):
    """
    scores: dict[model_path] -> dict[task] -> float | None
    """
    model_paths = list(scores.keys())
    if not model_paths:
        print("No models found.")
        return

    model_stems = [pathlib.Path(p).name for p in model_paths]

    col_width = 16
    stem_width = max(24, *(len(s) for s in model_stems), len("Model"))

    header_cells = [f"{'Model':<{stem_width}}"] + [
        f"{t:>{col_width}}" for t in all_tasks
    ]
    header = " " + " ".join(header_cells)
    divider = "=" * len(header)

    print()
    print(divider)
    print(header)
    print(divider)

    for model_path in model_paths:
        stem = pathlib.Path(model_path).name
        row_cells = [f"{stem:<{stem_width}}"]
        for task in all_tasks:
            val = scores[model_path].get(task)
            if val is None:
                cell = "-"
            else:
                cell = f"{val:.2f}"
            row_cells.append(f"{cell:>{col_width}}")
        print(" " + " ".join(row_cells))

    print(divider)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load config
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path):
    """Load a single YAML config file and return the parsed dict.  Exits on error."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML is required. Install with: pip install pyyaml",
              file=sys.stderr)
        sys.exit(1)

    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: eval
# ─────────────────────────────────────────────────────────────────────────────

def cmd_eval(args):
    cfg = _load_config(args.config)

    models      = cfg.get("models", [])
    tasks_cfg   = cfg.get("tasks", {})
    eval_dir    = cfg.get("eval_dir", "evaluation_data")
    results_dir = args.results_dir if args.results_dir else cfg.get("results_dir", "results")

    hparams = cfg.get("finetune_hparams", {})
    hparams.setdefault("lr",         3e-5)
    hparams.setdefault("batch_size", 32)
    hparams.setdefault("max_epochs", 10)
    hparams.setdefault("wsc_epochs", 30)
    hparams.setdefault("sequence_length", 128)
    hparams.setdefault("seed",       42)

    # ── Determine task lists (--tasks overrides config) ──────────────────────
    if args.tasks:
        zero_shot_tasks = [t for t in args.tasks if TASK_CATEGORY[t] == "zero_shot"]
        cogbench_tasks  = [t for t in args.tasks if TASK_CATEGORY[t] == "cogbench"]
        finetune_tasks  = [t for t in args.tasks if TASK_CATEGORY[t] == "finetune"]
    else:
        zero_shot_tasks = tasks_cfg.get("zero_shot", [])
        cogbench_tasks  = tasks_cfg.get("cogbench",  [])
        finetune_tasks  = tasks_cfg.get("finetune",  [])

    all_tasks       = zero_shot_tasks + cogbench_tasks + finetune_tasks
    skip_existing   = bool(args.tasks)
    force_redo      = args.force_redo

    # ── Run evaluations ──────────────────────────────────────────────────────
    for model_entry in models:
        model_path = model_entry["path"]
        backend    = model_entry["backend"]
        stem       = pathlib.Path(model_path).name

        for task in zero_shot_tasks:
            if skip_existing and not force_redo and _result_exists(results_dir, stem, backend, task):
                print(f"\n=== Skipping {stem} on {task} (result already exists). "
                      f"Use --force-redo to re-evaluate. ===")
                continue
            cmd = _build_zero_shot_cmd(model_path, backend, task, eval_dir, results_dir)
            _run(cmd, f"{stem} on {task}")

        for task in cogbench_tasks:
            if skip_existing and not force_redo and _result_exists(results_dir, stem, backend, task):
                print(f"\n=== Skipping {stem} on {task} (result already exists). "
                      f"Use --force-redo to re-evaluate. ===")
                continue
            cmd = _build_cogbench_cmd(model_path, backend, task, eval_dir, results_dir)
            _run(cmd, f"{stem} on {task}")

        for task in finetune_tasks:
            if skip_existing and not force_redo and _result_exists(results_dir, stem, backend, task):
                print(f"\n=== Skipping {stem} on {task} (result already exists). "
                      f"Use --force-redo to re-evaluate. ===")
                continue
            cmd = _build_finetune_cmd(model_path, backend, task, eval_dir, results_dir, hparams)
            _run(cmd, f"{stem} on {task}")

    # ── Collect results ──────────────────────────────────────────────────────
    print("\nCollecting results ...")
    scores = {}
    for model_entry in models:
        model_path = model_entry["path"]
        backend    = model_entry["backend"]
        stem       = pathlib.Path(model_path).name
        scores[model_path] = {}

        for task in zero_shot_tasks:
            scores[model_path][task] = _collect_zero_shot(results_dir, stem, backend, task)

        for task in cogbench_tasks:
            scores[model_path][task] = _collect_cogbench(results_dir, stem, task)

        for task in finetune_tasks:
            scores[model_path][task] = _collect_finetune(results_dir, stem, task)

    _print_summary(all_tasks, scores)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: gather results for a single config
# ─────────────────────────────────────────────────────────────────────────────

def _gather_one_config(config_path, results_dir_override=None):
    """
    Load one YAML config and collect results for every model × every task.
    Returns a dict:  model_path -> { task -> float|None }
    and the results_dir that was used.
    """
    cfg = _load_config(config_path)

    models      = cfg.get("models", [])
    results_dir = results_dir_override if results_dir_override else cfg.get("results_dir", "results")

    scores = {}
    for model_entry in models:
        model_path = model_entry["path"]
        backend    = model_entry["backend"]
        stem       = pathlib.Path(model_path).name
        scores[model_path] = {}

        for task in ALL_TASKS:
            category = TASK_CATEGORY[task]
            if category == "zero_shot":
                scores[model_path][task] = _collect_zero_shot(results_dir, stem, backend, task)
            elif category == "cogbench":
                scores[model_path][task] = _collect_cogbench(results_dir, stem, task)
            elif category == "finetune":
                scores[model_path][task] = _collect_finetune(results_dir, stem, task)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard export
# ─────────────────────────────────────────────────────────────────────────────

# Metric key for each task category in the leaderboard JSON
_EXPORT_METRIC = {
    "zero_shot": "accuracy",
    "finetune":  "accuracy",
    "cogbench":  "mean",
}


def _build_leaderboard_entry(task_scores):
    """Convert a model's {task: score_0_100} dict to leaderboard JSON format.

    Scores are converted from the internal 0-100 scale to 0-1.
    Tasks with no result (None) are omitted.
    """
    entry = {}
    for task in ALL_TASKS:
        score = task_scores.get(task)
        if score is None:
            continue
        metric_key = _EXPORT_METRIC[TASK_CATEGORY[task]]
        entry[task] = {metric_key: score / 100.0}
    return entry


def _export_leaderboard(export_path, combined_scores, *, is_multi):
    """Write leaderboard-compatible JSON file(s).

    If *is_multi* is True (directory of configs / multiple models), one file
    per model is written with the model name appended to the base filename.
    """
    export_path = pathlib.Path(export_path)
    model_paths = list(combined_scores.keys())

    if not model_paths:
        return

    if is_multi and len(model_paths) > 1:
        stem = export_path.stem
        suffix = export_path.suffix or ".json"
        for model_path in model_paths:
            model_name = pathlib.Path(model_path).name
            out = export_path.with_name(f"{stem}_{model_name}{suffix}")
            entry = _build_leaderboard_entry(combined_scores[model_path])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
            print(f"  Exported: {out}")
    else:
        entry = _build_leaderboard_entry(combined_scores[model_paths[0]])
        export_path = export_path if export_path.suffix else export_path.with_suffix(".json")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(f"  Exported: {export_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: gather
# ─────────────────────────────────────────────────────────────────────────────

def cmd_gather(args):
    config_path = pathlib.Path(args.config)

    # Resolve list of config files: single file or every .yaml/.yml in a dir
    if config_path.is_dir():
        config_files = sorted(
            list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))
        )
        if not config_files:
            print(f"ERROR: no .yaml / .yml files found in {config_path}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(config_files)} config(s) in {config_path}/\n")
    elif config_path.is_file():
        config_files = [config_path]
    else:
        print(f"ERROR: {config_path} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    results_dir_override = args.results_dir if args.results_dir else None

    # Collect results across all configs (each may list one or more models)
    combined_scores = {}
    for cf in config_files:
        print(f"  Reading {cf.name} ...")
        per_config = _gather_one_config(cf, results_dir_override)
        # Merge into combined_scores. If the same model_path appears in
        # multiple configs, the later config's results win (unlikely in
        # practice since each config usually describes a different model).
        combined_scores.update(per_config)

    print(f"\nCollected results for {len(combined_scores)} model(s).")
    _print_summary(ALL_TASKS, combined_scores)

    if args.export:
        _export_leaderboard(args.export, combined_scores, is_multi=config_path.is_dir())


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: detect
# ─────────────────────────────────────────────────────────────────────────────

def cmd_detect(args):
    """Detect optimal sequence_length / batch_size for finetune tasks."""
    cfg = _load_config(args.config)

    models    = cfg.get("models", [])
    tasks_cfg = cfg.get("tasks", {})
    eval_dir  = cfg.get("eval_dir", "evaluation_data")

    percentiles  = args.percentiles
    max_bs       = args.max_batch_size

    # Determine finetune tasks ------------------------------------------------
    if args.tasks:
        finetune_tasks = [t for t in args.tasks if TASK_CATEGORY[t] == "finetune"]
        non_ft = [t for t in args.tasks if TASK_CATEGORY[t] != "finetune"]
        if non_ft:
            print(f"NOTE: skipping non-finetune tasks: {', '.join(non_ft)}")
    else:
        finetune_tasks = tasks_cfg.get("finetune", [])

    if not finetune_tasks:
        print("No finetune tasks to analyse.")
        return

    from pipeline_util import compute_token_lengths, find_max_batch_sizes

    clue_dir = pathlib.Path(eval_dir) / "full_eval" / "clue"

    # Probe every model × task combination ------------------------------------
    rows = []
    for model_entry in models:
        model_path = model_entry["path"]
        backend    = model_entry["backend"]
        stem       = pathlib.Path(model_path).name

        for task in finetune_tasks:
            train_data = str(clue_dir / f"{task}.train.jsonl")
            spec = FINETUNE_SPECS[task]

            print(f"  Analysing {stem} × {task} ...")

            # 1) Token lengths at each percentile
            lengths = compute_token_lengths(
                model_path, train_data, percentiles=tuple(percentiles),
            )

            # 2) Max batch size at each resulting sequence length
            unique_seq_lens = sorted(set(lengths[p] for p in percentiles))
            batch_map = find_max_batch_sizes(
                model_path, backend, unique_seq_lens, spec["num_labels"],
                max_batch_size=max_bs,
            )

            rows.append({
                "model":        stem,
                "task":         task,
                "lengths":      lengths,
                "batch_sizes":  {p: batch_map[lengths[p]] for p in percentiles},
            })

    # Print summary table -----------------------------------------------------
    if not rows:
        return

    model_w = max(16, *(len(r["model"]) for r in rows))
    task_w  = max(12, *(len(r["task"])  for r in rows))
    col_w   = 10

    header = f"  {'Model':<{model_w}}  {'Task':<{task_w}}"
    for p in percentiles:
        header += f"  {'p' + f'{p:g}':>{col_w}}  {'BS@p' + f'{p:g}':>{col_w}}"

    divider = "=" * len(header)
    print()
    print(divider)
    print(header)
    print(divider)

    for row in rows:
        line = f"  {row['model']:<{model_w}}  {row['task']:<{task_w}}"
        for p in percentiles:
            line += f"  {row['lengths'][p]:>{col_w}}  {row['batch_sizes'][p]:>{col_w}}"
        print(line)

    print(divider)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chinese BabyLM Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # ── download ─────────────────────────────────────────────────────────────
    dl_parser = subparsers.add_parser(
        "download",
        help="Prepare / download all evaluation datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dl_parser.add_argument(
        "--eval_dir",
        default="evaluation_data",
        help="Output directory for prepared evaluation data",
    )
    dl_parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALL_TASKS,
        default=None,
        metavar="TASK",
        help="Download only these tasks (default: all). "
             "Choices: %(choices)s",
    )
    dl_parser.add_argument(
        "--force-download",
        action="store_true",
        default=False,
        help="Re-download data even if it already exists",
    )
    dl_parser.set_defaults(func=cmd_download)

    # ── eval ─────────────────────────────────────────────────────────────────
    ev_parser = subparsers.add_parser(
        "eval",
        help="Run evaluations according to a YAML config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ev_parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to the YAML config file",
    )
    ev_parser.add_argument(
        "--results_dir",
        default=None,
        help="Override the results directory from the config",
    )
    ev_parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALL_TASKS,
        default=None,
        metavar="TASK",
        help="Override config to evaluate only these tasks. "
             "Skips tasks whose results already exist. "
             "Choices: %(choices)s",
    )
    ev_parser.add_argument(
        "--force-redo",
        action="store_true",
        default=False,
        help="Force re-evaluation even if results already exist "
             "(only meaningful with --tasks)",
    )
    ev_parser.set_defaults(func=cmd_eval)

    # ── gather ───────────────────────────────────────────────────────────────
    ga_parser = subparsers.add_parser(
        "gather",
        help="Collect and print evaluation results without running anything. "
             "Pass a single config file or a directory of configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ga_parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to a YAML config file, or a directory containing "
             "multiple .yaml/.yml config files",
    )
    ga_parser.add_argument(
        "--results_dir",
        default=None,
        help="Override the results directory from the config(s)",
    )
    ga_parser.add_argument(
        "--export", "-e",
        default=None,
        metavar="PATH",
        help="Export results to a JSON file compatible with the "
             "ChineseBabyLM 2026 Leaderboard submission format. "
             "With a directory of configs, one JSON per model is "
             "written (model name appended to the filename).",
    )
    ga_parser.set_defaults(func=cmd_gather)

    # ── detect ───────────────────────────────────────────────────────────────
    det_parser = subparsers.add_parser(
        "detect",
        help="Detect optimal sequence_length and batch_size for finetune "
             "tasks (prints a table — does not modify the config)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    det_parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to the YAML config file",
    )
    det_parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALL_TASKS,
        default=None,
        metavar="TASK",
        help="Only detect for these tasks (non-finetune tasks are ignored). "
             "Default: all finetune tasks listed in the config.",
    )
    det_parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=[95, 99, 100],
        metavar="P",
        help="Token-length percentiles to report",
    )
    det_parser.add_argument(
        "--max_batch_size",
        type=int,
        default=128,
        help="Upper limit for the batch-size search",
    )
    det_parser.set_defaults(func=cmd_detect)

    args = parser.parse_args()
    args.func(args)
    
if __name__ == "__main__":
    main()