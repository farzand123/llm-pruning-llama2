"""Command-line interface:  ``llmprune <command> --config configs/full.yaml``.

Commands that only read files (``plan``, ``status``, ``report``) need no GPU stack; the others import
PyTorch lazily.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .config import describe, expand_conditions, load_config, parse_shard, select_conditions
from .io_utils import package_versions
from .status import STAGES, condition_status, plan_actions

log = logging.getLogger("llmprune")

# Environment of thesis Table 10 (NumPy: see requirements.txt for why 1.26.4).
PINNED = {"torch": "2.2.0", "transformers": "4.38.2", "peft": "0.10.0", "accelerate": "0.27.0",
          "datasets": "2.18.0", "lm_eval": "0.4.2", "numpy": "1.26.4"}


# --------------------------------------------------------------------------- argument helpers
def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="YAML configuration, e.g. configs/full.yaml or configs/quick.yaml")
    p.add_argument("--only", nargs="+", metavar="ID",
                   help="condition ids, glob patterns such as '7b-heads-*', or Table 17 numbers")
    p.add_argument("--shard", metavar="K/N", help="run only conditions with position %% N == K (one shard per GPU)")
    p.add_argument("--include-disabled", action="store_true", help="also select conditions marked 'Not reported' in Table 17")


def _selected(args):
    cfg = load_config(args.config)
    conds = select_conditions(expand_conditions(cfg), only=args.only, shard=parse_shard(args.shard),
                              include_disabled=args.include_disabled)
    if not conds:
        raise SystemExit("no condition matches the selection (see `llmprune plan`)")
    return cfg, conds


def _parse_stages(text: Optional[str]) -> List[str]:
    if not text:
        return list(STAGES)
    stages = [s.strip() for s in text.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {list(STAGES)}")
    return stages


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", force=True)
    for noisy in ("urllib3", "filelock", "fsspec", "datasets", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- commands
def cmd_plan(args) -> int:
    cfg, conds = _selected(args)
    print(describe(conds))
    n_lora = sum(1 for c in conds if c.lora_rank and c.enabled)
    n_recipe = len({c.recipe_key for c in conds if c.recipe_key})
    evals = {k: sum(1 for c in conds if k in c.evals) for k in ("ppl", "glue", "mmlu")}
    n_bench = sum(len(cfg["efficiency"]["settings"]) for c in conds if c.efficiency)
    print(f"\n{len(conds)} conditions: {n_recipe} pruning recipes, {n_lora} LoRA trainings, "
          f"evaluations {evals}, {n_bench} efficiency benchmarks")
    return 0


def cmd_run(args) -> int:
    cfg, conds = _selected(args)
    stages = _parse_stages(args.stages)
    if args.dry_run:
        print(describe(conds), "\n")
        for c in conds:
            plan = plan_actions(cfg, c, stages, args.force)
            todo = [s for s, v in plan["run"].items() if v]
            detail = []
            if plan["eval_kinds"]:
                detail.append("eval: " + ",".join(plan["eval_kinds"]))
            if plan["efficiency_settings"]:
                detail.append("efficiency: " + ",".join(plan["efficiency_settings"]))
            print(f"{c.id:<32} " + ("-> " + ", ".join(todo) + (f"  ({'; '.join(detail)})" if detail else "") if todo else "complete"))
        return 0
    from .pipeline import run_matrix

    failed = run_matrix(cfg, conds, stages, force=args.force, continue_on_error=args.continue_on_error)
    if failed:
        log.error("%d condition(s) failed: %s", len(failed), ", ".join(failed))
        return 1
    log.info("done")
    return 0


def cmd_status(args) -> int:
    cfg, conds = _selected(args)
    rows, n_incomplete = [], 0
    for c in conds:
        st = condition_status(cfg, c)
        cells, bad = [], False
        for stage, label in (("recipe", "recipe"), ("lora", "lora"), ("eval", "eval"), ("efficiency", "eff")):
            total = len(st[stage]["present"]) + len(st[stage]["missing"])
            if total == 0:
                continue
            ok = len(st[stage]["present"]) - len(st[stage]["stale"])
            cells.append(f"{label} {ok}/{total}" + (f" (STALE {len(st[stage]['stale'])})" if st[stage]["stale"] else ""))
            bad = bad or ok < total
        n_incomplete += bad
        if bad or not args.missing:
            rows.append(f"{c.no:>3}  {c.id:<32} " + ("  ".join(cells) if cells else "nothing reported in Table 17"))
    print("\n".join(rows) if rows else "everything selected is complete")
    print(f"\n{len(conds) - n_incomplete} of {len(conds)} conditions complete")
    return 0


def cmd_report(args) -> int:
    from .report import write_report

    cfg = load_config(args.config)
    out = Path(args.out) if args.out else Path("reports") / Path(args.config).stem
    files = write_report(cfg, out)
    print(f"wrote {len(files)} files to {out}/  (start with REPORT.md and comparison_with_thesis.csv)")
    return 0


def cmd_aggregate(args) -> int:
    from .replicates import write_replicate_report

    compare = args.compare if args.compare else None
    files = write_replicate_report(args.configs, Path(args.out), compare=compare, metric=args.metric)
    print(f"wrote {len(files)} files to {args.out}/  (start with REPLICATES.md)")
    return 0


def cmd_check(args) -> int:
    ok = True
    print(f"llmprune {__version__}, Python {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 10):
        print("  note: the thesis environment (Table 10) used Python 3.10")
    versions = package_versions(list(PINNED))
    for name, want in PINNED.items():
        have = versions.get(name)
        state = "MISSING" if have is None else ("ok" if have.split("+")[0] == want else f"differs (thesis: {want})")
        ok = ok and have is not None
        print(f"  {name:<14} {have or '-':<14} {state}")
    np_v, torch_v = versions.get("numpy"), versions.get("torch")
    if np_v and torch_v and np_v.split(".")[0] == "2" and torch_v.startswith("2.2"):
        print("  WARNING: torch 2.2 was built against NumPy 1.x; install numpy==1.26.4")
    try:
        import torch

        print(f"  CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"    cuda:{i} {p.name}, {p.total_memory / 1e9:.0f} GB")
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            print("  WARNING: this GPU does not support bfloat16 natively")
    except ImportError:
        print("  PyTorch is not installed: plan / status / report still work, nothing else does")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or (Path.home() / ".cache/huggingface/token").is_file()
    print(f"  Hugging Face token found: {bool(token)}  (needed for the gated meta-llama/Llama-2 models)")
    print(f"  free disk in the working directory: {shutil.disk_usage('.').free / 1e9:.0f} GB "
          "(recipes are tiny, adapters 0.07-0.3 GB each, model downloads ~13 GB (7B) and ~26 GB (13B))")
    return 0 if ok else 1


def cmd_selftest(args) -> int:
    from .selftest import run_all

    return 0 if run_all(verbose=True) else 1


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmprune", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=f"llmprune {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="print the experiment matrix (thesis Table 17) and what it implies")
    _add_selection(p)
    p.set_defaults(func=cmd_plan)

    def run_like(name: str, help_text: str, stages: Optional[str]):
        q = sub.add_parser(name, help=help_text)
        _add_selection(q)
        if stages is None:
            q.add_argument("--stages", help=f"comma-separated subset of {','.join(STAGES)} (default: all)")
        else:
            q.set_defaults(stages=stages)
        q.add_argument("--force", action="store_true", help="recompute the selected stages and everything built on them")
        q.add_argument("--dry-run", action="store_true", help="show what would run and exit")
        q.add_argument("--continue-on-error", action="store_true", help="go on with the next condition after a failure")
        q.set_defaults(func=cmd_run)

    run_like("run", "run (or resume) the selected conditions", None)
    run_like("prune", "only compute the pruning recipes", "recipe")
    run_like("lora", "only train the LoRA adapters (computes missing recipes first)", "lora")
    run_like("evaluate", "only run the evaluations (perplexity / MMLU / GLUE)", "eval")
    run_like("benchmark", "only run the efficiency benchmark", "efficiency")

    p = sub.add_parser("status", help="show which outputs exist, which are missing or stale")
    _add_selection(p)
    p.add_argument("--missing", action="store_true", help="list only incomplete conditions")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("report", help="regenerate the thesis tables, figures and the comparison with the printed values")
    p.add_argument("--config", required=True)
    p.add_argument("--out", help="output directory (default: reports/<config name>)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("aggregate", help="mean, SD, 95%% CI (and a paired test) over replicate runs with different seeds")
    p.add_argument("--configs", nargs="+", required=True, metavar="YAML",
                   help="one config per seed (each with its own seed and output folders, see docs/REPRODUCING.md)")
    p.add_argument("--out", default="reports/replicates")
    p.add_argument("--compare", nargs=2, metavar=("ID_A", "ID_B"), help="paired t-test of ID_A - ID_B over the shared seeds")
    p.add_argument("--metric", default="ppl", help="metric of --compare (ppl, mmlu, glue, tokens_per_s[single], ...)")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("check", help="check the environment against the thesis' pinned versions")
    p.add_argument("--config")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("selftest", help="verify head/layer/magnitude pruning, LoRA and the evaluation code on a tiny random model (no downloads)")
    p.set_defaults(func=cmd_selftest)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
