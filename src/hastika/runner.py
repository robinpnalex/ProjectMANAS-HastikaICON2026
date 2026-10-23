"""Plan or execute named final-system recipes, one TAPT group at a time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from hastika.common.paths import PROJECT_ROOT, RAW_DATA_DIR
from hastika.common.tasks import FILES
from hastika.common.validation import read_frame
from hastika.data.external import prepare_external
from hastika.data.splits import prepare, sha256
from hastika.evaluation.ensemble import combine, read_bundle


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    if recipe["task"] not in FILES or recipe["mode"] not in ("full", "holdout"):
        raise ValueError("Unknown task or training mode")
    groups = recipe["groups"]
    ids = [group["id"] for group in groups]
    if len(ids) != len(set(ids)) or any(not name.replace("_", "").isalnum() for name in ids):
        raise ValueError("Group names must be unique simple identifiers")
    for group in groups:
        seeds = group["classifier_seeds"]
        if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("Invalid classifier seeds")
        if group["rdrop"] < 0:
            raise ValueError("R-Drop weight must be non-negative")
    if sum(len(g["classifier_seeds"]) for g in groups) != recipe["member_count"]:
        raise ValueError("Recipe member count does not match groups")
    return recipe


def recipe_hash(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def commands(recipe, out, raw_dir, selected=None):
    out, raw_dir = Path(out).resolve(), Path(raw_dir).resolve()
    task, mode = recipe["task"], recipe["mode"]
    train = out / "data" / ("all.csv" if mode == "full" else "train.csv")
    predict = raw_dir / FILES[task]["test"] if mode == "full" else out / "data/holdout.csv"
    revision = ["--revision", recipe["model_revision"]] if recipe.get("model_revision") else []
    jobs = []
    for group in recipe["groups"]:
        if selected and group["id"] not in selected:
            continue
        directory = out / group["id"]
        adapter = directory / "tapt"
        tapt = [sys.executable, "-m", "hastika.models.llm_tapt", "--model", recipe["model"], *revision,
                "--texts", str(train), str(out / "external_text.csv"), "--out", str(adapter),
                "--epochs", str(recipe["tapt_epochs"]), "--seed", str(group["tapt_seed"]),
                "--lr", str(recipe["lr"]), "--r", str(recipe["lora_rank"]),
                "--bs", str(recipe["tapt_bs"]), "--grad-accum", str(recipe["tapt_grad_accum"]),
                "--max-len", str(recipe["tapt_max_len"])]
        jobs.append({"group": group["id"], "stage": "tapt", "command": tapt})
        for seed in group["classifier_seeds"]:
            cmd = [sys.executable, "-m", "hastika.models.llm_classifier", "--model", recipe["model"], *revision,
                   "--task", task, "--train", str(train), "--predict", str(predict),
                   "--out", str(directory / f"s{seed}"), "--init-adapter", str(adapter), "--seed", str(seed)]
            flags = {"epochs": "classifier_epochs", "lr": "lr", "head-lr": "head_lr", "r": "lora_rank",
                     "prompt": "prompt", "class-weight": "class_weight", "label-smoothing": "label_smoothing",
                     "bs": "bs", "grad-accum": "grad_accum", "max-len": "max_len"}
            for flag, key in flags.items():
                cmd += ["--" + flag, str(recipe[key])]
            cmd += ["--rdrop", str(group["rdrop"])]
            if mode == "holdout":
                cmd += ["--eval", str(predict)]
            jobs.append({"group": group["id"], "stage": "classifier", "requested_seed": seed, "command": cmd})
    return jobs


def launch(command, log):
    print("Running:", subprocess.list2cmdline(command), flush=True)
    with open(log, "w") as handle:
        code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
    if code:
        print(f"Exit {code}; see {log}", flush=True)
    return code


def assemble(recipe, out):
    prefixes = []
    for group in recipe["groups"]:
        prefix = Path(out) / group["id"] / "group"
        _, _, metadata = read_bundle(prefix, recipe["task"])
        if metadata["provenance"].get("recipe_hash") != recipe_hash(recipe):
            raise ValueError(f"{group['id']}: bundle belongs to another recipe or training mode")
        if metadata["member_count"] != len(group["classifier_seeds"]):
            raise ValueError("Incomplete group")
        prefixes.append(prefix)
    return combine(prefixes, recipe["task"], Path(out) / "final", recipe["member_count"])


def execute(recipe, out, raw_dir, external, selected):
    out, raw_dir, external = Path(out).resolve(), Path(raw_dir).resolve(), Path(external).resolve()
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "recipe.json"
    if marker.exists() and json.loads(marker.read_text()) != recipe:
        raise ValueError("Output directory already contains a different recipe")
    marker.write_text(json.dumps(recipe, indent=2) + "\n")
    report = prepare(recipe["task"], raw_dir, out / "data")
    source_manifest = json.loads((PROJECT_ROOT / "data/sources.json").read_text())
    if sha256(external) != source_manifest["external"]["sha256"]:
        raise ValueError("External corpus differs from the recorded processed corpus; document a new recipe/source manifest")
    exclusions = []
    if recipe["mode"] == "holdout":
        exclusions.append(read_frame(out / "data/holdout.csv", recipe["task"]))
    test = raw_dir / FILES[recipe["task"]]["test"]
    if test.exists():
        exclusions.append(read_frame(test, recipe["task"]))
    elif recipe["mode"] == "full":
        raise FileNotFoundError(test)
    report["external"] = prepare_external(external, exclusions, out / "external_text.csv")
    report["environment"] = {"python": sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    report["recipe_hash"] = recipe_hash(recipe)
    (out / "data_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    jobs = commands(recipe, out, raw_dir, selected)
    classifier_command = next(j for j in jobs if j["stage"] == "classifier")["command"]
    target_stem = Path(classifier_command[classifier_command.index("--predict") + 1]).stem
    for group in recipe["groups"]:
        name = group["id"]
        if selected and name not in selected:
            continue
        directory = out / name
        # Never reuse a partially trained adapter or stale probability file.
        if directory.exists():
            raise FileExistsError(f"{directory}: use a fresh output path or --combine-only for completed groups")
        directory.mkdir(parents=True)
        actual = []
        prefixes = []
        for index, job in enumerate(j for j in jobs if j["group"] == name):
            cmd = list(job["command"])
            code = launch(cmd, directory / f"job_{index}.log")
            effective = job.get("requested_seed")
            if code == 4 and job["stage"] == "classifier":
                effective += 1000
                cmd[cmd.index("--seed") + 1] = str(effective)
                retry_out = directory / f"s{job['requested_seed']}_retry{effective}"
                cmd[cmd.index("--out") + 1] = str(retry_out)
                code = launch(cmd, directory / f"job_{index}_retry.log")
            actual.append({**job, "effective_seed": effective, "executed_command": cmd, "returncode": code})
            (directory / "execution.json").write_text(json.dumps(actual, indent=2) + "\n")
            if code:
                raise RuntimeError(f"{name} failed; no partial ensemble will be produced")
            if job["stage"] == "classifier":
                prefixes.append(Path(cmd[cmd.index("--out") + 1]) / target_stem)
        prefix = directory / "group"
        combine(prefixes, recipe["task"], prefix, len(group["classifier_seeds"]))
        meta_path = directory / "group_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["provenance"].update({"recipe_hash": recipe_hash(recipe), "group": name, "execution": actual})
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    if not selected or selected == {g["id"] for g in recipe["groups"]}:
        assemble(recipe, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--external", type=Path, default=PROJECT_ROOT / "data/external/offenseval_kn.csv")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--groups", nargs="+")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="launch GPU jobs; otherwise print the plan only")
    action.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()
    recipe = load_recipe(args.config)
    selected = set(args.groups) if args.groups else None
    if selected and not selected.issubset({g["id"] for g in recipe["groups"]}):
        parser.error("Unknown group name")
    out = (args.out or PROJECT_ROOT / f"artifacts/task_{recipe['task']}/{recipe['mode']}").resolve()
    if args.combine_only:
        assemble(recipe, out)
    elif args.execute:
        execute(recipe, out, args.raw_dir, args.external, selected)
    else:
        print(json.dumps({"execute": False, "recipe_hash": recipe_hash(recipe),
                          "jobs": commands(recipe, out, args.raw_dir, selected)}, indent=2))


if __name__ == "__main__":
    main()
