#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "tools" / "bundles.json"
IGNORE_NAMES = shutil.ignore_patterns(".DS_Store", "__pycache__")


def load_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Package a CLUTCH asset bundle into a zip archive.")
    parser.add_argument("bundle", help="Bundle key from tools/bundles.json")
    parser.add_argument(
        "--source-root",
        default=str(REPO_ROOT.parent / "assets"),
        help="Source directory containing the files listed in the bundle manifest",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output zip path. Defaults to dist/<archive_name>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_manifest()
    if args.bundle not in manifest:
        raise SystemExit(f"Unknown bundle '{args.bundle}'. Available: {', '.join(sorted(manifest))}")

    bundle = manifest[args.bundle]
    source_root = Path(args.source_root).expanduser().resolve()
    if not source_root.exists():
        raise SystemExit(f"Source root does not exist: {source_root}")

    output = Path(args.output) if args.output else REPO_ROOT / "dist" / bundle["archive_name"]
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    staged_paths = []
    for rel_path in bundle["paths"]:
        src = source_root / rel_path
        if not src.exists():
            missing.append(str(src))
        else:
            staged_paths.append((src, rel_path))
    if missing:
        raise SystemExit("Missing bundle inputs:\n" + "\n".join(missing))

    staging_root = output.parent / f".{output.stem}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_target = staging_root / bundle["target_dir"]
    staging_target.mkdir(parents=True, exist_ok=True)

    try:
        for src, rel_path in staged_paths:
            dst = staging_target / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=IGNORE_NAMES)
            else:
                shutil.copy2(src, dst)
        archive_base = output.with_suffix("")
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=staging_root)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    print(f"Created {archive_path}")


if __name__ == "__main__":
    main()
