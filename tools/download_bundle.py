#!/usr/bin/env python3
import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "tools" / "bundles.json"


def load_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Download and install a CLUTCH bundle.")
    parser.add_argument("bundle", help="Bundle key from tools/bundles.json")
    parser.add_argument("--url", default=None, help="Override the bundle URL from the manifest")
    parser.add_argument("--archive", default=None, help="Install from a local zip file instead of downloading")
    parser.add_argument(
        "--target-root",
        default=str(REPO_ROOT),
        help="Repo root or alternate install root. Bundle target_dir is created inside it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing target directory.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path):
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main():
    args = parse_args()
    manifest = load_manifest()
    if args.bundle not in manifest:
        raise SystemExit(f"Unknown bundle '{args.bundle}'. Available: {', '.join(sorted(manifest))}")

    bundle = manifest[args.bundle]
    target_root = Path(args.target_root).expanduser().resolve()
    target_dir = target_root / bundle["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"clutch-{args.bundle}-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        archive_path = temp_dir / bundle["archive_name"]

        if args.archive:
            local_archive = Path(args.archive).expanduser().resolve()
            if not local_archive.exists():
                raise SystemExit(f"Archive does not exist: {local_archive}")
            shutil.copy2(local_archive, archive_path)
        else:
            url = args.url or bundle.get("url")
            if not url:
                raise SystemExit(
                    f"No URL configured for bundle '{args.bundle}'. "
                    "Pass --url or update tools/bundles.json."
                )
            print(f"Downloading {url}")
            download_file(url, archive_path)

        extract_root = temp_dir / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_root)

        extracted_target = extract_root / bundle["target_dir"]
        if not extracted_target.exists():
            raise SystemExit(
                f"Archive is missing the expected top-level directory '{bundle['target_dir']}'."
            )

        for child in extracted_target.iterdir():
            destination = target_dir / child.name
            if destination.exists():
                if not args.force:
                    raise SystemExit(
                        f"Target entry already exists: {destination}\n"
                        "Remove it first or rerun with --force."
                    )
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()

            if child.is_dir():
                shutil.copytree(child, destination)
            else:
                shutil.copy2(child, destination)

        print(f"Installed {args.bundle} into {target_dir}")


if __name__ == "__main__":
    main()
