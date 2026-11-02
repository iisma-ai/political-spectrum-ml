"""Download the Voteview roll-call CSVs into data/raw."""

import argparse
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://voteview.com/static/data/out"
FILES = {
    "HSall_members.csv": f"{BASE_URL}/members/HSall_members.csv",
    "HSall_parties.csv": f"{BASE_URL}/parties/HSall_parties.csv",
    "HSall_rollcalls.csv": f"{BASE_URL}/rollcalls/HSall_rollcalls.csv",
    "HSall_votes.csv": f"{BASE_URL}/votes/HSall_votes.csv",
}
RAW_DIR = Path(__file__).parent / "data" / "raw"


def download(url: str, target: Path) -> dict:
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
        expected = int(response.headers["Content-Length"])
        modified = response.headers["Last-Modified"]
    size = partial.stat().st_size
    if size != expected:
        partial.unlink()
        raise RuntimeError(f"{target.name}: expected {expected} bytes, got {size}")
    partial.replace(target)
    return {"url": url, "bytes": size, "last_modified": modified}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download files that already exist")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url in FILES.items():
        target = RAW_DIR / name
        if target.exists() and not args.force:
            print(f"skip {name}")
            continue
        print(f"download {name}")
        manifest[name] = download(url, target)
        manifest[name]["downloaded_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
