"""Download the HowTheyVote.eu export (European Parliament roll calls) into data/raw.

Data source: HowTheyVote.eu (Open Database License), https://howtheyvote.eu. The export is published weekly as
GitHub release assets at https://github.com/HowTheyVote/data/releases.
"""

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RELEASES = "https://github.com/HowTheyVote/data/releases"
TABLES = [
    "members",
    "countries",
    "groups",
    "group_memberships",
    "national_parties",
    "national_party_memberships",
    "votes",
    "member_votes",
    "committees",
    "responsible_committee_votes",
    "oeil_subjects",
    "oeil_subject_votes",
    "eurovoc_concepts",
    "eurovoc_concept_votes",
]
FILES = ["last_updated.txt"] + [f"{table}.csv.gz" for table in TABLES]
RAW_DIR = Path(__file__).parent / "data" / "raw"


def latest_tag() -> str:
    """The tag of the latest release, read from where GitHub redirects the 'latest' URL."""
    request = urllib.request.Request(f"{RELEASES}/latest", method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.url.rstrip("/").rsplit("/", 1)[-1]


def download(url: str, target: Path) -> dict:
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
        expected = int(response.headers["Content-Length"])
    size = partial.stat().st_size
    if size != expected:
        partial.unlink()
        raise RuntimeError(f"{target.name}: expected {expected} bytes, got {size}")
    partial.replace(target)
    return {"url": url, "bytes": size, "sha256": sha256(target)}


def sha256(path: Path) -> str:
    """Assets under a tag can be replaced, so the hash shows which version a result was built from."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag such as 2026-11-07 (default: the latest release)")
    parser.add_argument("--force", action="store_true", help="download files that already exist")
    args = parser.parse_args()

    tag = args.tag or latest_tag()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("release") not in (None, tag) and not args.force:
        raise SystemExit(f"data/raw holds release {manifest['release']}; pass --force to replace it with {tag}")
    manifest["release"] = tag
    files = manifest.setdefault("files", {})
    for name in FILES:
        target = RAW_DIR / name
        if target.exists() and not args.force:
            print(f"skip {name}")
            if name in files and "sha256" not in files[name]:
                files[name]["sha256"] = sha256(target)
                manifest_path.write_text(json.dumps(manifest, indent=2))
            continue
        print(f"download {name}")
        files[name] = download(f"{RELEASES}/download/{tag}/{name}", target)
        files[name]["downloaded_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"release {tag}")


if __name__ == "__main__":
    main()
