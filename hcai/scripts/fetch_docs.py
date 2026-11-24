"""Download the document snapshots into data/raw/docs (run once, on your machine).

python scripts/fetch_docs.py
python scripts/fetch_docs.py --only who-hypertension
python scripts/fetch_docs.py --add-local my_page.txt --doc-id x --title "T" --url U --license L
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hcai.config import load_config
from hcai.rag.fetch import add_local_document, fetch_all


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="*", help="doc_ids to fetch")
    ap.add_argument("--add-local", type=Path, help="manually saved text file to add")
    ap.add_argument("--doc-id")
    ap.add_argument("--title")
    ap.add_argument("--url")
    ap.add_argument("--license", dest="license_")
    args = ap.parse_args()
    docs_dir = load_config().paths.docs_dir
    if args.add_local:
        if not all([args.doc_id, args.title, args.url, args.license_]):
            ap.error("--add-local needs --doc-id --title --url --license")
        entry = add_local_document(
            docs_dir, args.add_local, args.doc_id, args.title, args.url, args.license_
        )
        print(f"added {entry['doc_id']} sha256={entry['sha256'][:12]}")
        return 0
    status = fetch_all(docs_dir, args.only)
    for doc_id, st in status.items():
        print(f"{doc_id:28s} {st}")
    failed = [d for d, s in status.items() if s != "ok"]
    if failed:
        print(
            f"\n{len(failed)} failed. Save the page text by hand and use --add-local.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
