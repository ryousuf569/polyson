#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time

import requests

SEARCH_URL = "https://freesound.org/apiv2/search/text/"

CLASS_QUERIES = {
    "explosion": ["explosion", "blast", "boom"],
    "footstep": ["footstep", "footsteps", "walking"],
    "whoosh": ["whoosh", "swoosh", "swish"],
    "impact": ["impact", "hit", "thud"],
    "glass": ["glass break", "glass shatter", "breaking glass"],
}

TARGET_PER_CLASS = 200
PAGE_SIZE = 150
DURATION_FILTER = "duration:[0.3 TO 8]"
FIELDS = "id,name,previews,license,username,url,duration,tags"

# Only permissive licenses: CC0 and CC-BY. NonCommercial and Sampling+ are out.
ALLOWED_LICENSE_MARKERS = (
    "creativecommons.org/publicdomain/zero",
    "creativecommons.org/licenses/by/",
)

REQUEST_DELAY = 1.0
MAX_RETRIES = 3

RAW_DIR = os.path.join("data", "raw")
MANIFEST_PATH = os.path.join(RAW_DIR, "manifest.csv")
MANIFEST_HEADER = [
    "class",
    "sound_id",
    "name",
    "username",
    "license",
    "url",
    "duration",
    "query",
]


def license_allowed(license_str):
    """True for CC0 / CC-BY, False for NonCommercial, Sampling+, anything else."""
    if not license_str:
        return False
    low = license_str.lower()
    if "noncommercial" in low or "sampling+" in low or "samplingplus" in low:
        return False
    return any(marker in low for marker in ALLOWED_LICENSE_MARKERS)


def api_get(params, token):
    """GET the search endpoint with 429 backoff. Returns parsed JSON or None."""
    query_params = dict(params)
    query_params["token"] = token

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(SEARCH_URL, params=query_params, timeout=30)
        except requests.RequestException as exc:
            if attempt >= MAX_RETRIES:
                print("  request failed: %s" % exc, file=sys.stderr)
                return None
            time.sleep(REQUEST_DELAY * (2 ** attempt))
            continue

        if resp.status_code == 429:
            if attempt >= MAX_RETRIES:
                print("  rate limited, giving up on this page", file=sys.stderr)
                return None
            backoff = REQUEST_DELAY * (2 ** (attempt + 1))
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    backoff = max(backoff, float(retry_after))
                except ValueError:
                    pass
            print("  429, backing off %.0fs" % backoff)
            time.sleep(backoff)
            continue

        if resp.status_code != 200:
            print(
                "  HTTP %s: %s" % (resp.status_code, resp.text[:200]),
                file=sys.stderr,
            )
            return None

        try:
            return resp.json()
        except ValueError:
            print("  bad JSON in response", file=sys.stderr)
            return None

    return None


def download_preview(url, dest):
    """Download a preview mp3. Returns True on success or if already present."""
    if os.path.exists(dest):
        return True
    try:
        resp = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        print("  download failed: %s" % exc, file=sys.stderr)
        return False
    if resp.status_code != 200:
        print("  download HTTP %s" % resp.status_code, file=sys.stderr)
        return False

    tmp = dest + ".part"
    with open(tmp, "wb") as handle:
        handle.write(resp.content)
    os.replace(tmp, dest)
    return True


def load_existing_ids(class_name):
    """Sound ids already recorded in the manifest for this class."""
    if not os.path.exists(MANIFEST_PATH):
        return set()
    seen = set()
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("class") == class_name:
                seen.add(str(row.get("sound_id")))
    return seen


def open_manifest():
    """Open the manifest for appending, writing the header if it is new."""
    is_new = not os.path.exists(MANIFEST_PATH) or os.path.getsize(MANIFEST_PATH) == 0
    handle = open(MANIFEST_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if is_new:
        writer.writerow(MANIFEST_HEADER)
        handle.flush()
    return handle, writer


def pull_class(class_name, queries, target, token, writer, manifest_file):
    """Cycle this class's queries, paginating, until target hit or queries dry."""
    class_dir = os.path.join(RAW_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    seen_ids = load_existing_ids(class_name)
    count = len(seen_ids)
    if count:
        print("%s: %d/%d (resuming)" % (class_name, count, target))

    # One cursor per query so we can round-robin without losing our place.
    next_page = {q: 1 for q in queries}
    exhausted = set()

    while count < target and len(exhausted) < len(queries):
        for query in queries:
            if count >= target:
                break
            if query in exhausted:
                continue

            page = next_page[query]
            params = {
                "query": query,
                "fields": FIELDS,
                "page_size": PAGE_SIZE,
                "filter": DURATION_FILTER,
                "page": page,
            }
            time.sleep(REQUEST_DELAY)
            data = api_get(params, token)
            if data is None:
                exhausted.add(query)
                continue

            results = data.get("results") or []
            if not results:
                exhausted.add(query)
                continue

            for item in results:
                if count >= target:
                    break

                sound_id = str(item.get("id"))
                if not sound_id or sound_id in seen_ids:
                    continue

                license_str = item.get("license") or ""
                if not license_allowed(license_str):
                    continue

                previews = item.get("previews") or {}
                preview_url = previews.get("preview-hq-mp3")
                if not preview_url:
                    continue

                dest = os.path.join(class_dir, "%s.mp3" % sound_id)
                if not download_preview(preview_url, dest):
                    continue

                seen_ids.add(sound_id)
                count += 1
                writer.writerow([
                    class_name,
                    sound_id,
                    item.get("name", ""),
                    item.get("username", ""),
                    license_str,
                    item.get("url", ""),
                    item.get("duration", ""),
                    query,
                ])
                manifest_file.flush()
                print("%s: %d/%d" % (class_name, count, target))

            next_page[query] = page + 1
            if not data.get("next"):
                exhausted.add(query)

    if count < target:
        print("%s: stopped at %d/%d (queries exhausted)" % (class_name, count, target))
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download sound effect clips from the Freesound API."
    )
    parser.add_argument(
        "--classes",
        default=",".join(CLASS_QUERIES),
        help="comma-separated class names (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=TARGET_PER_CLASS,
        help="clips per class (default: %d)" % TARGET_PER_CLASS,
    )
    args = parser.parse_args()

    token = os.environ.get("FREESOUND_API_KEY")
    if not token:
        print("FREESOUND_API_KEY is not set", file=sys.stderr)
        return 1

    selected = [c.strip() for c in args.classes.split(",") if c.strip()]
    unknown = [c for c in selected if c not in CLASS_QUERIES]
    if unknown:
        print(
            "unknown class(es): %s (known: %s)"
            % (", ".join(unknown), ", ".join(CLASS_QUERIES)),
            file=sys.stderr,
        )
        return 1

    os.makedirs(RAW_DIR, exist_ok=True)
    manifest_file, writer = open_manifest()
    try:
        for class_name in selected:
            pull_class(
                class_name,
                CLASS_QUERIES[class_name],
                args.limit,
                token,
                writer,
                manifest_file,
            )
    finally:
        manifest_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
