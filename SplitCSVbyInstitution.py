import os
import sys
import csv
import re
import argparse
from collections import defaultdict

# ---------- Config ----------
CHUNKSIZE = 100_000          # rows between flushes
INPUT_ENCODING = "utf-8"     # adjust if needed
INPUT_ERRORS = "replace"     # tolerate odd characters
OUTPUT_NEWLINE = ""          # good CSV behavior on Windows
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Split a CSV into files by (institutionCode, collectionCode) with optional filters."
    )
    p.add_argument(
        "csv_path",
        nargs="?",
        help="Path to input CSV file"
    )
    p.add_argument(
        "--review-only",
        help=("Comma-separated allowed values for the 'review' column (case-insensitive). "
              "Example: --review-only 'none,skip-none,ok'")
    )
    return p.parse_args()

def pick_csv_path(args):
    """Get input path from CLI arg or prompt."""
    if args.csv_path:
        return args.csv_path
    return input("Enter path to the CSV file: ").strip()

def normalize_token(s: str) -> str:
    return (s or "").strip().lower()

def parse_review_whitelist(arg_val: str):
    if not arg_val:
        return None
    # Build a set of normalized allowed values
    parts = [normalize_token(x) for x in arg_val.split(",")]
    return {x for x in parts if x != ""} or None

def safe_part(s: str) -> str:
    """
    Make a string safe for filenames:
    - lowercases, trims,
    - spaces -> underscores,
    - keep alphanumerics, dash, underscore, dot,
    - fallback to 'blank' when empty.
    """
    s = (s or "").strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9._-]", "-", s)
    return s if s else "blank"

def split_csv_by_combo(in_path: str, review_only_set=None):
    if not os.path.isfile(in_path):
        print(f"Error: file not found: {in_path}")
        sys.exit(1)

    base_dir = os.path.dirname(in_path)
    base_name = os.path.splitext(os.path.basename(in_path))[0]

    # Peek header
    with open(in_path, "r", encoding=INPUT_ENCODING, errors=INPUT_ERRORS, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("Error: file is empty.")
            sys.exit(1)

    # Case-insensitive header lookup
    header_lut = {h.strip().lower(): i for i, h in enumerate(header)}
    if "institutioncode" not in header_lut:
        print('Error: Required column "institutionCode" not found.')
        sys.exit(1)
    if "collectioncode" not in header_lut:
        print('Error: Required column "collectionCode" not found.')
        sys.exit(1)

    inst_idx = header_lut["institutioncode"]
    coll_idx = header_lut["collectioncode"]

    # Optional filter columns
    id_score_idx = header_lut.get("id_score")
    instcount_idx = header_lut.get("institutioncount")
    review_idx = header_lut.get("review")

    # Warn if user asked for review-only but column is missing
    if review_only_set is not None and review_idx is None:
        print("Warning: --review-only was provided, but 'review' column was not found. Review filter will be ignored.")
        review_only_set = None

    if review_only_set:
        print("Active review filter (allowed values):", ", ".join(sorted(review_only_set)))

    ### PRECHECK: Count how many rows will be excluded
    excluded_rows = 0
    excluded_by_review = 0
    excluded_by_idscore = 0
    excluded_by_instcount = 0
    malformed_rows = 0
    total_rows = 0

    with open(in_path, "r", encoding=INPUT_ENCODING, errors=INPUT_ERRORS, newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # skip header
        for row in reader:
            total_rows += 1
            if len(row) != len(header):
                malformed_rows += 1
                continue

            # Filters
            if id_score_idx is not None and row[id_score_idx].strip() == "0":
                excluded_rows += 1
                excluded_by_idscore += 1
                continue

            if instcount_idx is not None and row[instcount_idx].strip() == "0":
                excluded_rows += 1
                excluded_by_instcount += 1
                continue

            if review_only_set is not None:
                rv = normalize_token(row[review_idx])
                if rv not in review_only_set:
                    excluded_rows += 1
                    excluded_by_review += 1
                    continue

    print(f"\nPrecheck:")
    print(f"  Total data rows (excluding header): {total_rows:,}")
    if malformed_rows:
        print(f"  Malformed/skipped rows (column count mismatch): {malformed_rows:,}")
    print(f"  Will be excluded overall: {excluded_rows:,}")
    if id_score_idx is not None:
        print(f"    - by id_score == 0: {excluded_by_idscore:,}")
    if instcount_idx is not None:
        print(f"    - by institutioncount == 0: {excluded_by_instcount:,}")
    if review_only_set is not None:
        print(f"    - by review not in allowed set: {excluded_by_review:,}")
    print()

    header_written = set()
    counts = defaultdict(int)
    out_paths_by_key = {}

    def out_path_for(inst_val, coll_val):
        """
        Filename pattern:
          - If collectionCode is blank -> <inputbase>_<institution>.csv
          - Else                       -> <inputbase>_<institution>-<collection>.csv
        """
        safe_inst = safe_part(inst_val)
        coll_blank = coll_val is None or str(coll_val).strip() == ""
        if coll_blank:
            fname = f"{base_name}_{safe_inst}.csv"
        else:
            safe_coll = safe_part(coll_val)
            fname = f"{base_name}_{safe_inst}-{safe_coll}.csv"
        return os.path.join(base_dir, fname)

    def flush_grouped(grouped):
        """Write grouped rows to their respective files and clear the dict."""
        nonlocal header_written
        for key, rows in grouped.items():
            out_path = out_paths_by_key.setdefault(key, out_path_for(*key))
            write_header = out_path not in header_written
            with open(out_path, "a", encoding="utf-8", newline=OUTPUT_NEWLINE) as out_f:
                w = csv.writer(out_f)
                if write_header:
                    w.writerow(header)
                    header_written.add(out_path)
                w.writerows(rows)
            counts[key] += len(rows)
        grouped.clear()

    # Stream through the file and group rows
    with open(in_path, "r", encoding=INPUT_ENCODING, errors=INPUT_ERRORS, newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # skip header

        grouped = defaultdict(list)
        for i, row in enumerate(reader, start=1):
            if len(row) != len(header):
                continue

            # Skip unwanted rows
            if id_score_idx is not None and row[id_score_idx].strip() == "0":
                continue
            if instcount_idx is not None and row[instcount_idx].strip() == "0":
                continue
            if review_only_set is not None:
                rv = normalize_token(row[review_idx])
                if rv not in review_only_set:
                    continue

            inst_val = row[inst_idx]
            coll_val = row[coll_idx]
            key = (inst_val, coll_val)
            grouped[key].append(row)

            if i % CHUNKSIZE == 0:
                flush_grouped(grouped)

        if grouped:
            flush_grouped(grouped)

    # Summary
    print("\nDone. Created files:")
    for (inst, coll) in sorted(counts.keys(), key=lambda x: (str(x[0]).lower(), str(x[1] or "").lower())):
        outp = out_paths_by_key[(inst, coll)]
        print(f"  {os.path.basename(outp)}  —  {counts[(inst, coll)]:,} rows")

def main():
    args = parse_args()
    in_path = pick_csv_path(args)
    if not in_path:
        print("No file provided.")
        sys.exit(0)
    review_only_set = parse_review_whitelist(args.review_only)
    split_csv_by_combo(in_path, review_only_set)

if __name__ == "__main__":
    main()
