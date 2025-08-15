import os
import sys
import csv
import re
from collections import defaultdict

# ---------- Config ----------
CHUNKSIZE = 100_000          # rows between flushes
INPUT_ENCODING = "utf-8"     # adjust if needed
INPUT_ERRORS = "replace"     # tolerate odd characters
OUTPUT_NEWLINE = ""          # good CSV behavior on Windows
# ----------------------------

def pick_csv_path():
    """Get input path from CLI arg or prompt."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return input("Enter path to the CSV file: ").strip()

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

def split_csv_by_combo(in_path: str):
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
    # Keep original header for writing; build a lowercased copy for indexing
    header_lut = {h.strip().lower(): i for i, h in enumerate(header)}
    if "institutioncode" not in header_lut:
        print('Error: Required column "institutionCode" not found.')
        sys.exit(1)
    if "collectioncode" not in header_lut:
        print('Error: Required column "collectionCode" not found.')
        sys.exit(1)

    inst_idx = header_lut["institutioncode"]
    coll_idx = header_lut["collectioncode"]

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

    # Stream through the file and group rows in memory-efficient batches
    with open(in_path, "r", encoding=INPUT_ENCODING, errors=INPUT_ERRORS, newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # skip header (already read)

        grouped = defaultdict(list)
        for i, row in enumerate(reader, start=1):
            if len(row) != len(header):
                # skip malformed row length
                continue
            inst_val = row[inst_idx]
            coll_val = row[coll_idx]
            key = (inst_val, coll_val)
            grouped[key].append(row)

            if i % CHUNKSIZE == 0:
                flush_grouped(grouped)

        # Flush any remainder
        if grouped:
            flush_grouped(grouped)

    # Summary
    print("\nDone. Created files:")
    for (inst, coll) in sorted(counts.keys(), key=lambda x: (str(x[0]).lower(), str(x[1] or "").lower())):
        outp = out_paths_by_key[(inst, coll)]
        print(f"  {os.path.basename(outp)}  —  {counts[(inst, coll)]:,} rows")

def main():
    in_path = pick_csv_path()
    if not in_path:
        print("No file provided.")
        sys.exit(0)
    split_csv_by_combo(in_path)

if __name__ == "__main__":
    main()
