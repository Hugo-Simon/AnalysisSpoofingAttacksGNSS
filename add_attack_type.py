#!/usr/bin/env python3
"""add_attack_type.py

Usage: python add_attack_type.py input.csv ATTACK_VALUE [-o output.csv]

Reads a CSV, adds a column named `attack_type` filled with ATTACK_VALUE for all rows,
and writes the resulting CSV to the given output path (or a default suffix file).

Falls back to the stdlib csv module if pandas is not available.
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description='Add attack_type column to CSV')
    p.add_argument('input', help='Input CSV file path')
    p.add_argument('attack_value', help='Value to put in attack_type column (string or number)')
    p.add_argument('-o', '--output', help='Output CSV file path (optional)')
    args = p.parse_args()

    infile = args.input
    attack_val = args.attack_value
    out = args.output

    if not os.path.isfile(infile):
        print(f"Error: input file not found: {infile}", file=sys.stderr)
        sys.exit(2)

    if out is None:
        base, ext = os.path.splitext(infile)
        out = f"{base}_ml{ext}"

    # Try pandas first
    try:
        import pandas as pd
        df = pd.read_csv(infile)
        df['attack_type'] = attack_val
        df.to_csv(out, index=False)
        print(f"Wrote {len(df)} rows to {out}")
        return
    except Exception:
        # fallback to csv module
        pass

    import csv
    try:
        with open(infile, newline='', encoding='utf-8') as fin, open(out, 'w', newline='', encoding='utf-8') as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            try:
                header = next(reader)
            except StopIteration:
                # empty file, just write header
                writer.writerow(['attack_type'])
                print(f"Wrote empty file with header to {out}")
                return
            # write header appended
            writer.writerow(header + ['attack_type'])
            n = 0
            for row in reader:
                writer.writerow(row + [attack_val])
                n += 1
        print(f"Wrote {n} data rows to {out}")
    except Exception as e:
        print(f"Error processing CSV: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == '__main__':
    main()
