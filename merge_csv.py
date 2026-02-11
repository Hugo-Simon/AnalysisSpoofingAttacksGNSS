#!/usr/bin/env python3
"""merge_csv.py

Usage:
  python merge_csv.py file1.csv file2.csv -o out.csv --how concat
  python merge_csv.py file1.csv file2.csv -o out.csv --how merge --on id --join inner

Modes:
  - concat: stack rows of both files (union columns, fills missing with empty)
  - merge: perform database-style merge on a column (requires pandas)

If pandas is available, it's used for both modes. For concat mode, if pandas is missing
we fall back to a csv-based concatenation that unions headers.
"""
import argparse
import os
import sys


def concat_with_csv(f1, f2, out, skip=0):
    import csv
    # Read headers and write union
    with open(f1, newline='', encoding='utf-8') as a, open(f2, newline='', encoding='utf-8') as b:
        ra = csv.reader(a)
        rb = csv.reader(b)
        try:
            ha = next(ra)
        except StopIteration:
            ha = []
        try:
            hb = next(rb)
        except StopIteration:
            hb = []
        headers = []
        # preserve order: headers from first then any new from second
        for h in ha:
            if h not in headers:
                headers.append(h)
        for h in hb:
            if h not in headers:
                headers.append(h)

        with open(out, 'w', newline='', encoding='utf-8') as fo:
            w = csv.writer(fo)
            w.writerow(headers)
            # skip initial data rows for first file
            for _ in range(skip):
                try:
                    next(ra)
                except StopIteration:
                    break
            # write rows from first
            for row in ra:
                rowd = {k: v for k, v in zip(ha, row)}
                w.writerow([rowd.get(h, '') for h in headers])
            # skip initial data rows for second file
            for _ in range(skip):
                try:
                    next(rb)
                except StopIteration:
                    break
            # write rows from second
            for row in rb:
                rowd = {k: v for k, v in zip(hb, row)}
                w.writerow([rowd.get(h, '') for h in headers])


def main():
    p = argparse.ArgumentParser(description='Concatenate or merge two CSV files')
    p.add_argument('file1', help='First CSV file')
    p.add_argument('file2', help='Second CSV file')
    p.add_argument('-o','--output', required=True, help='Output CSV file')
    p.add_argument('--how', choices=['concat','merge'], default='concat', help='How to combine files')
    p.add_argument('--on', help='Column name to join on (required for merge)')
    p.add_argument('--join', choices=['inner','left','right','outer'], default='inner', help='Join type for merge')
    p.add_argument('--suffixes', default=(',_2'), help='Suffixes for overlapping columns, comma separated, e.g. "",_2"')
    p.add_argument('--skip', type=int, default=0, help='Skip the first N data rows (after header) of each input file')
    args = p.parse_args()

    f1 = args.file1
    f2 = args.file2
    out = args.output

    if not os.path.isfile(f1):
        print(f'Error: file not found: {f1}', file=sys.stderr); sys.exit(2)
    if not os.path.isfile(f2):
        print(f'Error: file not found: {f2}', file=sys.stderr); sys.exit(2)

    try:
        import pandas as pd
        have_pd = True
    except Exception:
        have_pd = False

    # Count rows and equalize lengths by trimming the longer file to the shorter
    def count_rows_pandas(path):
        import pandas as _pd
        # read with skiprows that preserves header (skip only data rows)
        if args.skip > 0:
            skiprows = range(1, 1 + args.skip)
            return len(_pd.read_csv(path, skiprows=skiprows))
        else:
            return len(_pd.read_csv(path))

    def trim_with_pandas(path, nrows):
        import pandas as _pd
        if args.skip > 0:
            skiprows = range(1, 1 + args.skip)
            df = _pd.read_csv(path, skiprows=skiprows)
        else:
            df = _pd.read_csv(path)
        return df.iloc[:nrows]

    try:
        import pandas as _pd
        have_pd_for_count = True
    except Exception:
        have_pd_for_count = False

    def count_rows_fallback(path):
        # count data rows (excluding header) efficiently
        c = 0
        with open(path, 'rb') as fh:
            for i, _ in enumerate(fh):
                c += 1
        # subtract header line if file non-empty
        # subtract header and skipped data rows
        return max(0, c - 1 - args.skip)

    if have_pd_for_count:
        try:
            len1 = count_rows_pandas(f1)
            len2 = count_rows_pandas(f2)
        except Exception:
            len1 = count_rows_fallback(f1)
            len2 = count_rows_fallback(f2)
    else:
        len1 = count_rows_fallback(f1)
        len2 = count_rows_fallback(f2)

    if len1 != len2:
        print(f'Row counts before equalize: {f1}={len1}, {f2}={len2}')
        minlen = min(len1, len2)
        print(f'Trimming larger file(s) to {minlen} data rows')
    else:
        minlen = len1

    if args.how == 'concat':
        if have_pd:
            try:
                if args.skip > 0:
                    skiprows = range(1, 1 + args.skip)
                    df1 = pd.read_csv(f1, skiprows=skiprows)
                    df2 = pd.read_csv(f2, skiprows=skiprows)
                else:
                    df1 = pd.read_csv(f1)
                    df2 = pd.read_csv(f2)
                # trim if necessary
                if len(df1) > minlen:
                    df1 = df1.iloc[:minlen]
                if len(df2) > minlen:
                    df2 = df2.iloc[:minlen]
                df = pd.concat([df1, df2], ignore_index=True, sort=False)
                df.to_csv(out, index=False)
                print(f'Wrote {len(df)} rows to {out}')
                return
            except Exception as e:
                print(f'Error using pandas for concat: {e}', file=sys.stderr)
                # fallback to csv
        # csv fallback
        try:
                # If we need to trim, perform trim while concatenating
                if minlen is not None and (len1 != len2):
                    import csv
                    # read headers
                    with open(f1, newline='', encoding='utf-8') as a, open(f2, newline='', encoding='utf-8') as b:
                        ra = csv.reader(a)
                        rb = csv.reader(b)
                        try:
                            ha = next(ra)
                        except StopIteration:
                            ha = []
                        try:
                            hb = next(rb)
                        except StopIteration:
                            hb = []
                        headers = []
                        for h in ha:
                            if h not in headers:
                                headers.append(h)
                        for h in hb:
                            if h not in headers:
                                headers.append(h)
                        with open(out, 'w', newline='', encoding='utf-8') as fo:
                            w = csv.writer(fo)
                            w.writerow(headers)
                            # skip first args.skip data rows (after header)
                            for _ in range(args.skip):
                                try:
                                    next(ra)
                                except StopIteration:
                                    break
                            # write up to minlen rows from first
                            for i, row in enumerate(ra):
                                if i >= minlen:
                                    break
                                rowd = {k: v for k, v in zip(ha, row)}
                                w.writerow([rowd.get(h, '') for h in headers])
                            # skip first args.skip data rows for second
                            for _ in range(args.skip):
                                try:
                                    next(rb)
                                except StopIteration:
                                    break
                            # write up to minlen rows from second
                            for i, row in enumerate(rb):
                                if i >= minlen:
                                    break
                                rowd = {k: v for k, v in zip(hb, row)}
                                w.writerow([rowd.get(h, '') for h in headers])
                else:
                    concat_with_csv(f1, f2, out, skip=args.skip)
                print(f'Concatenated files to {out}')
                return
        except Exception as e:
            print(f'Error concatenating with csv fallback: {e}', file=sys.stderr); sys.exit(3)

    else:  # merge
        if not have_pd:
            print('Error: merge mode requires pandas to be installed', file=sys.stderr); sys.exit(4)
        if not args.on:
            print('Error: --on is required for merge mode', file=sys.stderr); sys.exit(5)
        try:
            if args.skip > 0:
                skiprows = range(1, 1 + args.skip)
                df1 = pd.read_csv(f1, skiprows=skiprows)
                df2 = pd.read_csv(f2, skiprows=skiprows)
            else:
                df1 = pd.read_csv(f1)
                df2 = pd.read_csv(f2)
            # trim to equal lengths before merge if needed
            if len(df1) > minlen:
                df1 = df1.iloc[:minlen]
            if len(df2) > minlen:
                df2 = df2.iloc[:minlen]
            left_on = args.on
            right_on = args.on
            # perform merge
            df = pd.merge(df1, df2, how=args.join, on=args.on, suffixes=tuple(args.suffixes.split(',')))
            df.to_csv(out, index=False)
            print(f'Merged files to {out} with {len(df)} rows')
            return
        except Exception as e:
            print(f'Error merging files with pandas: {e}', file=sys.stderr); sys.exit(6)

if __name__ == '__main__':
    main()
