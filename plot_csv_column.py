#!/usr/bin/env python3
"""plot_csv_column.py

Usage:
    python plot_csv_column.py input.csv column_name [--x x_column] [--kind line|hist|box|scatter] [--ma window] [-o out.png] [--no-show]

Reads the CSV and plots `column_name`. If `--x` is provided, uses that column as x-axis. Saves to image file if `-o` given (default: input_column.png), otherwise shows the plot.

Example:
  python plot_csv_column.py trackData_gpsl1_19_merge_ml.csv CN0fromSNR

Requires: pandas, matplotlib
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description='Plot a column from a CSV file')
    p.add_argument('input', help='Input CSV file')
    p.add_argument('column', help='Column name to plot')
    p.add_argument('--x', dest='xcol', help='Optional column to use as x axis')
    p.add_argument('--kind', choices=['line','hist','box','scatter'], default='line', help='Type of plot')
    p.add_argument('--group', help='Optional column name to group by (plots one series per distinct value). Default: attack_type', default=None)
    p.add_argument('--ma', type=int, default=None, help='Optional moving average window for line plots (positive integer)')
    p.add_argument('-o','--output', help='Output image file (png, pdf, etc). Default: <input>_<column>.png')
    p.add_argument('--no-show', dest='show', action='store_false', help='Do not show the plot interactively')
    p.add_argument('--title', help='Plot title')
    args = p.parse_args()

    infile = args.input
    col = args.column
    xcol = args.xcol
    kind = args.kind
    out = args.output
    show = args.show
    title = args.title
    ma_window = args.ma

    if ma_window is not None and ma_window <= 0:
        print('Error: --ma must be a positive integer.', file=sys.stderr)
        sys.exit(10)

    if not os.path.isfile(infile):
        print(f'Error: input file not found: {infile}', file=sys.stderr)
        sys.exit(2)

    # Try to import pandas and matplotlib
    try:
        import pandas as pd
    except Exception as e:
        print('Error: pandas is required for this script.', file=sys.stderr)
        sys.exit(3)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('Error: matplotlib is required for this script.', file=sys.stderr)
        sys.exit(4)

    try:
        df = pd.read_csv(infile)
    except Exception as e:
        print(f'Error reading CSV: {e}', file=sys.stderr)
        sys.exit(5)

    if col not in df.columns:
        print(f'Error: column "{col}" not found in CSV. Available columns: {list(df.columns)}', file=sys.stderr)
        sys.exit(6)
    if xcol and xcol not in df.columns:
        print(f'Error: x column "{xcol}" not found in CSV. Available columns: {list(df.columns)}', file=sys.stderr)
        sys.exit(7)
    group_col = args.group
    if group_col is None and 'attack_type' in df.columns:
        group_col = 'attack_type'
    if group_col is not None and group_col not in df.columns:
        print(f'Error: group column "{group_col}" not found in CSV. Available columns: {list(df.columns)}', file=sys.stderr)
        sys.exit(9)

    # Prepare output filename default
    if out is None:
        base, _ = os.path.splitext(os.path.basename(infile))
        safe_col = col.replace(' ','_')
        out = f"{base}_{safe_col}.png"

    plt.figure()
    if title:
        plt.title(title)

    def moving_average(series, window):
        return series.rolling(window=window, min_periods=1).mean()

    if kind == 'line':
        if group_col:
            for name, group in df.groupby(group_col):
                y_vals = group[col]
                if ma_window:
                    y_vals = moving_average(y_vals, ma_window)
                if xcol:
                    plt.plot(group[xcol], y_vals, marker='.', linestyle='-', label=str(name))
                else:
                    plt.plot(y_vals.reset_index(drop=True), marker='.', linestyle='-', label=str(name))
            plt.legend()
        else:
            y_vals = df[col]
            if ma_window:
                y_vals = moving_average(y_vals, ma_window)
            if xcol:
                plt.plot(df[xcol], y_vals, marker='.', linestyle='-')
                plt.xlabel(xcol)
            else:
                plt.plot(y_vals, marker='.', linestyle='-')
                plt.xlabel('index')
        plt.ylabel(col)
    elif kind == 'hist':
        if group_col:
            for name, group in df.groupby(group_col):
                plt.hist(group[col].dropna(), bins=50, alpha=0.5, label=str(name))
            plt.legend()
        else:
            plt.hist(df[col].dropna(), bins=50)
        plt.xlabel(col)
        plt.ylabel('count')
    elif kind == 'box':
        if group_col:
            data = [g[col].dropna().values for _, g in df.groupby(group_col)]
            labels = [str(name) for name, _ in df.groupby(group_col)]
            plt.boxplot(data)
            plt.xticks(range(1, len(labels)+1), labels, rotation=45)
        else:
            plt.boxplot(df[col].dropna())
        plt.ylabel(col)
    elif kind == 'scatter':
        if not xcol:
            print('Error: scatter plot requires --x column', file=sys.stderr)
            sys.exit(8)
        if group_col:
            for name, group in df.groupby(group_col):
                plt.scatter(group[xcol], group[col], s=10, label=str(name))
            plt.legend()
        else:
            plt.scatter(df[xcol], df[col], s=10)
        plt.xlabel(xcol)
        plt.ylabel(col)

    plt.tight_layout()

    try:
        plt.savefig(out)
        print(f'Saved plot to {out}')
    except Exception as e:
        print(f'Warning: failed to save plot: {e}', file=sys.stderr)

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f'Warning showing plot interactively: {e}', file=sys.stderr)

if __name__ == '__main__':
    main()
