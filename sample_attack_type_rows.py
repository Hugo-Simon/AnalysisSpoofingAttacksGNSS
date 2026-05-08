import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extrae las primeras N filas con attack_type=0 y las primeras N "
            "filas con attack_type=1 a partir de una fila dada."
        )
    )
    parser.add_argument("input_csv", help="Ruta del CSV de entrada.")
    parser.add_argument("n", type=int, help="Numero de filas a tomar por cada attack_type.")
    parser.add_argument(
        "-o",
        "--output",
        help="Ruta del CSV de salida. Si se omite, se crea junto al CSV de entrada.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=200000,
        help=(
            "Primera fila de datos desde la que empezar a buscar. "
            "No cuenta la cabecera. Valor por defecto: 200000."
        ),
    )
    return parser.parse_args()


def default_output_path(input_path, n, start_row):
    return input_path.with_name(
        f"{input_path.stem}_n{n}_from_row{start_row}_attack_0_1{input_path.suffix}"
    )


def normalize_attack_type(value):
    value = value.strip()
    if value in {"0", "1"}:
        return value

    try:
        numeric_value = float(value)
    except ValueError:
        return value

    if numeric_value == 0:
        return "0"
    if numeric_value == 1:
        return "1"
    return value


def extract_rows(input_path, output_path, n, start_row):
    if n <= 0:
        raise ValueError("n debe ser mayor que 0.")
    if start_row < 1:
        raise ValueError("--start-row debe ser mayor o igual que 1.")

    selected = {"0": [], "1": []}

    with input_path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError("El CSV de entrada no tiene cabecera.")
        if "attack_type" not in reader.fieldnames:
            raise ValueError("El CSV de entrada no contiene la columna 'attack_type'.")

        for data_row_number, row in enumerate(reader, start=1):
            if data_row_number < start_row:
                continue

            attack_type = normalize_attack_type(row["attack_type"])
            if attack_type in selected and len(selected[attack_type]) < n:
                selected[attack_type].append(row)

            if len(selected["0"]) >= n and len(selected["1"]) >= n:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(selected["0"])
        writer.writerows(selected["1"])

    return len(selected["0"]), len(selected["1"])


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output) if args.output else default_output_path(
        input_path, args.n, args.start_row
    )

    count_0, count_1 = extract_rows(input_path, output_path, args.n, args.start_row)

    print(f"Archivo generado: {output_path}")
    print(f"Filas con attack_type=0: {count_0}")
    print(f"Filas con attack_type=1: {count_1}")
    if count_0 < args.n or count_1 < args.n:
        print("Aviso: no se encontraron suficientes filas para alguno de los tipos.")


if __name__ == "__main__":
    main()
