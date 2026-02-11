import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

print("Cargando datos...")

# parsear argumento de fichero CSV (opcional)
parser = argparse.ArgumentParser(description="Analiza tracking GNSS y detecta spoofing.")
parser.add_argument("csv", nargs="?", default="trackData_gpsl1_19_merge_ml.csv",
					help="Ruta al fichero CSV de tracking (por defecto: trackData_gpsl1_19_merge_ml.csv)")
args = parser.parse_args()

df = pd.read_csv(args.csv)

# -------------------------------------------------

# 1) Reconstrucción del correlador complejo

# -------------------------------------------------

print("Reconstruyendo fase del PLL...")

df["carrier_phase"] = np.arctan2(df["Q_P"], df["I_P"])
df["carrier_phase_unwrapped"] = np.unwrap(df["carrier_phase"])

# velocidad de cambio de fase (PLL persiguiendo la señal)

df["phase_rate"] = df["carrier_phase_unwrapped"].diff()

# -------------------------------------------------

# 2) Varianza de fase -> fase inicial incorrecta / inestabilidad

# -------------------------------------------------

window = 200
df["phase_variance"] = df["carrier_phase"].rolling(window).var()

# -------------------------------------------------

# 3) Magnitud del correlador (potencia real de señal)

# -------------------------------------------------

df["prompt_magnitude"] = np.sqrt(df["I_P"]**2 + df["Q_P"]**2)

# -------------------------------------------------

# 4) Inestabilidad del PLL (re-enganches)

# -------------------------------------------------

df["pll_changes"] = df["pllLockIndicator"].diff().abs()

# -------------------------------------------------

# 5) Arrastre del código (DLL pull-off attack)

# -------------------------------------------------

df["dll_drift"] = df["dllDiscr"].rolling(window).mean()

# -------------------------------------------------

# 6) Estabilidad de potencia

# -------------------------------------------------

df["cn0_variation"] = df["CN0fromSNR"].rolling(window).std()

# limpiar NaNs

df = df.dropna()

# separar real vs spoofing

real = df[df["attack_type"] == 0]
spoof = df[df["attack_type"] == 1]

print("\n========= ESTADÍSTICAS =========")
print("Varianza de fase media:")
print("Real:", real["phase_variance"].mean())
print("Spoofing:", spoof["phase_variance"].mean())

print("\nVelocidad de fase (|phase_rate| media):")
print("Real:", real["phase_rate"].abs().mean())
print("Spoofing:", spoof["phase_rate"].abs().mean())

print("\nCambios de PLL:")
print("Real:", real["pll_changes"].sum())
print("Spoofing:", spoof["pll_changes"].sum())

print("\nDrift DLL medio:")
print("Real:", real["dll_drift"].mean())
print("Spoofing:", spoof["dll_drift"].mean())

print("\nVariación CN0:")
print("Real:", real["cn0_variation"].mean())
print("Spoofing:", spoof["cn0_variation"].mean())

# -------------------------------------------------

# GRÁFICAS DEMOSTRATIVAS

# -------------------------------------------------

segment = 6000   # muestras a mostrar

plt.figure(figsize=(14,5))
plt.plot(df["carrier_phase_unwrapped"][:segment])
plt.title("Fase de portadora reconstruida (PLL)")
plt.xlabel("Tiempo")
plt.ylabel("Fase (rad)")
plt.grid()
plt.savefig("01_carrier_phase.png")

plt.figure(figsize=(14,5))
plt.plot(df["phase_variance"][:segment])
plt.title("Varianza de fase (inestabilidad del PLL)")
plt.xlabel("Tiempo")
plt.ylabel("Varianza")
plt.grid()
plt.savefig("02_phase_variance.png")

plt.figure(figsize=(14,5))
plt.plot(df["pll_changes"][:segment])
plt.title("Cambios de estado del PLL (re-enganches)")
plt.xlabel("Tiempo")
plt.ylabel("Cambios")
plt.grid()
plt.savefig("03_pll_changes.png")

plt.figure(figsize=(14,5))
plt.plot(df["dll_drift"][:segment])
plt.title("Drift del DLL (pull-off del código)")
plt.xlabel("Tiempo")
plt.ylabel("DLL discriminador")
plt.grid()
plt.savefig("04_dll_drift.png")

plt.figure(figsize=(14,5))
plt.plot(df["cn0_variation"][:segment])
plt.title("Variación del C/N0")
plt.xlabel("Tiempo")
plt.ylabel("STD C/N0")
plt.grid()
plt.savefig("05_cn0_variation.png")

print("\nGráficas generadas. Revisa las imágenes PNG.")
