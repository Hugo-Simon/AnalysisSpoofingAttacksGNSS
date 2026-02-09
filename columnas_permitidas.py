import pandas as pd
import sys
from pathlib import Path

# Lista de columnas permitidas (modifica según tus necesidades)
"""
columnas_permitidas = [
    "cn0_db_hz",
    #"prn",
    #"PRN_start_sample_count",
    #"prompt_i",
    #"prompt_q",
    "abs_E", # Correlador Early. Ayuda a mantener sincronización precisa con el código PRN del satélite. Indica qué tan bien está alineado el receptor con la señal
    "abs_L", # Correlador Late. Similar a abs_E, pero mide la potencia de la señal en una posición ligeramente retrasada respecto al código PRN. Usado para ajustar la sincronización del código.
    "abs_P", # Correlador Prompt. Mide la potencia de la señal en la posición correcta del código PRN. Es crucial para evaluar la calidad de la señal recibida y mantener el tracking.
    #"abs_VE",
    #"abs_VL",
    #"acc_carrier_phase_rad", # Fase portadora acumulada expresada en radianes. Es la suma de la fase de la portadora GNSS desde que el receptor comenzó a rastrear la señal de un satélite.  Permite calcular distancias con precisión de centímetros, mucho mejor que usando solo el código.
    #"aux1",
    #"aux2",
    #"carr_error_filt_hz",
    #"carr_error_hz",
    "carrier_doppler_hz",
    #"carrier_doppler_rate_hz",
    #"carrier_lock_test",
    #"code_error_chips",
    #"code_error_filt_chips",
    #"code_freq_chips",
    #"code_freq_rate_chips",
    #"rx_time",
    "attack_type"
]
"""

columnas_permitidas = [
    "CN0fromSNR",
    # Componente In-phase (I) del correlador Prompt. 
    # Cuando el receptor GNSS está siguiendo una señal, correlaciona la señal recibida con una réplica local del código PRN en tres alineaciones:
    # Prompt (P) → alineado
    # Y cada una tiene dos componentes:
    # I (In-phase, en fase)
    # Q (Quadrature, en cuadratura)
    # Si el seguimiento es bueno → I_P tiene valor alto (positivo o negativo)
    # Si hay pérdida de lock o spoofing/jamming → I_P cae, se vuelve ruidoso o errático
    "I_P", 
    "Q_P", 
    "doppler", 
    # Indicador del estado de bloqueo del PLL (Phase-Locked Loop) del receptor GNSS.
    # Dice si el lazo de seguimiento de fase está “enganchado” (lock) o no a la portadora de la señal GPS.
    # El PLL es el lazo que:
    # Sigue la fase y frecuencia de la portadora
    # Permite:
    # Demodular correctamente los bits de navegación. Mantener estable I_P y minimizar Q_P
    # 1 / true → PLL en lock
    # 0 / false → PLL fuera de lock
    "pllLockIndicator",
    "dllDiscr", 
    #"carrFreq",
    "attack_type"
]
def filtra_columnas_csv(input_csv, columnas_permitidas):
    df = pd.read_csv(input_csv)
    columnas_encontradas = [col for col in columnas_permitidas if col in df.columns]
    df_filtrado = df[columnas_encontradas]
    output_csv = str(Path(input_csv).with_suffix('')) + '_ml.csv'
    df_filtrado.to_csv(output_csv, index=False)
    print(f"Archivo guardado: {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python filtra_columnas.py <archivo.csv>")
        sys.exit(1)
    filtra_columnas_csv(sys.argv[1], columnas_permitidas)