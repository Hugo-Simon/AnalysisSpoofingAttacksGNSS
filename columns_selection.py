import pandas as pd
import sys
from pathlib import Path

# Lista de columnas permitidas (modifica según tus necesidades)
columnas_permitidas = [
    # Carrier-to-Noise density ratio (C/N₀) estimado a partir del SNR (Signal-to-Noise Ratio) medido por el receptor GNSS.
    # Una estimación del C/N₀ obtenida por el receptor a partir del SNR medido en los correladores Prompt (Iₚ,Qₚ) y el tiempo de integración coherente.
    # C/N₀ es cuánta señal útil tiene el satélite por Hz de ruido térmico. Es una medida de la calidad de la señal recibida
    # C/N₀ se calcula dividiendo la potencia portadora por la densidad expectral de ruido, expresada en decibelios por hercio (dB-Hz).
    # SNR se calcula dividiendo la potencia del correlador Prompt por la variaznza del ruido (estimada a partir de los correladores laterales Early y Late). 
    "CN0fromSNR",

    # Componente In-phase (I) del correlador Prompt. 
    # Representa la potencia de la señal alineada con la réplica local del código PRN.
    # Cuando el receptor GNSS está siguiendo una señal, correlaciona la señal recibida con una réplica local del código PRN en tres alineaciones:
    # Prompt (P) → alineado
    # Y cada una tiene dos componentes:
    # I (In-phase, en fase)
    # Q (Quadrature, en cuadratura)
    # Si el seguimiento es bueno → I_P tiene valor alto (positivo o negativo)
    # Si hay pérdida de lock o spoofing/jamming → I_P cae, se vuelve ruidoso o errático
    "I_P", 

    # El valor de correlación de cuadratura y refleja el componente de ruido ortogonal a la señal.
    "Q_P", 

    "pllLockIndicator",
    # Indicador del estado de bloqueo del PLL (Phase-Locked Loop) del receptor GNSS.
    # Dice si el lazo de seguimiento de fase está “enganchado” (lock) o no a la portadora de la señal GPS.
    # El PLL es el lazo que:
    # Sigue la fase y frecuencia de la portadora
    # Permite:
    # Demodular correctamente los bits de navegación. Mantener estable I_P y minimizar Q_P
    # 1 / true → PLL en lock
    # 0 / false → PLL fuera de lock

    # El Doppler de portadora mide los cambios de frecuencia de grano fino debido al movimiento relativo entre el receptor y el satélite. Es una medida de la velocidad radial entre ambos.
    "doppler", 

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