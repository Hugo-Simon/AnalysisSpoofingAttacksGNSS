
# Analysis Spoofing Attacks GNSS

- Descarga del proyecto
git clone https://github.com/Hugo-Simon/AnalysisSpoofingAttacksGNSS.git

# Descargar el Matlab y el proyecto FGI-GSRx
# Sustituir el fichero doTracking.m por el nuestro

# Descargar los escenarios de TEXBAT

# Procesar los escenarios en Matlab con los parámetros adecuados  
# Modificar el fichero complete_Texbat_Dataset-cleanStatic_GPS_L1_CA.txt y el que se ejecuta
runGNSSSingleSatelliteTracking.bat

# Copiar los ficheros csv generados a las carpetas correspondientes de los escenarios
cp XXX dsXXX/

# Preparación de datos una vez descargados y procesados con Matlab
cd AnalysisSpoofingAttacksGNSS
Set-ExecutionPolicy Unrestricted -Scope Process
.\venv\Scripts\activate

# Elegimos el canal/satelite que nos interese por ejemplo el 19
python.exe add_attack_type.py -o tracking_gpsl1_19_cleanStatic_ml.csv ..\Matlab\cleanStatic\tracking_gpsl1_19.csv 0
python.exe add_attack_type.py -o tracking_gpsl1_ds1_19_ml.csv ..\Matlab\ds1\tracking_gpsl1_19.csv 1
python.exe merge_csv.py -o trackData_gpsl1_ds1_19_merge.csv tracking_gpsl1_19_cleanStatic_ml.csv tracking_gpsl1_ds1_19_ml.csv

# En el script columnas_permitidas.py se seleccionan las características que nos interesan
# Los gráficos se crean en la carpeta donde se ejecuta
python.exe columns_selection.py trackData_ds1_gpsl1_19_merge.csv
python.exe spoofing_gnss_ml_dl.py -i trackData_gpsl1_ds1_19_merge_ml.csv --prefix ds1_19_

# Otro ejemplo con skip a 110000 porque hay varios escenarios que el spoofing no empieza hasta los 100 segundos
python.exe add_attack_type.py -o tracking_gpsl1_16_cleanStatic_ml.csv ..\Matlab\cleanStatic\tracking_gpsl1_16.csv 0
python.exe add_attack_type.py -o tracking_gpsl1_ds8_16_ml.csv ..\Matlab\ds8\tracking_gpsl1_16.csv 1
python.exe merge_csv.py --skip 110000 -o trackData_gpsl1_ds8_16_merge.csv tracking_gpsl1_16_cleanStatic_ml.csv tracking_gpsl1_ds8_16_ml.csv
python.exe columns_selection.py trackData_gpsl1_ds8_16_merge.csv
python.exe spoofing_gnss_ml_dl.py -i trackData_gpsl1_ds8_16_merge_ml.csv --prefix ds8_16_

# Otro
python.exe add_attack_type.py -o tracking_gpsl1_16_cleanStatic_ml.csv ..\Matlab\cleanStatic\tracking_gpsl1_16.csv 0
python.exe add_attack_type.py -o tracking_gpsl1_ds3_16_ml.csv ..\Matlab\ds3\tracking_gpsl1_16.csv 1
python.exe merge_csv.py -o trackData_gpsl1_ds3_16_merge.csv tracking_gpsl1_16_cleanStatic_ml.csv tracking_gpsl1_ds3_16_ml.csv
python.exe columns_selection.py trackData_gpsl1_ds3_16_merge.csv
python.exe spoofing_gnss_ml_dl.py -i trackData_gpsl1_ds3_16_merge_ml.csv --prefix ds3_16_

# Extras
python.exe plot_csv_column.py trackData_gpsl1_ds3_16_merge.csv CN0fromSNR 
python.exe plot_csv_column.py trackData_gpsl1_ds3_16_merge.csv I_P --ma 1000
python.exe spoofing_tracking_analysis.py trackData_gpsl1_19_merge_ml.csv

##### Varios

# Creación del entorno virtual
# Windows
cd AnalysisSpoofingAttacksGNSS
Set-ExecutionPolicy Unrestricted -Scope Process
python.exe -m venv venv
# Activación del entorno virtual
.\venv\Scripts\activate
# Actualización del pip 
python.exe -m pip install --upgrade pip
# Instalación de las librerías Python necesarias
pip install -r requirements.txt

# Creación del entorno virtual
# Linux
cd AnalysisSpoofingAttacksGNSS
python3 -m venv venv
# Activación del entorno virtual
source venv/bin/activate
# Actualización del pip 
python3 -m pip install --upgrade pip
# Instalación de las librerías Python necesarias
python3 -m pip install -r requirements.txt


# Windows
# Si ya está creado el entorno e instaladas las librerías instaladas para activar el proyecto hay que ejecutar 
cd AnalysisSpoofingAttacksGNSS
Set-ExecutionPolicy Unrestricted -Scope Process
.\venv\Scripts\activate


