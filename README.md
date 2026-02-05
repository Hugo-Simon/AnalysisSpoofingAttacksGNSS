# GNSSGPS-Spoofing-Generation-and-Identification
GNSSGPS Spoofing Generation and Identification

- Descarga del proyecto
git clone https://github.com/juliansimon/GNSSGPS-Spoofing-Generation-and-Identification.git

# Descargar el Matlab y el proyecto FGI-GSRx
# Sustituir el fichero doTracking.m por el nuestro

# Descargar los escenarios de TEXBAT

# Procesar los escenarios en Matlab con los parámetros adecuados  
# Modificar el fichero complete_Texbat_Dataset-cleanStatic_GPS_L1_CA.txt y el que se ejecuta
runGNSSSingleSatelliteTracking.bat

# Copiar los ficheros csv generados a las carpetas correspondientes de los escenarios
cp XXX dsXXX/

# Preparación de datos una vez descargados y procesados con Matlab
cd GNSSGPS-Spoofing-Generation-and-Identification
Set-ExecutionPolicy Unrestricted -Scope Process
.\gps_spoofing_env\Scripts\activate

# Elegimos el canal/satelite que nos interese por ejemplo el 19
cd cleanStatic
python.exe ..\add_attack_type.py tracking_gpsl1_19.csv 0

cd ../ds3
python.exe ..\add_attack_type.py tracking_gpsl1_19.csv 1

cd ..
python.exe .\merge_csv.py -o .\trackData_gpsl1_19_merge.csv .\cleanStatic\tracking_gpsl1_19_ml.csv .\ds3\tracking_gpsl1_19_ml.csv

# En el script columnas_permitidas.py se seleccionan las características que nos interesan
python.exe columnas_permitidas.py .\trackData_gpsl1_19_merge.csv
python.exe ml_dl_test.py -i .\trackData_gpsl1_19_merge_ml.csv

# Los gráficos se crean en la carpeta donde se ejecuta


##### Varios

# Linux
# Creación del entorno virtual
cd GNSSGPS-Spoofing-Generation-and-Identification
python3 -m venv gps_spoofing_env
# Activación del entorno virtual
source gps_spoofing_env/bin/activate
# Actualización del pip 
python3 -m pip install --upgrade pip
# Instalación de las librerías Python necesarias
python3 -m pip install -r requirements.txt

# Windows
# Creación del entorno virtual
cd GNSSGPS-Spoofing-Generation-and-Identification
Set-ExecutionPolicy Unrestricted -Scope Process
python.exe -m venv gps_spoofing_env
# Activación del entorno virtual
.\gps_spoofing_env\Scripts\activate
# Actualización del pip 
python.exe -m pip install --upgrade pip
# Instalación de las librerías Python necesarias
pip install -r requirements.txt

# Windows
# Si ya está creado el entorno e instaladas las librerías instaladas para activar el proyecto hay que ejecutar 
cd GNSSGPS-Spoofing-Generation-and-Identification
Set-ExecutionPolicy Unrestricted -Scope Process
.\gps_spoofing_env\Scripts\activate


