# Data 
# https://ieee-dataport.org/documents/dataset-gps-spoofing-detection-autonomous-vehicles

""" Dudas sobre las carácterísticas que otros han utilizado para entrenar los modelos ML/DL de detección de spoofing GPS.
    Si se entrena con las posiciones de una ruta cuando se haga otra ruta distinta la detectará como spoofing"""

import numpy as np
import pandas as pd
import argparse
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import shap


# Setting the numpy random seed
np.random.seed(37)
tf.random.set_seed(37)

OUTPUT_PREFIX = ''


def add_prefix(filename, prefix):
    if not prefix:
        return filename
    root, ext = os.path.splitext(filename)
    base = os.path.basename(root)
    directory = os.path.dirname(root)
    prefixed = f"{prefix}{base}"
    return os.path.join(directory, f"{prefixed}{ext}")


def plot_confusion_matrix_percent(cm, labels=None, filename=None, normalize='true', cmap='Blues'):
    """Plot confusion matrix with counts and percentages (row-normalized by default).

    Args:
        cm: confusion matrix (2D array)
        labels: list of class labels (optional)
        filename: if provided, save the figure to this path
        normalize: 'true' for row-wise, 'all' for overall percent, None for counts only
    """
    cm = np.array(cm, dtype=float)
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    # Compute percentage matrix
    if normalize == 'true':
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_perc = np.divide(cm, row_sums, where=(row_sums != 0)) * 100
    elif normalize == 'all':
        total = cm.sum()
        cm_perc = (cm / total) * 100 if total != 0 else np.zeros_like(cm)
    else:
        cm_perc = None

    # Build annotation strings
    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            if cm_perc is not None:
                annot[i, j] = f"{count}\n{cm_perc[i, j]:.1f}%"
            else:
                annot[i, j] = f"{count}"

    plt.figure(figsize=(6 + 0.5 * len(labels), 5))
    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix (counts and %)' )
    plt.tight_layout()
    if filename:
        plt.savefig(add_prefix(filename, OUTPUT_PREFIX), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Get the Data
def load_data(filename):
    """Carga los datos desde el archivo CSV especificado."""
    try:
        df = pd.read_csv(filename)
        print(f"Datos cargados desde: {filename}")
        print(f"   Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error al cargar el archivo {filename}: {e}")
        sys.exit(1)

def parse_arguments():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Entrenamiento y evaluación de modelos ML/DL para detección de spoofing GPS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python spoofing_gnss_ml_dl.py
  python spoofing_gnss_ml_dl.py -i datos_spoofing.csv
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Ruta al archivo CSV con los datos de entrada (obligatorio)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='Prefijo opcional para todos los ficheros de salida (ej: v1_)'
    )
    parser.add_argument(
        '--split-method',
        type=str,
        choices=['random', 'temporal_class'],
        default='random',
        help='Estrategia de split: random (estratificado) o temporal_class (bloques temporales por clase)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporción de test para el split (por defecto: 0.2)'
    )
    parser.add_argument(
        '--knn-only',
        action='store_true',
        help='Ejecuta solo entrenamiento/evaluación de KNN'
    )
    parser.add_argument(
        '--xgb-only',
        action='store_true',
        help='Ejecuta solo entrenamiento/evaluación de XGBoost'
    )
    parser.add_argument(
        '--xgb-threshold',
        type=float,
        default=0.5,
        help='Umbral de decisión para clase positiva en XGBoost binario (por defecto: 0.5)'
    )
    parser.add_argument(
        '--xgb-scale-pos-weight',
        type=float,
        default=1.0,
        help='Peso de clase positiva para XGBoost binario (por defecto: 1.0)'
    )
    parser.add_argument(
        '--dl-only',
        action='store_true',
        help='Ejecuta solo entrenamiento/evaluación de Deep Learning'
    )
    parser.add_argument(
        '--dl-epochs',
        type=int,
        default=60,
        help='Número máximo de épocas para Deep Learning (por defecto: 60)'
    )
    parser.add_argument(
        '--dl-batch-size',
        type=int,
        default=64,
        help='Batch size para Deep Learning (por defecto: 64)'
    )
    parser.add_argument(
        '--outlier-filter',
        type=str,
        choices=['none', 'iqr'],
        default='none',
        help='Filtrado de outliers/fuera de rango: none o iqr'
    )
    parser.add_argument(
        '--outlier-iqr-k',
        type=float,
        default=1.5,
        help='Multiplicador IQR para outlier-filter=iqr (por defecto: 1.5)'
    )
    parser.add_argument(
        '--feature-range',
        action='append',
        default=[],
        help='Rango manual por feature en formato columna:min:max (puede repetirse)'
    )
    # Add other arguments here as needed (e.g., model selection, output paths)

    return parser.parse_args()


def split_data(X, y, method='random', test_size=0.2, random_state=42):
    """Split train/test con opción robusta para datos ordenados temporalmente por clase."""
    if method == 'random':
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # temporal_class: mantiene el orden temporal dentro de cada clase y toma
    # el final de cada bloque como test para reducir fuga por vecindad temporal.
    idx = np.arange(len(y))
    y_arr = np.asarray(y)
    train_idx, test_idx = [], []
    for cls in np.unique(y_arr):
        cls_idx = idx[y_arr == cls]
        cut = int((1.0 - test_size) * len(cls_idx))
        cut = max(1, min(cut, len(cls_idx) - 1))
        train_idx.extend(cls_idx[:cut])
        test_idx.extend(cls_idx[cut:])

    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def parse_feature_ranges(range_specs):
    """Parsea rangos manuales en formato columna:min:max."""
    ranges = {}
    for spec in range_specs or []:
        parts = spec.split(':')
        if len(parts) != 3:
            raise ValueError(f"Formato invalido en --feature-range: {spec}. Usa columna:min:max")
        col, min_s, max_s = parts
        try:
            min_v = float(min_s)
            max_v = float(max_s)
        except ValueError as e:
            raise ValueError(f"Rango no numerico en --feature-range: {spec}") from e
        if min_v > max_v:
            raise ValueError(f"Rango invalido (min > max) en --feature-range: {spec}")
        ranges[col] = (min_v, max_v)
    return ranges


def build_iqr_ranges(X_train_raw, iqr_k=1.5):
    numeric_cols = X_train_raw.select_dtypes(include=[np.number]).columns
    q1 = X_train_raw[numeric_cols].quantile(0.25)
    q3 = X_train_raw[numeric_cols].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr
    return {col: (float(lower[col]), float(upper[col])) for col in numeric_cols}


def apply_range_filter(X, y, ranges):
    if not ranges:
        return X, y, 0

    mask = pd.Series(True, index=X.index)
    for col, (min_v, max_v) in ranges.items():
        if col not in X.columns:
            continue
        mask &= X[col].between(min_v, max_v)

    removed = int((~mask).sum())
    return X.loc[mask], y.loc[mask], removed

def run_knn(X_train, y_train, X_test, y_test, cv, params=None, output_prefix='knn'):
    """Train and evaluate a K-Nearest Neighbors model using GridSearchCV.

    Returns: (score, best_estimator, y_pred, confusion_matrix)
    """
    if params is None:
        params = {
            'n_neighbors': [9, 11, 13, 15, 17, 21],
            'weights': ['distance', 'uniform'],
            'metric': ['manhattan', 'euclidean'],
            'leaf_size': [20, 30, 40]
        }

    n_classes = len(np.unique(y_train))
    scoring = 'f1' if n_classes == 2 else 'f1_weighted'

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn,
                        param_grid=params,
                        scoring=scoring,
                        n_jobs=-1,
                        cv=cv,
                        verbose=1)
    grid.fit(X_train, y_train)

    print('Best Score:', grid.best_score_)
    print('Best Params:', grid.best_params_)
    print('Best Estimator:', grid.best_estimator_)

    knn_grid = grid.best_estimator_
    y_pred = knn_grid.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Model Accuracy:', score)
    print('Classification Report:\n', classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    try:
        labels = [str(x) for x in knn_grid.classes_]
    except Exception:
        labels = None
    plot_confusion_matrix_percent(cm, labels=labels, filename=f"{output_prefix}.png")

    return score, knn_grid, y_pred, cm


def run_gaussian_nb(X_train, y_train, X_test, y_test, cv, params=None, output_prefix='gaussian_nb'):
    """Train and evaluate Gaussian Naive Bayes using GridSearchCV.

    Returns: (score, best_estimator, y_pred, confusion_matrix)
    """
    if params is None:
        params = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

    gb = GaussianNB()
    grid = GridSearchCV(gb,
                        param_grid=params,
                        scoring='accuracy',
                        n_jobs=-1,
                        cv=cv,
                        verbose=1)
    grid.fit(X_train, y_train)

    print('Best Score:', grid.best_score_)
    print('Best Estimator:', grid.best_estimator_)

    gb_grid = grid.best_estimator_
    y_pred = gb_grid.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Model Accuracy:', score)
    print('Classification Report:\n', classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    try:
        labels = [str(x) for x in gb_grid.classes_]
    except Exception:
        labels = None
    plot_confusion_matrix_percent(cm, labels=labels, filename=f"{output_prefix}.png")

    return score, gb_grid, y_pred, cm


def run_decision_tree(X_train, y_train, X_test, y_test, cv, params=None, output_prefix='decision_tree'):
    """Train and evaluate Decision Tree using GridSearchCV.

    Returns: (score, best_estimator, y_pred, confusion_matrix)
    """
    if params is None:
        params = {
            'max_features': [1, 3, 10],
            'min_samples_split': [2, 3, 10],
            'min_samples_leaf': [1, 3, 10],
            'criterion': ["entropy", "gini"]
        }

    dtc = DecisionTreeClassifier()
    grid = GridSearchCV(dtc,
                        param_grid=params,
                        scoring='accuracy',
                        n_jobs=-1,
                        cv=cv,
                        verbose=1)
    grid.fit(X_train, y_train)

    print('Best Score:', grid.best_score_)
    print('Best Params:', grid.best_params_)
    print('Best Estimator:', grid.best_estimator_)

    dtc_grid = grid.best_estimator_
    y_pred = dtc_grid.predict(X_test)
    score = accuracy_score(y_test, y_pred) if (y_test is not None and len(y_test) > 0) else 0
    print('Model Accuracy:', score)
    print('Classification Report:\n', classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    try:
        labels = [str(x) for x in dtc_grid.classes_]
    except Exception:
        labels = None
    plot_confusion_matrix_percent(cm, labels=labels, filename=f"{output_prefix}.png")

    return score, dtc_grid, y_pred, cm


# Function to apply balancing technique and print results
def apply_balancing(X, y, technique, technique_name):
    X_resampled, y_resampled = technique.fit_resample(X, y)
    print(f"\n{technique_name} class distribution:", Counter(y_resampled))
    return X_resampled, y_resampled


def run_random_forest(X_train, y_train, X_test, y_test, cv, params=None, output_prefix='random_forest'):
    """Train and evaluate Random Forest using GridSearchCV.

    Returns: (score, best_estimator, y_pred, confusion_matrix)
    """
    if params is None:
        params = {
            'max_features': [3, 10],
            'min_samples_leaf': [3, 10],
            'n_estimators': [100, 300]
        }

    rfc = RandomForestClassifier()
    grid = GridSearchCV(rfc,
                        param_grid=params,
                        scoring='accuracy',
                        n_jobs=-1,
                        cv=cv,
                        verbose=1)
    grid.fit(X_train, y_train)

    print('Best Score:', grid.best_score_)
    print('Best Params:', grid.best_params_)
    print('Best Estimator:', grid.best_estimator_)

    rfc_grid = grid.best_estimator_
    y_pred = rfc_grid.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Model Accuracy:', score)
    print('Classification Report:\n', classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    try:
        labels = [str(x) for x in rfc_grid.classes_]
    except Exception:
        labels = None
    plot_confusion_matrix_percent(cm, labels=labels, filename=f"{output_prefix}.png")

    return score, rfc_grid, y_pred, cm


def save_xgb_misclassified_rows(X_test_raw, y_test, y_pred, filename='xgboost_misclassified.csv'):
    """Guarda en un CSV las filas mal clasificadas por XGBoost.

    El CSV incluye las características originales más las columnas
    'true_attack_type' y 'predicted_attack_type'.
    """
    # Asegurar alineación por índice entre y_test y las predicciones
    y_pred_series = pd.Series(y_pred, index=y_test.index, name='predicted_attack_type')
    mis_mask = y_test != y_pred_series

    if mis_mask.sum() == 0:
        print("XGBoost: no hay filas mal clasificadas que guardar.")
        return

    mis_df = X_test_raw.loc[mis_mask].copy()
    mis_df['true_attack_type'] = y_test[mis_mask].values
    mis_df['predicted_attack_type'] = y_pred_series[mis_mask].values

    out_path = add_prefix(filename, OUTPUT_PREFIX)
    mis_df.to_csv(out_path, index=False)
    print(f"XGBoost: guardadas {mis_df.shape[0]} filas mal clasificadas en: {out_path}")


def save_xgb_all_rows_mark_attack_type(X_test_raw, y_test, y_pred, filename='xgboost_all_marked.csv'):
    """Guarda un CSV con todas las filas de test.

    La columna 'attack_type' se mantiene con la etiqueta real y
    se crea una nueva columna 'misclassified' que vale 2 en las
    filas mal clasificadas y 0 en las correctamente clasificadas.
    """
    # Alinear predicciones con índices de y_test
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    df_all = X_test_raw.copy()

    # Añadimos la columna attack_type con la etiqueta real
    df_all['attack_type'] = y_test.values

    # Máscara de errores
    mis_mask = y_test != y_pred_series
    n_errors = int(mis_mask.sum())

    # Nueva columna para marcar filas mal clasificadas
    df_all['misclassified'] = 0
    if n_errors > 0:
        df_all.loc[mis_mask, 'misclassified'] = 2

    out_path = add_prefix(filename, OUTPUT_PREFIX)
    df_all.to_csv(out_path, index=False)
    print(f"XGBoost: guardadas {df_all.shape[0]} filas de test en: {out_path} (errores marcados con misclassified=2; n_errores={n_errors})")


def explain_xgb_misclassified_shap(xgb_model, X_test, y_test, y_pred, feature_names, filename='xgboost_misclassified_shap.csv'):
    """Calcula explicaciones SHAP para las muestras mal clasificadas por XGBoost.

    Genera un CSV con, para cada muestra mal clasificada, las 3
    características más influyentes (por |SHAP|) que han llevado
    al modelo a equivocarse.
    """
    # Asegurar que y_test es una Serie con índice
    if isinstance(y_test, pd.Series):
        y_true = y_test.copy()
    else:
        y_true = pd.Series(y_test)

    y_pred_series = pd.Series(y_pred, index=y_true.index)

    mis_mask = y_true != y_pred_series
    if mis_mask.sum() == 0:
        print("XGBoost-SHAP: no hay filas mal clasificadas, no se genera CSV de explicaciones.")
        return

    # Convertir X_test a DataFrame para poder manejar índices/columnas
    try:
        X_test_df = pd.DataFrame(X_test, columns=feature_names, index=y_true.index)
    except Exception:
        # Fallback sin índice si algo falla
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        y_true = y_true.reset_index(drop=True)
        y_pred_series = y_pred_series.reset_index(drop=True)
        mis_mask = y_true != y_pred_series

    X_err = X_test_df[mis_mask]
    y_true_err = y_true[mis_mask]
    y_pred_err = y_pred_series[mis_mask]

    print(f"XGBoost-SHAP: calculando explicaciones para {X_err.shape[0]} muestras mal clasificadas...")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_err)

    # Construir matriz SHAP por muestra, usando la clase predicha
    shap_per_sample = []
    if isinstance(shap_values, list):
        # multiclase: lista de arrays [n_clases][n_samples, n_features]
        class_to_index = {int(c): i for i, c in enumerate(xgb_model.classes_)}
        y_pred_err_list = list(y_pred_err.astype(int))
        for i, pred_label in enumerate(y_pred_err_list):
            class_idx = class_to_index.get(pred_label, None)
            if class_idx is None:
                vec = np.array(shap_values[0][i])
            else:
                vec = np.array(shap_values[class_idx][i])
            shap_per_sample.append(vec)
    else:
        # binaria / regresión: array [n_samples, n_features]
        shap_per_sample = [np.array(shap_values[i]) for i in range(len(shap_values))]

    shap_per_sample = np.array(shap_per_sample)

    # Asegurar que el número de columnas coincide con feature_names
    if shap_per_sample.ndim == 3:
        shap_per_sample = shap_per_sample.reshape(shap_per_sample.shape[0], -1)

    n_shap_features = shap_per_sample.shape[1]
    if n_shap_features != len(feature_names):
        if n_shap_features > len(feature_names):
            shap_per_sample = shap_per_sample[:, :len(feature_names)]
        else:
            pad_width = len(feature_names) - n_shap_features
            shap_per_sample = np.pad(shap_per_sample, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)

    rows = []
    orig_indices = X_err.index
    for i in range(shap_per_sample.shape[0]):
        abs_shap = np.abs(shap_per_sample[i])
        # índices de las 3 características más influyentes
        top_idx = np.argsort(abs_shap)[::-1][:3]

        row = {
            'original_index': orig_indices[i],
            'true_attack_type': int(y_true_err.iloc[i]),
            'predicted_attack_type': int(y_pred_err.iloc[i]),
        }

        for rank, f_idx in enumerate(top_idx, start=1):
            if f_idx >= len(feature_names):
                continue
            row[f'top{rank}_feature'] = feature_names[int(f_idx)]
            row[f'top{rank}_shap'] = float(shap_per_sample[i][f_idx])

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = add_prefix(filename, OUTPUT_PREFIX)
    out_df.to_csv(out_path, index=False)
    print(f"XGBoost-SHAP: explicaciones de muestras mal clasificadas guardadas en: {out_path}")


def run_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    cv,
    params=None,
    output_prefix='xgboost',
    decision_threshold=0.5,
    scale_pos_weight=1.0
):
    """Train and evaluate XGBoost (XGBClassifier) using GridSearchCV.

    Returns: (score, best_estimator, y_pred, confusion_matrix)
    """
    if params is None:
        """
        params = {
            'max_depth': [6, 7],
            'n_estimators': [500, 900],
            'learning_rate': [0.05],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [1, 2],
            'gamma': [0.0, 0.05],
            'reg_alpha': [0.0, 0.1],
            'reg_lambda': [1.0]
        }
        """

        # Paper Nadir Khan (2024-06-10): Se han probado muchos parámetros y estos son los que dan mejores resultados en general, aunque el espacio de búsqueda es muy grande y no se ha podido explorar exhaustivamente. Se pueden ajustar más finamente si se quiere optimizar aún más el modelo, pero esto ya da buenos resultados.
        params = {
            'max_depth': [3, 15],
            'n_estimators': [50, 250],
            'learning_rate': [0.03, 0.3],
            'subsample': [0.5, 1.0],
            'colsample_bytree': [0.5, 1.0],
            'min_child_weight': [1, 2],
            'gamma': [0.0, 0.05],
            'reg_alpha': [0.0, 1.0],
            'reg_lambda': [0.0, 1.0]
        }

    n_classes = len(np.unique(y_train))
    xgb_kwargs = {
        'eval_metric': 'logloss',
        'n_jobs': -1,
        'random_state': 42,
        'tree_method': 'hist'
    }
    if n_classes > 2:
        xgb_kwargs['objective'] = 'multi:softprob'
        xgb_kwargs['num_class'] = n_classes
        xgb_kwargs['eval_metric'] = 'mlogloss'
    else:
        xgb_kwargs['objective'] = 'binary:logistic'
        xgb_kwargs['scale_pos_weight'] = scale_pos_weight

    scoring = 'f1' if n_classes == 2 else 'f1_weighted'

    xgb = XGBClassifier(**xgb_kwargs)
    grid = GridSearchCV(xgb,
                        param_grid=params,
                        scoring=scoring,
                        n_jobs=-1,
                        cv=cv,
                        verbose=1)
    grid.fit(X_train, y_train)

    print('Best Score:', grid.best_score_)
    print('Best Params:', grid.best_params_)
    print('Best Estimator:', grid.best_estimator_)

    xgb_grid = grid.best_estimator_
    if n_classes == 2:
        y_prob = xgb_grid.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= decision_threshold).astype(int)
        print(f"XGBoost threshold aplicado: {decision_threshold:.3f} | scale_pos_weight={scale_pos_weight}")
    else:
        y_pred = xgb_grid.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Model Accuracy:', score)
    print('Classification Report:\n', classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    try:
        labels = [str(x) for x in xgb_grid.classes_]
    except Exception:
        labels = None
    plot_confusion_matrix_percent(cm, labels=labels, filename=f"{output_prefix}.png")

    return score, xgb_grid, y_pred, cm


def run_deep_learning(X_train, y_train, X_test, y_test, epochs=60, batch_size=64, output_prefix='deep_learning'):
    """Train and evaluate a regularized Keras MLP with stratified validation."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Avoid validation leakage/sesgo por orden temporal en y_train.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    num_classes = len(np.unique(y_train))
    y_tr_cat = to_categorical(y_tr, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    input_dim = X_tr.shape[1]
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.35),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.25),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    classes = np.unique(y_tr)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, class_weights)}

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
    ]

    history = model.fit(
        X_tr,
        y_tr_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
        shuffle=True
    )

    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy (Deep Learning): {accuracy:.4f}")

    d_pred_prob = model.predict(X_test, verbose=0)
    d_pred = d_pred_prob.argmax(axis=1)
    print(classification_report(y_test, d_pred))

    cm = confusion_matrix(y_test, d_pred)
    try:
        labels = [str(x) for x in np.unique(np.concatenate((y_test, d_pred)))]
    except Exception:
        labels = None
    plot_confusion_matrix_percent(cm, labels=labels, filename=f"{output_prefix}.png")

    dl_test_score = accuracy
    dl_val_score = max(history.history.get('val_accuracy', [0])) * 100 if history is not None else 0
    print(f"Best Val Accuracy (Deep Learning): {dl_val_score:.2f}%")
    return dl_test_score, dl_val_score, model, history


# Function to plot class distribution with enhanced aesthetics and save as high-res image
def plot_class_distribution(y):
    class_counts = Counter(y)
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # Create a DataFrame for seaborn
    data = {'Class': classes, 'Count': counts}

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Count', data=data, palette='viridis')

    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Number of Instances', fontsize=14)
    plt.title('Original Class Distribution', fontsize=16)
    plt.xticks(classes)  # Set x-ticks to class labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjusting the aesthetics for publication quality
    sns.despine()  # Remove top and right spines for a cleaner look
    plt.tight_layout()  # Adjust layout to fit labels better

    # Save the figure as a high-resolution image
    plt.savefig(add_prefix("class_distribution.png", OUTPUT_PREFIX), dpi=300, bbox_inches='tight')  # Save at 300 DPI
    # plt.show()

# Function to plot total samples across different methods and save as high-res image
def plot_total_samples(original_count, ros_count, rus_count, smote_count):
    methods = ['Original', 'Random Oversampling', 'Random Undersampling', 'SMOTE']
    counts = [original_count, ros_count, rus_count, smote_count]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=methods, y=counts, palette='viridis')

    plt.xlabel('Methods', fontsize=14)
    plt.ylabel('Total Number of Samples', fontsize=14)
    plt.title('Total Number of Samples Across Different Balancing Methods', fontsize=16)

    # Adding count labels on top of bars
    for i in range(len(counts)):
        plt.text(i, counts[i], counts[i], ha='center', va='bottom', fontsize=12)

    sns.despine()  # Remove top and right spines for a cleaner look
    plt.tight_layout()  # Adjust layout to fit labels better

    # Save the figure as a high-resolution image
    plt.savefig(add_prefix("total_samples_comparison.png", OUTPUT_PREFIX), dpi=300, bbox_inches='tight')  # Save at 300 DPI
    # plt.show()

def model_training_evaluation():
    X = df.drop(columns=['attack_type'])
    y = df.attack_type

    # Split antes del escalado para evitar data leakage
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        X, y, method=args.split_method, test_size=args.test_size, random_state=42
    )
    print(f"Split method: {args.split_method} | test_size={args.test_size}")

    # Filtrado opcional de datos fuera de rango.
    active_ranges = {}
    if args.outlier_filter == 'iqr':
        iqr_ranges = build_iqr_ranges(X_train_raw, iqr_k=args.outlier_iqr_k)
        active_ranges.update(iqr_ranges)

    manual_ranges = parse_feature_ranges(args.feature_range)
    active_ranges.update(manual_ranges)

    if active_ranges:
        old_train_len, old_test_len = len(X_train_raw), len(X_test_raw)
        X_train_raw, y_train, removed_train = apply_range_filter(X_train_raw, y_train, active_ranges)
        X_test_raw, y_test, removed_test = apply_range_filter(X_test_raw, y_test, active_ranges)
        print(
            f"Range filtering activo | metodo={args.outlier_filter} | "
            f"eliminadas train={removed_train}/{old_train_len}, test={removed_test}/{old_test_len}"
        )
    else:
        print("Range filtering desactivado.")

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(
            "El filtrado de rango dejo menos de 2 clases en train o test. "
            "Reduce el filtrado (outlier-iqr-k mayor o rangos manuales menos restrictivos)."
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Print the shapes of each split
    print("Shapes of the splits:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Print the total number of samples
    total_samples = len(X)
    print(f"\nTotal number of samples: {total_samples}")

    # Print the number of samples in each split
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of testing samples: {len(X_test)}")

    # If you want to see the distribution of classes in y_train and y_test

    print("\nClass distribution in training set:")
    print(Counter(y_train))

    print("\nClass distribution in testing set:")
    print(Counter(y_test))

    # Calculate and print percentages
    train_percentage = (len(X_train) / total_samples) * 100
    test_percentage = (len(X_test) / total_samples) * 100

    print(f"\nPercentage of data in training set: {train_percentage:.2f}%")
    print(f"Percentage of data in testing set: {test_percentage:.2f}%")

    # K-fold splits
    cv = StratifiedShuffleSplit(n_splits=5, test_size=.30, random_state=42)

    if args.dl_only:
        run_deep_learning(
            X_train, y_train, X_test, y_test,
            epochs=args.dl_epochs,
            batch_size=args.dl_batch_size,
            output_prefix='deep_learning'
        )
        print("\nModo --dl-only activo: finalizando tras Deep Learning.")
        return

    if args.xgb_only:
        xgb_params = {
            'max_depth': [6, 7],
            'n_estimators': [500, 900],
            'learning_rate': [0.05],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [1, 2],
            'gamma': [0.0, 0.05],
            'reg_alpha': [0.0, 0.1],
            'reg_lambda': [1.0]
        }
        xgb_grid_score, xgb_grid, y_pred, cm = run_xgboost(
            X_train, y_train, X_test, y_test, cv=cv, params=xgb_params, output_prefix='xgboost',
            decision_threshold=args.xgb_threshold,
            scale_pos_weight=args.xgb_scale_pos_weight
        )
        # Guardar filas mal clasificadas por XGBoost usando los datos originales de test
        save_xgb_misclassified_rows(X_test_raw, y_test, y_pred)
        # Guardar todas las filas de test marcando los errores con attack_type = 2
        save_xgb_all_rows_mark_attack_type(X_test_raw, y_test, y_pred)
        # Explicar con SHAP por qué se han mal clasificado esas muestras
        feature_names = X.columns.tolist()
        explain_xgb_misclassified_shap(xgb_grid, X_test, y_test, y_pred, feature_names)
        print("\nModo --xgb-only activo: finalizando tras XGBoost.")
        return

    # Defining all the parameters
    params = {
        'penalty': ['l2'],
        # 'C': [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
        'C': [0.1 ,1 ,10]
        }

    # Building model
    logreg = LogisticRegression(solver='liblinear')

    # Parameter estimating using GridSearch
    grid = GridSearchCV(logreg,
                        param_grid=params,
                        scoring='accuracy',
                        n_jobs =-1,
                        cv=cv,
                        verbose=1)

    # Fitting the model
    grid.fit(X_train, y_train)

    print('Best Score:', grid.best_score_)
    print('Best Params:', grid.best_params_)
    print('Best Estimator:', grid.best_estimator_)

    logreg_grid = grid.best_estimator_
    y_pred = logreg_grid.predict(X_test)

    # Calculating metrics
    logreg_grid_score = accuracy_score(y_test, y_pred)
    print('Model Accuracy:', logreg_grid_score)
    print('Classification Report:\n', classification_report(y_test, y_pred))

    # Confusion matrix of test set
    cm = confusion_matrix(y_test, y_pred)
    try:
        labels = [str(x) for x in logreg_grid.classes_]
    except Exception:
        labels = None
    plot_confusion_matrix_percent(cm, labels=labels, filename="logistic_regression.png")

    # ## **`2. K-Nearest Neighbor (KNN)`**
    params = {
        'n_neighbors': [9, 11, 13, 15, 17, 21],
        'weights': ['distance', 'uniform'],
        'metric': ['manhattan', 'euclidean'],
        'leaf_size': [20, 30, 40]
    }

    knn_grid_score, knn_grid, y_pred, cm = run_knn(X_train, y_train, X_test, y_test, cv=cv, params=params, output_prefix='knn')

    if args.knn_only:
        print("\nModo --knn-only activo: finalizando tras KNN.")
        return


    # ## **`3. Gaussian Naive Bayes (gaussianNB)`**

    params = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

    gb_grid_score, gb_grid, y_pred, cm = run_gaussian_nb(X_train, y_train, X_test, y_test, cv=cv, params=params, output_prefix='gaussian_nb')

    # ## **`5. Decision Tree Classifier`**

    # Use helper to train and evaluate Decision Tree
    dtc_grid_score, dtc_grid, y_pred, cm = run_decision_tree(X_train, y_train, X_test, y_test, cv=cv, params={
        'max_features': [1, 3, 10],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'criterion': ["entropy", "gini"]
    }, output_prefix='decision_tree')


    # ## **`6. Random Forest Classifier`**

    params = {
        'max_features': [3, 10],
        'min_samples_leaf': [3, 10],
        'n_estimators': [100, 300]
    }

    # Use helper to train and evaluate Random Forest
    rfc_grid_score, rfc_grid, y_pred, cm = run_random_forest(X_train, y_train, X_test, y_test, cv=cv, params=params, output_prefix='random_forest')

    # Defining all the parameters
    params = {
        'max_depth': [6, 7],
        'n_estimators': [500, 900],
        'learning_rate': [0.05],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1, 2],
        'gamma': [0.0, 0.05],
        'reg_alpha': [0.0, 0.1],
        'reg_lambda': [1.0]
    }

    # Use helper to train and evaluate XGBoost
    xgb_grid_score, xgb_grid, y_pred, cm = run_xgboost(
        X_train, y_train, X_test, y_test, cv=cv, params=params, output_prefix='xgboost',
        decision_threshold=args.xgb_threshold,
        scale_pos_weight=args.xgb_scale_pos_weight
    )

    # Guardar filas mal clasificadas por XGBoost usando los datos originales de test
    save_xgb_misclassified_rows(X_test_raw, y_test, y_pred)
    # Guardar todas las filas de test marcando los errores con attack_type = 2
    save_xgb_all_rows_mark_attack_type(X_test_raw, y_test, y_pred)
    # Explicar con SHAP por qué se han mal clasificado esas muestras
    feature_names = X.columns.tolist()
    explain_xgb_misclassified_shap(xgb_grid, X_test, y_test, y_pred, feature_names)

    # De momento dejamos fuera XGBoost Deep Learning por tiempo de ejecución 
    # # **Modelo de Deep Learning (Red Neuronal Multicapa)**
    # A continuación se entrena un modelo de red neuronal simple usando TensorFlow/Keras para la clasificación del dataset balanceado.


    # Deep Learning model training/evaluation (encapsulado)
    dl_test_score, dl_val_score, dl_model, dl_history = run_deep_learning(
        X_train, y_train, X_test, y_test,
        epochs=args.dl_epochs,
        batch_size=args.dl_batch_size,
        output_prefix='deep_learning'
    )


    # ===============================================
    # ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
    # ===============================================

    # Feature importance para Random Forest
    print("\n" + "="*50)
    print("ANÁLISIS DE IMPORTANCIA - RANDOM FOREST")
    print("="*50)

    feature_names = X.columns.tolist()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rfc_grid.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 5 características más importantes:")
    print(importance_df.head())

    # Visualizar
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(8), x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.savefig(add_prefix("feature_random_forest_feature_importance.png", OUTPUT_PREFIX), dpi=300, bbox_inches='tight')  # Save at 300 DPI
    #plt.show()

    # Permutation importance

    print("\nCalculando Permutation Importance...")
    # Use the actual test split used above (no balancing variables)
    perm_importance = permutation_importance(
        rfc_grid, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )

    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)

    print("Top 5 características (Permutation Importance):")
    print(perm_df.head())

    # SHAP Analysis (instalar: pip install shap)
    try:
        print("\nCalculando valores SHAP...")

        # Crear explainer (TreeExplainer para modelos de árbol)
        explainer = shap.TreeExplainer(rfc_grid)

        # Calcular SHAP para una muestra (computacionalmente costoso)
        sample_size = min(200, X_test.shape[0])
        # Convert sample to DataFrame so plots and column names work
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        X_sample = X_test_df.iloc[:sample_size]

        shap_values = explainer.shap_values(X_sample)

        # Build per-sample shap vector for the predicted class (robust to label ordering)
        predictions = rfc_grid.predict(X_sample.values)

        # shap_values can be a list (one array per class) or a single array
        shap_per_sample = []
        if isinstance(shap_values, list):
            # map class label to index in rfc_grid.classes_
            class_to_index = {int(c): i for i, c in enumerate(rfc_grid.classes_)}
            for i, pred in enumerate(predictions):
                class_idx = class_to_index.get(int(pred), None)
                if class_idx is None:
                    # fallback: choose first class
                    vec = np.array(shap_values[0][i])
                else:
                    vec = np.array(shap_values[class_idx][i])
                shap_per_sample.append(vec)
        else:
            # Binary/regression case: shap_values is array-like (samples x features)
            for i in range(len(shap_values)):
                shap_per_sample.append(np.array(shap_values[i]))

        shap_per_sample = np.array(shap_per_sample)

        # Ensure shap_per_sample shape matches feature_names length
        if shap_per_sample.ndim == 2:
            n_shap_features = shap_per_sample.shape[1]
        elif shap_per_sample.ndim == 3:
            # sometimes shap_values may have extra leading dim; collapse
            shap_per_sample = shap_per_sample.reshape(shap_per_sample.shape[0], -1)
            n_shap_features = shap_per_sample.shape[1]
        else:
            n_shap_features = 0

        if n_shap_features != len(feature_names):
            print(f"SHAP/features length mismatch: shap_features={n_shap_features}, feature_names={len(feature_names)}")
            # If SHAP has more values than feature names, truncate SHAP vectors.
            if n_shap_features > len(feature_names):
                shap_per_sample = shap_per_sample[:, :len(feature_names)]
                n_shap_features = len(feature_names)
            else:
                # If fewer, pad with zeros
                pad_width = len(feature_names) - n_shap_features
                shap_per_sample = np.pad(shap_per_sample, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
                n_shap_features = len(feature_names)

        # Característica más influyente por predicción (tipo de spoofing)
        most_influential = []
        for i in range(shap_per_sample.shape[0]):
            abs_shap = np.abs(shap_per_sample[i])
            most_important_idx = int(np.argmax(abs_shap))
            # safety clamp
            if most_important_idx >= len(feature_names):
                most_important_idx = len(feature_names) - 1
            most_influential.append({
                'sample': i,
                'most_influential_feature': feature_names[most_important_idx],
                'shap_value': float(shap_per_sample[i][most_important_idx]),
                'prediction (spoofing)': int(predictions[i])
            })

        influence_df = pd.DataFrame(most_influential)


        # Frecuencia de características más influyentes
        influence_freq = influence_df['most_influential_feature'].value_counts()
        if len(influence_freq) > 0:
            print(f"\nCaracterística más frecuentemente influyente: {influence_freq.index[0]}")
            print(influence_freq.head())

        # Guardar resultados (incluyendo original sample indices and predictions)
        influence_df = influence_df.sort_values(['prediction (spoofing)', 'shap_value'], 
                            ascending=[True, False],  # prediction ascendente, |shap| descendente
                            key=lambda x: x.abs() if x.name == 'shap_value' else x)

        print("\nCaracterísticas más influyentes por predicción (Spoofing), valor absoluto:")
        print(influence_df.head(20))
        influence_df.to_csv(add_prefix('most_influential_features_per_prediction.csv', OUTPUT_PREFIX), index=False)

    except ImportError:
        print("SHAP no instalado. Ejecuta: pip install shap")

    # ===============================================
    # COMPARACIÓN DE MODELOS CON IMPORTANCIA
    # ===============================================

    models_summary = {
        'Model': ['Logistic Regression', 'KNN', 'Gaussian NB', 'Decision Tree', 'Random Forest', 'XGBoost', 'Deep Learning'],
        'Accuracy': [logreg_grid_score, knn_grid_score, gb_grid_score, dtc_grid_score, rfc_grid_score, xgb_grid_score, dl_test_score],
        'Top_Feature': ['N/A', 'N/A', 'N/A', 'N/A', importance_df.iloc[0]['feature'], 'N/A', 'N/A']
    }

    summary_df = pd.DataFrame(models_summary)
    print("\nResumen de modelos y característica más importante:")
    print(summary_df)


if __name__ == '__main__':
    # Parsear argumentos de línea de comandos
    args = parse_arguments()
    OUTPUT_PREFIX = args.prefix or ''
    # Support both `--file` (older) and `--input` (`-i`) argument names for backwards compatibility
    filename = getattr(args, 'file', None) or getattr(args, 'input', None)
    # Enforce that filename is provided (argparse 'required' should handle this),
    # but validate defensively in case args were modified programmatically.
    if filename is None:
        print('Error: se requiere el argumento -i/--input con la ruta al CSV')
        sys.exit(2)
    # normalize path
    try:
        import os as _os
        filename = _os.path.abspath(_os.path.expanduser(filename))
    except Exception:
        pass
else:
    # Si se importa como módulo, usar el archivo por defecto
    filename = 'gnss_log_2025_11_22_17_18_54_spoofed_features.csv'

# Cargar datos
df = load_data(filename)

# Show the columns
for col in df.columns:
  print(col, end = '    ')

# Show the distribution of the target variable
print(df['attack_type'].value_counts())

# Exclude non-numeric and the target column 'attack_type' from correlation
corr_df = df.drop(columns=['attack_type'], errors='ignore')
corr_df = corr_df.select_dtypes(include=[np.number])
corr_matrix = corr_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd')
plt.title('Correlation Heatmap')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.savefig(add_prefix("correlation_heatmap.png", OUTPUT_PREFIX), dpi=300, bbox_inches='tight')  # Save at 300 DPI
#plt.show()

df.head()
df.describe()
df.info()
print(df.shape)
df.isnull().sum()

model_training_evaluation()
