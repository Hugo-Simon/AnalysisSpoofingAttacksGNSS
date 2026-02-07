# Data 
# https://ieee-dataport.org/documents/dataset-gps-spoofing-detection-autonomous-vehicles

""" Dudas sobre las carácterísticas que otros han utilizado para entrenar los modelos ML/DL de detección de spoofing GPS.
    Si se entrena con las posiciones de una ruta cuando se haga otra ruta distinta la detectará como spoofing"""

import numpy as np
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
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
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import shap


# Setting the numpy random seed
np.random.seed(37)


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
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Get the Data
def load_data(filename):
    """Carga los datos desde el archivo CSV especificado."""
    try:
        df = pd.read_csv(filename)
        print(f"✅ Datos cargados desde: {filename}")
        print(f"   Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se pudo encontrar el archivo {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al cargar el archivo {filename}: {e}")
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
    # Add other arguments here as needed (e.g., model selection, output paths)

    return parser.parse_args()

def run_knn(X_train, y_train, X_test, y_test, cv, params=None, output_prefix='knn'):
    """Train and evaluate a K-Nearest Neighbors model using GridSearchCV.

    Returns: (score, best_estimator, y_pred, confusion_matrix)
    """
    if params is None:
        params = {'n_neighbors': [9,11,13,15,16,17], 'weights': ['uniform','distance']}

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn,
                        param_grid=params,
                        scoring='accuracy',
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


def run_xgboost(X_train, y_train, X_test, y_test, cv, params=None, output_prefix='xgboost'):
    """Train and evaluate XGBoost (XGBClassifier) using GridSearchCV.

    Returns: (score, best_estimator, y_pred, confusion_matrix)
    """
    if params is None:
        params = {
            'max_depth': range(2, 10, 1),
            'n_estimators': range(60, 220, 40),
            'learning_rate': [0.1, 0.01, 0.05]
        }

    xgb = XGBClassifier(objective='binary:logistic')
    grid = GridSearchCV(xgb,
                        param_grid=params,
                        scoring='accuracy',
                        n_jobs=-1,
                        cv=cv,
                        verbose=1)
    grid.fit(X_train, y_train)

    print('Best Score:', grid.best_score_)
    print('Best Params:', grid.best_params_)
    print('Best Estimator:', grid.best_estimator_)

    xgb_grid = grid.best_estimator_
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


def run_deep_learning(X_train, y_train, X_test, y_test, epochs=30, batch_size=32, output_prefix='deep_learning'):
    """Train and evaluate a simple Keras MLP classifier.

    Returns: (test_accuracy, best_val_accuracy_percent, model, history)
    """
    # Convert labels to categorical
    try:
        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)
    except Exception:
        # If labels are already categorical-like, try converting via numpy
        y_train_cat = to_categorical(np.asarray(y_train))
        y_test_cat = to_categorical(np.asarray(y_test))

    input_dim = X_train.shape[1]
    num_classes = y_train_cat.shape[1]

    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train_cat, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy (Deep Learning): {accuracy:.4f}")

    d_pred_prob = model.predict(X_test)
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
    plt.savefig("class_distribution.png", dpi=300, bbox_inches='tight')  # Save at 300 DPI
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
    plt.savefig("total_samples_comparison.png", dpi=300, bbox_inches='tight')  # Save at 300 DPI
    # plt.show()

def model_training_evaluation():
    # Assuming df is your DataFrame containing the dataset
    X = df.drop(columns=['attack_type'])
    y = df.attack_type
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)


    # Count original samples
    original_count = len(y)

    # Function to apply balancing technique and print results
    def apply_balancing(X, y, technique, technique_name):
        X_resampled, y_resampled = technique.fit_resample(X, y)
        print(f"\n{technique_name} class distribution:", Counter(y_resampled))
        return X_resampled, y_resampled

    X = df.drop(columns=['attack_type'])
    y = df.attack_type
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Usar el dataset original sin balanceo
    X = df.drop(columns=['attack_type'])
    y = df.attack_type

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train test split con datos originales (sin balanceo)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Print the shapes of each split
    print("Shapes of the splits:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Print the total number of samples
    total_samples = len(X_scaled)
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
    grid.fit(X_train, y_train)  # Cambiar X_rus_train por X_train

    print('Best Score:', grid.best_score_)
    print('Best Params:', grid.best_params_)
    print('Best Estimator:', grid.best_estimator_)

    logreg_grid = grid.best_estimator_
    y_pred = logreg_grid.predict(X_test)  # Cambiar X_rus_test por X_test

    # Calculating metrics
    logreg_grid_score = accuracy_score(y_test, y_pred)  # Cambiar y_rus_test por y_test
    print('Model Accuracy:', logreg_grid_score)
    print('Classification Report:\n', classification_report(y_test, y_pred))

    # Confusion matrix of test set
    cm = confusion_matrix(y_test, y_pred)  # Cambiar y_rus_test por y_test
    try:
        labels = [str(x) for x in logreg_grid.classes_]
    except Exception:
        labels = None
    plot_confusion_matrix_percent(cm, labels=labels, filename="logistic_regression.png")

    # ## **`2. K-Nearest Neighbor (KNN)`**
    params = {
        'n_neighbors': [9,11,13,15,16,17],
        'weights': ['uniform','distance']}

    knn_grid_score, knn_grid, y_pred, cm = run_knn(X_train, y_train, X_test, y_test, cv=cv, params=params, output_prefix='knn')


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
        'max_depth': range (2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05]
    }

    # Use helper to train and evaluate XGBoost
    xgb_grid_score, xgb_grid, y_pred, cm = run_xgboost(X_train, y_train, X_test, y_test, cv=cv, params=params, output_prefix='xgboost')

    # De momento dejamos fuera XGBoost Deep Learning por tiempo de ejecución 
    # # **Modelo de Deep Learning (Red Neuronal Multicapa)**
    # A continuación se entrena un modelo de red neuronal simple usando TensorFlow/Keras para la clasificación del dataset balanceado.


    # Deep Learning model training/evaluation (encapsulado)
    dl_test_score, dl_val_score, dl_model, dl_history = run_deep_learning(X_train, y_train, X_test, y_test, epochs=30, batch_size=32, output_prefix='deep_learning')


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
    plt.savefig("feature_random_forest_feature_importance.png", dpi=300, bbox_inches='tight')  # Save at 300 DPI
    #plt.show()

    # Permutation importance

    print("\n🔄 Calculando Permutation Importance...")
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
        print("\n🎯 Calculando valores SHAP...")

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
            print(f"⚠️ SHAP/features length mismatch: shap_features={n_shap_features}, feature_names={len(feature_names)}")
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
        influence_df.to_csv('most_influential_features_per_prediction.csv', index=False)

    except ImportError:
        print("⚠️ SHAP no instalado. Ejecuta: pip install shap")

    # ===============================================
    # COMPARACIÓN DE MODELOS CON IMPORTANCIA
    # ===============================================

    models_summary = {
        'Model': ['Logistic Regression', 'KNN', 'Gaussian NB', 'Decision Tree', 'Random Forest', 'XGBoost', 'Deep Learning'],
        'Accuracy': [logreg_grid_score, knn_grid_score, gb_grid_score, 0, rfc_grid_score, xgb_grid_score, dl_test_score],
        'Top_Feature': ['N/A', 'N/A', 'N/A', 'N/A', importance_df.iloc[0]['feature'], 'N/A', 'N/A']
    }

    summary_df = pd.DataFrame(models_summary)
    print("\n📊 Resumen de modelos y característica más importante:")
    print(summary_df)


if __name__ == '__main__':
    # Parsear argumentos de línea de comandos
    args = parse_arguments()
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
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')  # Save at 300 DPI
#plt.show()

df.head()
df.describe()
df.info()
print(df.shape)
df.isnull().sum()

model_training_evaluation()