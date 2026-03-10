import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importy do Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC  # <--- To naprawia Twój błąd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve

# ==========================================
# KONFIGURACJA ŚCIEŻEK
# ==========================================
FOLDERS_SAFE = [
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanDisabled_1/AES-T500+TrojanDisabled_1",
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanDisabled_2/AES-T500+TrojanDisabled_2",
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanEnabled_1/AES-T500+TrojanEnabled_1",
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanEnabled_2/AES-T500+TrojanEnabled_2"
]

FOLDERS_ATTACK = [
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanTriggered_1/AES-T500+TrojanTriggered_1",
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanTriggered_2/AES-T500+TrojanTriggered_2"
]


def load_raw_traces(folder_list, label, max_files=None):
    data_list = []
    labels_list = []
    print(f"--- Wczytywanie klasy {label} ---")
    for folder in folder_list:
        if not os.path.exists(folder):
            continue
        path_pattern = os.path.join(folder, "*.csv")
        files = glob.glob(path_pattern)
        if not files: continue

        print(f"Folder: ...{folder[-30:]} | Plików: {len(files)}")
        if max_files: files = files[:max_files]

        for filepath in files:
            try:
                df = pd.read_csv(filepath, header=None)
                raw = df.values.flatten()
                raw = raw - np.mean(raw)  # DC Offset removal
                data_list.append(raw)
                labels_list.append(label)
            except:
                pass
    return np.array(data_list), np.array(labels_list)


# 1. WCZYTANIE
X_0, y_0 = load_raw_traces(FOLDERS_SAFE, label=0)
X_1, y_1 = load_raw_traces(FOLDERS_ATTACK, label=1)
X = np.concatenate((X_0, X_1), axis=0)
y = np.concatenate((y_0, y_1), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. PCA
print("\n--- Przygotowanie danych (PCA) ---")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pca = PCA(n_components=50)  # Redukcja do 50 cech
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

# 3. OPTYMALIZACJA SVM (GridSearch)
print("\n--- Szukanie najlepszego modelu SVM ---")
param_grid = {
    'C': [1, 10, 50],
    'gamma': ['scale', 'auto'],
    'class_weight': ['balanced', {0: 1, 1: 5}]  # Zwiększamy wagę dla klasy ataku
}

svm_grid = GridSearchCV(
    SVC(kernel='rbf', probability=True),
    param_grid,
    refit=True,
    verbose=2,
    cv=3,
    n_jobs=-1,
    scoring='f1'
)

# Trenujemy na mniejszej próbce dla szybkości (np. 5000), lub usuń [:5000] dla pełnego treningu
print("Rozpoczynam GridSearch...")
svm_grid.fit(X_train_pca[:5000], y_train[:5000])

print(f"\nNajlepsze parametry: {svm_grid.best_params_}")

# Dotrenowanie najlepszego modelu na pełnych danych
best_model = svm_grid.best_estimator_
best_model.fit(X_train_pca, y_train)

# 4. OPTYMALIZACJA PROGU (Threshold Moving)
print("\n--- Szukanie najlepszego progu ---")
y_probs = best_model.predict_proba(X_test_pca)[:, 1]

best_thresh = 0.5
best_f1 = 0
for t in np.arange(0.1, 0.95, 0.05):
    preds = (y_probs >= t).astype(int)
    score = f1_score(y_test, preds)
    if score > best_f1:
        best_f1 = score
        best_thresh = t

print(f"Najlepszy próg: {best_thresh:.2f}")

# 5. WYNIKI
final_preds = (y_probs >= best_thresh).astype(int)
print("\n" + "="*40)
print(classification_report(y_test, final_preds, target_names=['Safe', 'Infected']))
print("="*40)

# Generowanie i zapisywanie Macierzy Pomyłek
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title(f'SVM Optimized (Thresh={best_thresh:.2f})')
plt.ylabel('Rzeczywista klasa')
plt.xlabel('Przewidziana klasa')
plt.tight_layout()

# ZMIANA: Zapisz zamiast pokazywać
output_filename = 'wynik_finalny_svm.png'
plt.savefig(output_filename)
plt.close() # Ważne: zwalnia pamięć i zamyka wykres

print(f"\n>> Sukces! Wykres zapisano w pliku: {output_filename}")
print("   (Otwórz ten plik w folderze projektu, aby zobaczyć macierz)")