import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# KONFIGURACJA ŚCIEŻEK
# ==========================================
# --- KLASA 0: SAFE (Bezpieczne) ---
FOLDERS_DISABLED = [
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanDisabled_1/AES-T500+TrojanDisabled_1",
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanDisabled_2/AES-T500+TrojanDisabled_2"
]

FOLDERS_ENABLED = [
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanEnabled_1/AES-T500+TrojanEnabled_1",
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanEnabled_2/AES-T500+TrojanEnabled_2"
]

# --- KLASA 1: INFECTED (Atak) ---
FOLDERS_TRIGGERED = [
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanTriggered_1/AES-T500+TrojanTriggered_1",
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanTriggered_2/AES-T500+TrojanTriggered_2"
]


def load_traces_from_folders(folder_list, label, max_files=None):
    data_list = []
    labels_list = []

    # Wyciągamy nazwy folderów do logów, żeby było czytelniej
    folder_names = [os.path.basename(f) for f in folder_list]
    print(f"--> Wczytywanie Label {label} z: {folder_names}")

    for folder in folder_list:
        path_pattern = os.path.join(folder, "*.csv")
        files = glob.glob(path_pattern)

        if not files:
            print(f"   [!] OSTRZEŻENIE: Pusty folder lub błędna ścieżka: {folder}")
            continue

        if max_files:
            files = files[:max_files]

        for filepath in files:
            try:
                df = pd.read_csv(filepath, header=None)

                # --- PREPROCESSING ---
                # Wygładzanie (Rolling Mean) - usuwa szum wysokoczęstotliwościowy
                df = df.rolling(window=5, center=True).mean().bfill().ffill()

                # Spłaszczenie do wektora 1D
                trace_vector = df.values.flatten()

                data_list.append(trace_vector)
                labels_list.append(label)
            except Exception as e:
                print(f"   Błąd pliku {filepath}: {e}")

    return data_list, labels_list


# ==========================================
# KROK 1: WCZYTANIE I ŁĄCZENIE DANYCH
# ==========================================
print("--- KROK 1: Ładowanie danych ---")
X_dis, y_dis = load_traces_from_folders(FOLDERS_DISABLED, label=0)
X_enb, y_enb = load_traces_from_folders(FOLDERS_ENABLED, label=0)
X_trig, y_trig = load_traces_from_folders(FOLDERS_TRIGGERED, label=1)

# Łączenie wszystkiego w jedną macierz
# Kolejność list nie ma znaczenia, bo i tak będziemy mieszać (shuffle) przy podziale
X_data = X_dis + X_enb + X_trig
y_data = y_dis + y_enb + y_trig

if len(X_data) == 0:
    raise ValueError("CRITICAL ERROR: Brak danych. Sprawdź ścieżki do folderów!")

X = np.array(X_data)
y = np.array(y_data)

print(f"\nSTATUS DANYCH:")
print(f"Całkowity rozmiar X: {X.shape}")
print(f"Klasa 0 (Safe: Disabled + Enabled): {len(y_dis) + len(y_enb)} próbek")
print(f"Klasa 1 (Infected: Triggered):      {len(y_trig)} próbek")

# ==========================================
# KROK 2: SELEKCJA CECH (Zamiast PCA)
# ==========================================
print("\n--- KROK 2: Preprocessing i Selekcja Cech ---")

# Podział trening/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Skalowanie
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SelectKBest - Wybieramy 500 punktów, które najbardziej różnią się między atakiem a ciszą.
# To działa lepiej niż PCA, bo zachowuje oryginalną domenę czasu.
print("Uruchamiam SelectKBest (szukanie 500 kluczowych punktów)...")
selector = SelectKBest(score_func=f_classif, k=500)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)

print(f"Wymiary danych treningowych po selekcji: {X_train_sel.shape}")

# ==========================================
# KROK 3: TRENING MODELU
# ==========================================
print("\n--- KROK 3: Trenowanie Random Forest ---")

rf_model = RandomForestClassifier(
    n_estimators=400,  # Duża liczba drzew dla stabilności
    max_depth=None,  # Brak limitu głębokości (pozwala wyłapać subtelne różnice)
    min_samples_split=2,
    min_samples_leaf=1,  # Precyzja (może prowadzić do overfittingu, ale przy SelectKBest jest bezpieczniej)
    max_features='sqrt',
    n_jobs=-1,  # Użycie wszystkich rdzeni CPU
    random_state=42,
    class_weight='balanced'  # KLUCZOWE: Wyrównuje wagi, bo klasy Safe (Dis+Enb) jest teraz więcej niż Triggered
)

rf_model.fit(X_train_sel, y_train)

# ==========================================
# KROK 4: WYNIKI I WIZUALIZACJA
# ==========================================
y_pred = rf_model.predict(X_test_sel)
acc = accuracy_score(y_test, y_pred)

print(f"\n=== WYNIKI KOŃCOWE (Accuracy: {acc:.2%}) ===")
print(classification_report(y_test, y_pred, target_names=['Safe (Dis+Enb)', 'Infected (Triggered)']))

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Safe (Dis+Enb)', 'Attack (Triggered)'],
            yticklabels=['Safe (Dis+Enb)', 'Attack (Triggered)'])
plt.title(f'Macierz Pomyłek\nAccuracy: {acc:.2%}')
plt.ylabel('Rzeczywista klasa')
plt.xlabel('Predykcja modelu')
plt.tight_layout()
plt.savefig('wynik_full_dataset.png')

print("Wykres zapisano jako: wynik_full_dataset.png")