import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, \
    Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================================
# KONFIGURACJA ŚCIEŻEK
# ==========================================

# TRAKTUJEMY JAKO TŁO (SAFE / CLASS 0)
FOLDERS_SAFE = [
    # Disabled
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanDisabled_1/AES-T500+TrojanDisabled_1",
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanDisabled_2/AES-T500+TrojanDisabled_2",
    # Enabled (Włączony, ale nieaktywowany - traktujemy jako tło, aby sieć uczyła się samego triggera)
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanEnabled_1/AES-T500+TrojanEnabled_1",
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanEnabled_2/AES-T500+TrojanEnabled_2"
]

# TRAKTUJEMY JAKO ATAK (INFECTED / CLASS 1)
FOLDERS_ATTACK = [
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanTriggered_1/AES-T500+TrojanTriggered_1",
    "../datasets/AES-T500_power_Temp25C/AES-T500_power_Temp25C/AES-T500+TrojanTriggered_2/AES-T500+TrojanTriggered_2"
]


def load_raw_traces(folder_list, label, max_files=None):
    """ Wczytuje surowe dane czasowe, usuwa DC Offset """
    data_list = []
    labels_list = []
    print(f"Wczytywanie danych dla etykiety {label} z {len(folder_list)} folderów...")

    for folder in folder_list:
        # Sprawdzenie czy ścieżka istnieje
        if not os.path.exists(folder):
            print(f"BŁĄD: Folder nie istnieje: {folder}")
            continue

        path_pattern = os.path.join(folder, "*.csv")
        files = glob.glob(path_pattern)

        if not files:
            print(f"OSTRZEŻENIE: Pusty folder lub brak plików CSV: {folder}")
            continue

        print(f" -> Folder: ...{folder[-40:]} | Znaleziono plików: {len(files)}")

        if max_files:
            files = files[:max_files]

        for filepath in files:
            try:
                # Wczytanie CSV (zakładamy, że dane są w kolumnach lub wierszach bez nagłówka)
                df = pd.read_csv(filepath, header=None)
                # Spłaszczamy do 1D array
                raw_signal = df.values.flatten()

                # Usuwamy składową stałą (DC Offset)
                raw_signal = raw_signal - np.mean(raw_signal)

                data_list.append(raw_signal)
                labels_list.append(label)
            except Exception as e:
                print(f"Skipping bad file {filepath}: {e}")

    return np.array(data_list), np.array(labels_list)


# ==========================================
# KROK 1: WCZYTANIE DANYCH
# ==========================================
print("--- KROK 1: Wczytywanie i łączenie danych ---")

# Wczytujemy Safe (Disabled + Enabled) jako klasę 0
X_0, y_0 = load_raw_traces(FOLDERS_SAFE, label=0)

# Wczytujemy Attack (Triggered) jako klasę 1
X_1, y_1 = load_raw_traces(FOLDERS_ATTACK, label=1)

if len(X_0) == 0 or len(X_1) == 0:
    raise ValueError("Brak danych! Sprawdź, czy ścieżki względne są poprawne względem miejsca uruchomienia skryptu.")

# Łączenie
X = np.concatenate((X_0, X_1), axis=0)
y = np.concatenate((y_0, y_1), axis=0)

print(f"\nWymiary całkowite X: {X.shape}")
print(f"Liczba próbek Safe (0): {len(y_0)}")
print(f"Liczba próbek Infected (1): {len(y_1)}")

# ==========================================
# KROK 2: PREPROCESSING
# ==========================================
print("\n--- KROK 2: Przygotowanie dla CNN ---")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Skalowanie
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape do 3D: [batch, steps, 1]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Obliczenie wag klas (Automatyczne balansowanie, bo Safe jest teraz więcej niż Infected)
class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(zip(np.unique(y_train), class_weights_vals))
print(f"Wyliczone wagi klas: {class_weights}")

# ==========================================
# KROK 3: MODEL CNN (Zmodernizowany)
# ==========================================
print("\n--- KROK 3: Budowa i Trening CNN ---")

model = Sequential([
    # Blok 1 - Wykrywanie prostych cech (np. zbocza sygnału)
    Conv1D(filters=32, kernel_size=16, padding='same', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),

    # Blok 2
    Conv1D(filters=64, kernel_size=16, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),

    # Blok 3 - Głębsza ekstrakcja
    Conv1D(filters=128, kernel_size=8, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),

    # Blok 4
    Conv1D(filters=256, kernel_size=8, padding='same'),
    BatchNormalization(),
    Activation('relu'),

    # ZMIANA: Global Average Pooling zamiast Flatten
    # Pozwala na lepszą generalizację w sygnałach czasowych i mniejszy rozmiar modelu
    GlobalAveragePooling1D(),

    # Klasyfikator
    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001, verbose=1)

# TRENING
history = model.fit(
    X_train, y_train,
    epochs=40,  # Więcej epok, bo mamy EarlyStopping
    batch_size=64,  # Mniejszy batch często poprawia jakość uczenia w sygnałach
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# ==========================================
# KROK 4: WYNIKI I DIAGNOSTYKA
# ==========================================
print("\n--- KROK 4: Generowanie Raportu ---")

y_probs = model.predict(X_test).flatten()

# --- WYKRES 1: HISTOGRAM PEWNOŚCI ---
plt.figure(figsize=(10, 6))
sns.histplot(y_probs[y_test == 0], color='green', label='Safe (Disabled+Enabled)', kde=True, alpha=0.5, bins=40)
sns.histplot(y_probs[y_test == 1], color='red', label='Infected (Triggered)', kde=True, alpha=0.5, bins=40)
plt.axvline(0.5, color='black', linestyle='--', label='Próg 0.5')
plt.title('Rozkład prawdopodobieństwa wykrycia Trojana')
plt.xlabel('Prawdopodobieństwo (0=Safe, 1=Infected)')
plt.ylabel('Liczba próbek')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('wynik_histogram_aes.png')
print(">> Zapisano: wynik_histogram_aes.png")

# Szukanie optymalnego progu
best_thresh = 0.5
best_f1 = 0
thresholds = np.arange(0.1, 0.95, 0.01)
for thresh in thresholds:
    y_tmp = (y_probs >= thresh).astype(int)
    score = f1_score(y_test, y_tmp)
    if score > best_f1:
        best_f1 = score
        best_thresh = thresh

print(f"\nOptymalny Threshold (wg F1-score): {best_thresh:.3f}")
y_pred = (y_probs >= best_thresh).astype(int)

# Raport tekstowy
print("\n" + "=" * 50)
print(classification_report(y_test, y_pred, target_names=['Safe', 'Infected']))
print("=" * 50)

# --- WYKRES 2: MACIERZ POMYŁEK ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Safe', 'Infected'], yticklabels=['Safe', 'Infected'])
plt.title(f'Confusion Matrix (Próg: {best_thresh:.3f})')
plt.ylabel('Rzeczywista klasa')
plt.xlabel('Przewidziana klasa')
plt.tight_layout()
plt.savefig('wynik_macierz_aes.png')
print(">> Zapisano: wynik_macierz_aes.png")