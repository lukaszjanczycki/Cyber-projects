import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import glob
import math
import os
import re


matplotlib.use('Agg') # Używamy backendu nie wymagającego GUI

def generate_all_plots():
    targets = ["AES-T400", "AES-T500", "AES-T600", "AES-T800", "AES-T1000", "AES-T1600"]
    conditions = ["TrojanTriggered", "TrojanDisabled"]

    # Folder wyjściowy na obrazki (zostanie utworzony w katalogu src)
    output_dir = "Wyniki_Wykresy"
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Rozpoczynam generowanie wykresów. Zapis do: {os.path.abspath(output_dir)} ---")

    for target in targets:
        for condition in conditions:
            search_pattern = os.path.join(
                "..", "datasets",
                f"{target}_power_Temp25C",  # Folder zewnętrzny
                f"{target}_power_Temp25C",  # Folder wewnętrzny (powtórzony)
                f"{target}+{condition}_*",  # Folder próbki (np. AES-T400+TrojanDisabled_1)
                f"{target}+{condition}_*",  # Podfolder próbki (powtórzony)
                "Sample_*.csv"
            )

            all_files = glob.glob(search_pattern)

            if not all_files:
                alt_pattern = os.path.join("..", "datasets", f"{target}_power_Temp25C", "**", f"{target}+{condition}_*",
                                           "Sample_*.csv")
                all_files = glob.glob(alt_pattern, recursive=True)

            if not all_files:
                print(f"[BRAK DANYCH] Nie znaleziono plików dla: {target} -> {condition}")
                continue

            try:
                files = sorted(all_files, key=lambda x: int(re.search(r'Sample_(\d+)', os.path.basename(x)).group(1)))
            except (AttributeError, ValueError):
                files = sorted(all_files)  # Fallback

            files = files[:10]

            # === RYSOWANIE ===
            n_files = len(files)
            cols = 2
            rows = math.ceil(n_files / cols)

            # Tworzenie figury
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), constrained_layout=True)
            axes_flat = axes.flatten() if n_files > 1 else [axes]

            print(f"Generowanie wykresu: {target} - {condition} ({n_files} próbek)...")

            for i, file in enumerate(files):
                try:
                    df = pd.read_csv(file, header=None)
                    ax = axes_flat[i]
                    ax.plot(df[0], linewidth=1)

                    # Tytuł podwykresu (nazwa pliku)
                    ax.set_title(os.path.basename(file), fontsize=9)
                    ax.grid(True, alpha=0.3)

                    if i % cols == 0:
                        ax.set_ylabel("Moc/Wartość")

                except Exception as e:
                    print(f"  Błąd w pliku {os.path.basename(file)}: {e}")

            for j in range(i + 1, len(axes_flat)):
                axes_flat[j].axis('off')

            plt.suptitle(f"{target} - {condition}", fontsize=16)
            filename = f"{target}_{condition}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=150)

            plt.close(fig)

    print("--- Zakończono! Wszystkie pliki zapisane w folderze 'Wyniki_Wykresy'. ---")


if __name__ == "__main__":
    generate_all_plots()