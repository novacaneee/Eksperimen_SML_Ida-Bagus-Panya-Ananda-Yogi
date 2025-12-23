import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =======================
# KONFIGURASI PATH DINAMIS
# =======================
# Agar script ini bisa jalan di Laptop (Windows) maupun GitHub Actions (Linux) tanpa ubah path manual

# 1. Cari tahu di mana script ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Lokasi Root Repository (Naik satu tingkat dari folder preprocessing)
ROOT_DIR = os.path.dirname(BASE_DIR)

# 3. Path ke Data Mentah (heart_failure_raw.csv ada di Root)
RAW_DATA_PATH = os.path.join(ROOT_DIR, "heart_failure_raw.csv")

# 4. Folder Output (Akan dibuat di dalam folder preprocessing)
OUTPUT_DIR = os.path.join(BASE_DIR, "heart_failure_preprocessing")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")


def load_data(path):
    """Memuat data mentah dengan pengecekan error"""
    print(f"[INFO] Mencoba memuat data dari: {path}")
    
    if not os.path.exists(path):
        # Coba fallback: mungkin filenya ada di sebelah script (kalau user salah taruh)
        fallback_path = os.path.join(BASE_DIR, "heart_failure_raw.csv")
        if os.path.exists(fallback_path):
            print(f"[WARN] File tidak ditemukan di root, tapi ditemukan di preprocessing folder.")
            path = fallback_path
        else:
            raise FileNotFoundError(f"FATAL ERROR: File dataset tidak ditemukan di: {path}\nPastikan file 'heart_failure_raw.csv' ada di folder utama repository.")
            
    return pd.read_csv(path)

def preprocess_and_save(df):
    """
    1. Scaling fitur numerik
    2. Split Train/Test
    3. Simpan Scaler & Dataset CSV
    """
    
    # --- 1. PREPROCESSING ---
    print("[INFO] Memulai Preprocessing...")
    
    # Hapus duplikat
    df = df.drop_duplicates(keep="first")
    
    # Pisahkan Fitur & Target
    # Pastikan nama kolom target sesuai datasetmu (DEATH_EVENT)
    target_col = 'DEATH_EVENT'
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dataset.")

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Fitur yang akan di-scale
    cols_to_scale = [
        'age', 'creatinine_phosphokinase', 'ejection_fraction', 
        'platelets', 'serum_creatinine', 'serum_sodium', 'time'
    ]
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    
    # Simpan Scaler (Penting buat Kriteria 2 & 4)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[SUCCESS] Scaler berhasil disimpan di: {SCALER_PATH}")

    # --- 2. SPLITTING ---
    print("[INFO] Membagi Data Train & Test (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Gabungkan kembali X dan y
    train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    # --- 3. SAVING ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_file = os.path.join(OUTPUT_DIR, "train_data.csv")
    test_file = os.path.join(OUTPUT_DIR, "test_data.csv")
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"[SUCCESS] Data Train disimpan: {train_file} ({train_df.shape})")
    print(f"[SUCCESS] Data Test disimpan:  {test_file} ({test_df.shape})")

if __name__ == "__main__":
    print("=== START AUTOMATION ===")
    try:
        df = load_data(RAW_DATA_PATH)
        preprocess_and_save(df)
        print("=== FINISHED ===")
    except Exception as e:
        print(f"[ERROR] {e}")
        exit(1) # Keluar dengan kode error biar GitHub Actions tahu kalau gagal
