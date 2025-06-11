import serial
import pandas as pd
import time # Untuk mengukur runtime
import sys
import os
import re # For regex-based cleaning of column names
import numpy as np # Untuk scikit-fuzzy dan operasi numerik
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler # Untuk penskalaan fitur KNN

# scikit-fuzzy imports
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Konfigurasi ---
SERIAL_PORT = 'COM3'  # Ganti dengan port serial ESP32 Anda
BAUD_RATE = 115200

# Nama file data historis ASLI (akan diproses)
RAW_DATA_FILE = 'data_sensor1.xlsx' 

# Nama file data historis setelah pembersihan (ini yang akan digunakan untuk model)
CLEAN_DATA_FILE = 'data_sensor_clean_3000.xlsx' 

# String header yang diharapkan
HEADER_STRING = "Timestamp,Temperature (?C),Humidity (%),CO2 (ppm),Light Intensity (lux),Motion Detected,Ventilation Status"
expected_columns = [col.strip() for col in HEADER_STRING.split(',')]

# Data hardcode untuk pengujian (hanya untuk MODE HARDCODE)
HARDCODE_SENSOR_DATA = {
    "temp": 25.8,
    "hum": 45.6,
    "co2": 997.0,
    "light": 252.0,
    "motion": 1
}

# --- Fungsi Pembantu untuk Pembacaan Data ---
def read_data_from_file(file_path, header_string, expected_cols):
    """
    Memuat data dari file CSV/Excel, mencari header, dan mengembalikan DataFrame yang bersih.
    Menangani berbagai delimiter dan tipe file.
    """
    print(f"\nMembaca data dari file: {file_path}")
    df_processed = None
    
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.xlsx':
        try:
            df_raw = pd.read_excel(file_path, header=None)
            print(f"DEBUG: df_raw shape after initial read (xlsx): {df_raw.shape}")
            
            header_row_index = -1
            for index, row in df_raw.iterrows():
                row_values_as_str = ','.join(filter(pd.notna, map(str, row.tolist())))
                if header_string in row_values_as_str:
                    header_row_index = index
                    break
            
            if header_row_index != -1:
                df_temp = df_raw.iloc[header_row_index + 1:].copy() 
                
                if df_temp.shape[1] == len(expected_cols):
                    df_temp.columns = expected_cols
                elif df_temp.shape[1] > len(expected_cols):
                    print(f"WARNING: File Excel memiliki {df_temp.shape[1]} kolom, lebih banyak dari yang diharapkan ({len(expected_cols)}). Kolom ekstra akan diabaikan.")
                    df_temp = df_temp.iloc[:, :len(expected_cols)]
                    df_temp.columns = expected_cols
                else: 
                    print(f"WARNING: File Excel memiliki {df_temp.shape[1]} kolom, lebih sedikit dari yang diharapkan ({len(expected_cols)}). Kolom yang hilang akan diisi NA.")
                    df_temp.columns = [str(col) if pd.notna(col) else f"Unnamed_Col{i}" for i, col in enumerate(df_temp.iloc[0])]
                    df_temp = df_temp[1:].copy()
                
                df_processed = df_temp.copy()
                print("DEBUG: Header ditemukan dan kolom ditetapkan untuk file XLSX.")
            else:
                print(f"ERROR: Header '{header_string}' tidak ditemukan di file Excel.")
                print(f"Kolom yang ditemukan di Excel adalah: {df_raw.columns.tolist()}")
                print("Pastikan header di Excel dipisah ke kolom-kolom terpisah (gunakan 'Text to Columns' jika perlu).")
                return None

        except Exception as e:
            print(f"Error saat membaca file Excel '{file_path}': {e}")
            return None

    elif file_extension == '.csv':
        delimiters_to_try = [',', ';', '\t'] 
        for delim in delimiters_to_try:
            print(f"Mencoba membaca CSV dengan delimiter: '{delim}'")
            try:
                df_raw = pd.read_csv(file_path, header=None, sep=delim, skipinitialspace=True)
                print(f"DEBUG: df_raw shape after initial read with delimiter '{delim}': {df_raw.shape}")

                if df_raw.shape[1] == 1 and len(expected_cols) > 1:
                    sample_row_str = df_raw.iloc[0, 0] if not df_raw.empty else ""
                    if delim in sample_row_str and len(sample_row_str.split(delim)) > 1:
                        print(f"DEBUG: Dengan delimiter '{delim}', terdeteksi hanya 1 kolom tetapi kontennya mengandung delimiter. Mungkin ada masalah parsing atau quoting. Mencoba delimiter lain.")
                        continue
                    else:
                        print(f"DEBUG: Dengan delimiter '{delim}', jumlah kolom ({df_raw.shape[1]}) lebih sedikit dari yang diharapkan ({len(expected_cols)}). Mencoba delimiter lain.")
                        continue

                header_row_index = -1
                for index, row in df_raw.iterrows():
                    row_values_as_str = ','.join(filter(pd.notna, map(str, row.tolist())))
                    if header_string in row_values_as_str:
                        header_row_index = index
                        break
                    cleaned_row_str = re.sub(r'[^a-zA-Z0-9,]', '', row_values_as_str).replace(' ', '')
                    cleaned_header_str = re.sub(r'[^a-zA-Z0-9,]', '', header_string).replace(' ', '')
                    if cleaned_row_str == cleaned_header_str:
                        header_row_index = index
                        break

                if header_row_index != -1:
                    df_temp = df_raw.iloc[header_row_index + 1:].copy()
                    
                    if df_temp.shape[1] == len(expected_cols):
                        df_temp.columns = expected_cols
                    elif df_temp.shape[1] > len(expected_cols):
                        print(f"WARNING: File CSV memiliki {df_temp.shape[1]} kolom, lebih banyak dari yang diharapkan ({len(expected_cols)}). Kolom ekstra akan diabaikan.")
                        df_temp = df_temp.iloc[:, :len(expected_cols)]
                        df_temp.columns = expected_cols
                    else: 
                        print(f"WARNING: File CSV memiliki {df_temp.shape[1]} kolom, lebih sedikit dari yang diharapkan ({len(expected_cols)}). Kolom yang hilang akan diisi NA.")
                        df_temp.columns = [str(col) if pd.notna(col) else f"Unnamed_Col{i}" for i, col in enumerate(df_temp.iloc[0])]
                        df_temp = df_temp[1:].copy()

                    df_processed = df_temp.copy()
                    print(f"CSV berhasil diproses dengan delimiter: '{delim}'")
                    break
                else:
                    print(f"Header '{header_string}' tidak ditemukan di file CSV dengan delimiter '{delim}'. Mencoba delimiter lain.")
                    continue

            except Exception as e:
                print(f"Error saat membaca CSV dengan delimiter '{delim}': {e}")
                continue
    else:
        print(f"Error: Ekstensi file '{file_extension}' tidak didukung. Mohon gunakan .csv atau .xlsx.")
        return None

    if df_processed is not None and not df_processed.empty:
        final_df_cols_data = {}
        for col in expected_cols:
            found_col_name = None
            for existing_col in df_processed.columns:
                if existing_col.strip() == col.strip():
                    found_col_name = existing_col
                    break
            
            if found_col_name:
                final_df_cols_data[col] = df_processed[found_col_name]
            else:
                final_df_cols_data[col] = pd.NA 
        df_processed = pd.DataFrame(final_df_cols_data)
        
        df_processed.dropna(how='all', inplace=True)
        df_processed.reset_index(drop=True, inplace=True)

        for col in expected_cols:
            if col in df_processed.columns:
                try:
                    if 'Timestamp' in col:
                        df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                    elif 'Temperature' in col or 'Humidity' in col or 'CO2' in col or 'Light Intensity' in col:
                        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    elif 'Motion Detected' in col:
                        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').astype('Int64')
                    elif 'Ventilation Status' in col:
                        df_processed[col] = df_processed[col].astype(str)
                except Exception as e:
                    print(f"WARNING: Gagal mengkonversi tipe data untuk kolom '{col}': {e}")
        
        critical_cols_for_dropna = [
            "Temperature (?C)", "Humidity (%)", "CO2 (ppm)", 
            "Light Intensity (lux)", "Motion Detected"
        ]
        df_processed.dropna(subset=critical_cols_for_dropna, inplace=True)

        print("\nData Historis berhasil dimuat dan diproses.")
        print("--- Seluruh Data Historis (sebelum imputasi) ---")
        print(df_processed.to_string())
        print("--------------------------------------------------\n")
        return df_processed
    else:
        print(f"\nTidak dapat memproses file '{file_path}' dengan delimiter yang dicoba atau file kosong.")
        return None

# --- Fungsi Memfilter Outlier dan Memperpendek Data ---
def process_and_shorten_data(df, target_rows=3000):
    """
    Memfilter data dari outlier, mengurutkan berdasarkan Timestamp,
    dan memperpendek data hingga jumlah baris target.
    CATATAN: Filter outlier berdasarkan rentang nilai telah dihapus.
    """
    print("\n--- Memfilter Outlier dan Memperpendek Data ---")
    df_filtered = df.copy()

    # --- 1. Filter Outlier (Kriteria "Data Masuk Akal") - BAGIAN INI DIHAPUS ---
    # print("Menerapkan filter outlier berdasarkan rentang nilai sensor...")
    # df_filtered = df_filtered[
    #     (df_filtered["Temperature (?C)"] >= 0) & (df_filtered["Temperature (?C)"] <= 50)
    # ]
    # df_filtered = df_filtered[
    #     (df_filtered["Humidity (%)"] >= 0) & (df_filtered["Humidity (%)"] <= 100)
    # ]
    # df_filtered = df_filtered[
    #     (df_filtered["CO2 (ppm)"] >= 300) & (df_filtered["CO2 (ppm)"] <= 5000)
    # ]
    # df_filtered = df_filtered[
    #     (df_filtered["Light Intensity (lux)"] >= 0) & (df_filtered["Light Intensity (lux)"] <= 2000)
    # ]
    # print(f"Jumlah baris setelah filter outlier: {len(df_filtered)}.")

    # --- 2. Mengurutkan Data berdasarkan Timestamp ---
    if 'Timestamp' in df_filtered.columns and not df_filtered['Timestamp'].empty:
        df_filtered = df_filtered.sort_values(by='Timestamp').reset_index(drop=True)
        print("Data berhasil diurutkan berdasarkan Timestamp.")
    else:
        print("Peringatan: Kolom 'Timestamp' tidak ditemukan atau kosong, data tidak diurutkan.")

    # --- 3. Memperpendek Data ---
    if len(df_filtered) > target_rows:
        df_shortened = df_filtered.head(target_rows).copy()
        print(f"Data dipersingkat menjadi {target_rows} baris.")
    else:
        df_shortened = df_filtered.copy()
        print(f"Jumlah baris ({len(df_filtered)}) kurang dari atau sama dengan target ({target_rows}), tidak perlu memperpendek.")

    print("--- Pemrosesan Data Selesai ---")
    print(f"Jumlah baris final: {len(df_shortened)}")
    return df_shortened


# --- Implementasi Logika KNN ---
def train_knn_model(df):
    """
    Melatih model K-Nearest Neighbors (KNN) menggunakan data historis.
    """
    feature_cols = [
        "Temperature (?C)", "Humidity (%)", "CO2 (ppm)",
        "Light Intensity (lux)", "Motion Detected"
    ]
    target_col = "Ventilation Status"

    if not all(col in df.columns for col in feature_cols + [target_col]):
        print("Error: Kolom fitur atau target tidak ditemukan untuk pelatihan KNN. Pastikan header sesuai.")
        return None, None, None, None

    # Mengubah target menjadi numerik: 'Open'/'ON' ke 1, 'Closed'/'OFF' ke 0
    # Menggunakan pd.to_numeric dengan errors='coerce' untuk mengubah nilai yang tidak dapat dipetakan menjadi NaN,
    # kemudian mengisi NaN dengan 0 sebelum mengubah ke int.
    df[target_col] = df[target_col].replace({'Open': 1, 'ON': 1, 'Closed': 0, 'OFF': 0})
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int) # <--- PERBAIKAN DI SINI
    
    X = df[feature_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_neighbors = int(np.sqrt(len(df)))
    if k_neighbors == 0: k_neighbors = 1 
    if k_neighbors % 2 == 0: k_neighbors += 1 

    knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn_model.fit(X_scaled, y)
    
    print(f"\nModel KNN dilatih dengan K={k_neighbors} menggunakan {len(df)} sampel.")
    return knn_model, scaler, feature_cols, k_neighbors 

def run_knn_logic(knn_model, scaler, historical_data_df, k_neighbors, sensor_data):
    """
    Menjalankan logika KNN untuk memprediksi output relay berdasarkan data sensor baru.
    Menampilkan tetangga terdekat dari data historis.
    Output servo diatur ke 0 (default/non-aktif).
    """
    input_data = np.array(sensor_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    
    # Dapatkan jarak dan indeks tetangga terdekat
    distances, indices = knn_model.kneighbors(input_data_scaled, n_neighbors=k_neighbors)
    
    print("\n--- Tetangga Terdekat KNN Ditemukan ---")
    for i, idx in enumerate(indices[0]):
        if idx < len(historical_data_df):
            neighbor_data = historical_data_df.iloc[idx]
            print(f"Tetangga {i+1} (Indeks: {idx}, Jarak: {distances[0][i]:.4f}):")
            print(neighbor_data.to_string())
            print("---")
        else:
            print(f"Peringatan: Indeks tetangga {idx} di luar rentang DataFrame historis.")
    print("---------------------------------------\n")

    prediction = knn_model.predict(input_data_scaled)[0]
    
    # OUTPUT BERDASARKAN VENTILATION STATUS:
    output_servo_angle = 0 
    output_relay_status = "RELAY_OFF"

    if prediction == 1: # Prediksi: 'Open' / 'ON'
        output_servo_angle = 90 
        output_relay_status = "RELAY_ON" 
    else: # Prediksi: 'Closed' / 'OFF'
        output_servo_angle = 0 
        output_relay_status = "RELAY_OFF" 
    
    return output_servo_angle, output_relay_status 

# --- Implementasi Logika Fuzzy ---
def setup_fuzzy_logic():
    """
    Menyiapkan sistem Fuzzy Logic dengan variabel, fungsi keanggotaan, dan aturan.
    Fokus pada kendali Relay dan Servo berdasarkan Ventilation Status.
    """
    # Definisikan variabel input fuzzy (Universe of Discourse)
    temperature = ctrl.Antecedent(np.arange(0, 40, 1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    co2 = ctrl.Antecedent(np.arange(300, 2000, 1), 'co2')
    light = ctrl.Antecedent(np.arange(0, 1000, 1), 'light')
    motion = ctrl.Antecedent(np.arange(-0.5, 1.5, 1), 'motion') 

    # Definisikan variabel output fuzzy
    ventilation_status_fuzzy = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'ventilation_status_fuzzy')

    # Definisikan fungsi keanggotaan (Membership Functions)
    temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
    temperature['medium'] = fuzz.trimf(temperature.universe, [15, 25, 35])
    temperature['high'] = fuzz.trimf(temperature.universe, [30, 40, 40])

    humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 40])
    humidity['medium'] = fuzz.trimf(humidity.universe, [30, 60, 90])
    humidity['high'] = fuzz.trimf(humidity.universe, [80, 100, 100])

    co2['low'] = fuzz.trimf(co2.universe, [300, 300, 550]) 
    co2['medium'] = fuzz.trimf(co2.universe, [500, 1000, 1400]) 
    co2['high'] = fuzz.trimf(co2.universe, [1300, 2000, 2000])

    light['dark'] = fuzz.trimf(light.universe, [0, 0, 250]) 
    light['medium'] = fuzz.trimf(light.universe, [150, 500, 850]) 
    light['bright'] = fuzz.trimf(light.universe, [600, 1000, 1000])

    motion['no'] = fuzz.trimf(motion.universe, [-0.5, 0, 0.5])
    motion['yes'] = fuzz.trimf(motion.universe, [0.5, 1, 1.5])

    # Output Fuzzy for Ventilation Status
    ventilation_status_fuzzy['closed'] = fuzz.trimf(ventilation_status_fuzzy.universe, [0, 0, 0.4])
    ventilation_status_fuzzy['default_or_transition'] = fuzz.trimf(ventilation_status_fuzzy.universe, [0.3, 0.5, 0.7])
    ventilation_status_fuzzy['open'] = fuzz.trimf(ventilation_status_fuzzy.universe, [0.6, 1, 1])

    # Definisikan Aturan Fuzzy (Contoh Aturan)
    rules = [
        ctrl.Rule(co2['high'] | temperature['high'] | humidity['high'], ventilation_status_fuzzy['open']),
        ctrl.Rule(motion['yes'] & (temperature['high'] | humidity['high']), ventilation_status_fuzzy['open']),
        
        ctrl.Rule(co2['low'] & humidity['low'] & temperature['low'], ventilation_status_fuzzy['closed']),
        ctrl.Rule(light['bright'] & motion['no'], ventilation_status_fuzzy['closed']), 
        
        ctrl.Rule(temperature['medium'] & humidity['medium'] & co2['medium'], ventilation_status_fuzzy['default_or_transition']),
        ctrl.Rule(co2['medium'] & light['medium'], ventilation_status_fuzzy['default_or_transition']),
    ]

    system_control = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system_control)
    
    print("\nSistem Fuzzy Logic disiapkan. Output akan merepresentasikan 'Ventilation Status'.")
    return sim

def run_fuzzy_logic(fuzzy_sim, sensor_data):
    """
    Menjalankan inferensi Fuzzy Logic untuk memprediksi output berdasarkan data sensor baru.
    Output servo dan relay akan disesuaikan berdasarkan 'Ventilation Status' fuzzy.
    """
    temp, hum, co2_val, light_val, motion_val = sensor_data

    try:
        fuzzy_sim.input['temperature'] = temp
        fuzzy_sim.input['humidity'] = hum
        fuzzy_sim.input['co2'] = co2_val
        fuzzy_sim.input['light'] = light_val
        fuzzy_sim.input['motion'] = motion_val
    except ValueError as e:
        print(f"Error: Input sensor melebihi jangkauan fuzzy universe: {e}")
        return 0, "RELAY_OFF" 

    fuzzy_sim.compute()

    ventilation_crisp = fuzzy_sim.output.get('ventilation_status_fuzzy', 0.0) 

    if ventilation_crisp >= 0.6:
        output_servo_angle = 90 
        output_relay_status = "RELAY_ON" 
    elif ventilation_crisp <= 0.4:
        output_servo_angle = 0 
        output_relay_status = "RELAY_OFF" 
    else: 
        output_servo_angle = 0 
        output_relay_status = "RELAY_OFF" 
    
    return output_servo_angle, output_relay_status 

# --- Fungsi Komunikasi Serial ---
def send_command_to_esp32(ser_conn, command):
    """Mengirim perintah ke ESP32 melalui Serial Port jika koneksi aktif."""
    if ser_conn and ser_conn.is_open:
        try:
            ser_conn.write(f"{command}\n".encode('utf-8')) 
            print(f"[PC -> ESP32] Mengirim perintah: {command}")
            time.sleep(0.1) 
        except serial.SerialException as e:
            print(f"Error serial saat mengirim perintah: {e}")
        except Exception as e:
            print(f"Error tak terduga saat mengirim perintah: {e}")
    else:
        print("Koneksi serial tidak aktif. Tidak dapat mengirim perintah.")

# --- Fungsi Utama Program ---
if __name__ == "__main__":
    # --- LANGKAH 1: Memuat, Memfilter, dan Memperpendek Data ---
    historical_data_df_raw = read_data_from_file(RAW_DATA_FILE, HEADER_STRING, expected_columns)
    
    if historical_data_df_raw is None or historical_data_df_raw.empty:
        print("Program dihentikan karena tidak dapat membaca data historis mentah.")
        sys.exit(1)

    # Memfilter Outlier, Mengurutkan, dan Memperpendek Data (tanpa imputasi)
    final_historical_data_df = process_and_shorten_data(historical_data_df_raw, target_rows=3000)

    try:
        final_historical_data_df.to_excel(CLEAN_DATA_FILE, index=False)
        print(f"\nFile Excel data historis yang sudah dibersihkan (tanpa imputasi) dan dipersingkat:")
        print(f"'{CLEAN_DATA_FILE}' berhasil dibuat. Data ini bisa digunakan untuk model Anda.")
        print(f"--- Seluruh data dari file '{CLEAN_DATA_FILE}':\n{final_historical_data_df.to_string()}")
    except Exception as e:
        print(f"Error saat menyimpan file Excel hasil pemrosesan: {e}")
        print("Program akan melanjutkan dengan data yang diproses di memori.")


    # --- LANGKAH 3: Menyiapkan Logika dan Mode Operasi Menggunakan DATA YANG TELAH DIPROSES ---
    historical_data_for_models = final_historical_data_df 

    knn_model, scaler, knn_feature_cols = None, None, None 
    k_neighbors_val = None 

    fuzzy_sim = None

    print("\n--- Pilih Logika yang Digunakan ---")
    print("1. KNN (K-Nearest Neighbors)")
    print("2. Fuzzy Logic")
    logic_choice = input("Masukkan pilihan logika (1/2): ").strip()

    if logic_choice == '1':
        print("\n[SETUP LOGIKA] Menyiapkan Logika KNN...")
        knn_model, scaler, knn_feature_cols, k_neighbors_val = train_knn_model(historical_data_for_models) 
        if knn_model is None:
            print("Gagal menyiapkan model KNN. Keluar.")
            sys.exit(1)
    elif logic_choice == '2':
        print("\n[SETUP LOGIKA] Menyiapkan Logika Fuzzy...")
        fuzzy_sim = setup_fuzzy_logic()
    else:
        print("Pilihan logika tidak valid. Keluar dari program.")
        sys.exit(1)

    print("\n--- Pilih Mode Operasi ---")
    print("1. Mode Hardcode (menggunakan data internal untuk simulasi)")
    print("2. Mode Auto (mengambil data dari ESP32 secara real-time)")
    print("Ketik 'quit' untuk keluar.")

    mode_choice = input("Masukkan pilihan mode (1/2/quit): ").strip().lower()

    ser_connection = None 

    try:
        if mode_choice == '1':
            print("\n--- Memulai Mode Hardcode ---")
            sensor_data_for_hardcode = (
                HARDCODE_SENSOR_DATA["temp"],
                HARDCODE_SENSOR_DATA["hum"],
                HARDCODE_SENSOR_DATA["co2"],
                HARDCODE_SENSOR_DATA["light"],
                HARDCODE_SENSOR_DATA["motion"]
            )
            print(f"\nData Sensor Hardcode: {sensor_data_for_hardcode}")

            output_servo_angle = 0 
            output_relay_status = "RELAY_OFF"

            start_time_logic = time.time() 
            if logic_choice == '1':
                output_servo_angle, output_relay_status = run_knn_logic(knn_model, scaler, historical_data_for_models, k_neighbors_val, sensor_data_for_hardcode)
                print(f"Hasil KNN: Servo Angle={output_servo_angle}, Relay Status={output_relay_status}")
            elif logic_choice == '2':
                output_servo_angle, output_relay_status = run_fuzzy_logic(fuzzy_sim, sensor_data_for_hardcode)
                print(f"Hasil Fuzzy Logic: Servo Angle={output_servo_angle}, Relay Status={output_relay_status}")
            end_time_logic = time.time() 
            runtime_ms = (end_time_logic - start_time_logic) * 1000
            print(f"Runtime Kalkulasi Logika: {runtime_ms:.2f} ms") 

            try:
                ser_connection = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                time.sleep(2)
                print(f"Terhubung ke {SERIAL_PORT} untuk kontrol di mode hardcode.")
            except serial.SerialException as e:
                print(f"Peringatan: Tidak dapat terhubung ke ESP32 di mode hardcode: {e}")
                ser_connection = None

            if ser_connection:
                send_command_to_esp32(ser_connection, f"SERVO:{output_servo_angle}")
                send_command_to_esp32(ser_connection, output_relay_status)
            else:
                print("Tidak dapat mengirim perintah kontrol ke ESP32 (tidak terhubung).")

            print("\nMode Hardcode Selesai. Tekan Enter untuk keluar.")
            input()

        elif mode_choice == '2':
            print("\n--- Memulai Mode Auto (dari ESP32) ---")
            try:
                ser_connection = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                time.sleep(2)
                print(f"Terhubung ke {SERIAL_PORT} dengan baud rate {BAUD_RATE}")
                print("Menunggu data dari ESP32... Tekan Ctrl+C untuk menghentikan.")

                while True:
                    if ser_connection.in_waiting > 0:
                        line = ser_connection.readline().decode('utf-8').strip()
                        print(f"\n[ESP32 -> PC] Menerima: {line}") 

                        try:
                            sensor_values_str = line.split(',')
                            if len(sensor_values_str) >= 5:
                                sensor_data_from_esp = (
                                    float(sensor_values_str[0]),    # Temperatur
                                    float(sensor_values_str[1]),    # Kelembaban
                                    float(sensor_values_str[2]),    # CO2
                                    float(sensor_values_str[3]),    # Intensitas Cahaya
                                    int(float(sensor_values_str[4])) # Deteksi Gerak (jadikan int dari float)
                                )
                                print(f"Data Sensor dari ESP32: {sensor_data_from_esp}")

                                output_servo_angle = 0 
                                output_relay_status = "RELAY_OFF"

                                start_time_logic = time.time() 
                                if logic_choice == '1':
                                    output_servo_angle, output_relay_status = run_knn_logic(knn_model, scaler, historical_data_for_models, k_neighbors_val, sensor_data_from_esp)
                                    print(f"Hasil KNN: Servo Angle={output_servo_angle}, Relay Status={output_relay_status}")
                                elif logic_choice == '2':
                                    output_servo_angle, output_relay_status = run_fuzzy_logic(fuzzy_sim, sensor_data_from_esp)
                                    print(f"Hasil Fuzzy Logic: Servo Angle={output_servo_angle}, Relay Status={output_relay_status}")
                                end_time_logic = time.time() 
                                runtime_ms = (end_time_logic - start_time_logic) * 1000
                                print(f"Runtime Kalkulasi Logika: {runtime_ms:.2f} ms") 

                                send_command_to_esp32(ser_connection, f"SERVO:{output_servo_angle}")
                                send_command_to_esp32(ser_connection, output_relay_status)
                            else:
                                print("Peringatan: Data sensor tidak lengkap atau tidak valid dari ESP32.")
                        except ValueError:
                            print(f"Peringatan: Gagal memparsing data sensor dari ESP32 (format salah?): '{line}'")
                        except Exception as e:
                            print(f"Error saat memproses data sensor dari ESP32: {e}")

                    time.sleep(0.5)

            except serial.SerialException as e:
                print(f"\nError: Tidak dapat membuka port serial {SERIAL_PORT}.")
                print("Pastikan ESP32 terhubung dengan kabel dan driver terinstal, serta tidak digunakan oleh aplikasi lain.")
                print(f"Detail error: {e}")
            except KeyboardInterrupt:
                print("\nProgram dihentikan oleh pengguna (Ctrl+C).")

        elif mode_choice == 'quit':
            print("Keluar dari program.")
        else:
            print("Pilihan mode tidak valid. Keluar dari program.")

    except Exception as e:
        print(f"Terjadi error tak terduga di bagian utama program: {e}")
    finally:
        if ser_connection and ser_connection.is_open:
            ser_connection.close()
            print("Port serial ditutup.")