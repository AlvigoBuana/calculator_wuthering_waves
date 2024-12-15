from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat model yang sudah dilatih
model = joblib.load('model.pkl')  # Ganti dengan path model Anda

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Halaman untuk menerima input dan menampilkan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil input dari pengguna
        level = int(request.form['level'])
        jml_item_std = int(request.form['jml_item_std'])
        jml_dupe_std = int(request.form['jml_dupe_std'])
        jml_item_limited = int(request.form['jml_item_limited'])
        jml_dupe_limited = int(request.form['jml_dupe_limited'])
        jml_sign = int(request.form['jml_sign'])
        jml_asterite = int(request.form['jml_asterite'])

        # Simpan input ke dalam DataFrame
        input_data = pd.DataFrame({
            'jml_item_limited': [jml_item_limited],
            'jml_dupe_limited': [jml_dupe_limited],
            'jml_sign': [jml_sign]
        })

        # Prediksi harga akun
        predicted_price = model.predict(input_data)

        # Logika bonus harga
        bonus_price = 0
        if level <= 10 and jml_item_limited > 2 and jml_sign > 1:
            bonus_price = 100000  # Tambah 100.000 untuk setiap ketentuan yang dipenuhi

        # Hitung harga total dengan bonus
        total_price = predicted_price[0] + bonus_price

        # Kirim data input dan hasil prediksi ke template
        return render_template('index.html', 
                            predicted_price=total_price,
                            level=level,
                            jml_item_std=jml_item_std,
                            jml_dupe_std=jml_dupe_std,
                            jml_item_limited=jml_item_limited,
                            jml_dupe_limited=jml_dupe_limited,
                            jml_sign=jml_sign,
                            jml_asterite=jml_asterite)

if __name__ == '__main__':
    app.run(debug=True)
