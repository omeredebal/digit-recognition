# 🧠 Rakam Tanıma Sistemi

CNN (Convolutional Neural Network) ile el yazısı rakam tanıma web uygulaması.

## 🎯 Özellikler

- ✏️ Canvas üzerinde rakam çizme
- 📷 Kamera ile rakam tanıma
- 🤖 %99+ doğruluk oranı
- 📊 Olasılık dağılımı gösterimi

## 🚀 Kurulum

```bash
# Sanal ortam oluştur
python3 -m venv venv
source venv/bin/activate

# Paketleri yükle
pip install -r requirements.txt

# Modeli eğit (ilk seferde)
python3 model_training.py

# Uygulamayı çalıştır
python3 app.py
```

Tarayıcıda aç: http://127.0.0.1:5001

## 👥 Ekip

- Ömer Edebali
- Taha İslam Güven

## 🛠️ Teknolojiler

- Python, Flask
- TensorFlow, Keras
- MNIST Dataset
- HTML, CSS, JavaScript
