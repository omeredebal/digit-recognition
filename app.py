from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
from tensorflow import keras
import cv2

app = Flask(__name__)

# Modeli yükle
print("🔄 Model yükleniyor...")
model = keras.models.load_model('model/digit_model.h5')
print("✅ Model başarıyla yüklendi!")

def preprocess_image(image_data):
    """Çizilen görüntüyü model için hazırla"""
    
    # Base64 string'i decode et
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    # PIL Image'a dönüştür (RGBA olarak aç)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
    
    # Beyaz arka plan oluştur ve alpha kanalını işle
    background = Image.new('RGBA', image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image)
    
    # Gri tonlamaya çevir
    image = image.convert('L')
    
    # Numpy array'e dönüştür
    image_array = np.array(image)
    
    # Görüntüyü ters çevir (beyaz arka plan → siyah arka plan, siyah çizgi → beyaz çizgi)
    image_array = 255 - image_array
    
    # Debug: Görüntüde çizim var mı kontrol et
    print(f"📊 Görüntü max değer: {image_array.max()}, min değer: {image_array.min()}")
    
    # Rakamın etrafındaki boşlukları kırp
    rows = np.any(image_array > 30, axis=1)
    cols = np.any(image_array > 30, axis=0)
    
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Biraz padding ekle
        padding = 20
        rmin = max(0, rmin - padding)
        rmax = min(image_array.shape[0], rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(image_array.shape[1], cmax + padding)
        
        image_array = image_array[rmin:rmax+1, cmin:cmax+1]
    else:
        # Eğer çizim yoksa boş bir array döndür
        print("⚠️ Görüntüde çizim bulunamadı!")
    
    # Kare şekline getir (en-boy oranını koru)
    h, w = image_array.shape
    if h > w:
        pad = (h - w) // 2
        image_array = np.pad(image_array, ((0, 0), (pad, pad)), mode='constant', constant_values=0)
    elif w > h:
        pad = (w - h) // 2
        image_array = np.pad(image_array, ((pad, pad), (0, 0)), mode='constant', constant_values=0)
    
    # 28x28 boyutuna getir
    image_array = cv2.resize(image_array, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize et
    image_array = image_array.astype('float32') / 255.0
    
    # Boyutları düzenle: (28, 28) → (1, 28, 28, 1)
    image_array = image_array.reshape(1, 28, 28, 1)
    
    print(f"✅ İşlenmiş görüntü shape: {image_array.shape}, max: {image_array.max():.2f}")
    
    return image_array

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Tahmin endpoint'i"""
    try:
        # Gelen veriyi al
        data = request.get_json()
        image_data = data['image']
        
        # Görüntüyü işle
        processed_image = preprocess_image(image_data)
        
        # Tahmin yap
        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)
        
        # Tüm sınıfların olasılıklarını al
        probabilities = {str(i): float(prediction[0][i] * 100) for i in range(10)}
        
        return jsonify({
            'success': True,
            'digit': predicted_digit,
            'confidence': round(confidence, 2),
            'probabilities': probabilities
        })
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Web Uygulaması Başlatılıyor...")
    print("="*50)
    print("📍 Tarayıcınızda şu adresi açın: http://127.0.0.1:5001")
    print("⛔ Durdurmak için CTRL+C basın")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)