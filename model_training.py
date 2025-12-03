import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Model klasörü oluştur
if not os.path.exists('model'):
    os.makedirs('model')

print("📊 MNIST veri seti yükleniyor...")
# MNIST veri setini yükle
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"✅ Eğitim verisi: {X_train.shape}")
print(f"✅ Test verisi: {X_test.shape}")

# Veriyi normalize et (0-255 → 0-1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Boyutları ayarla (28, 28) → (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Etiketleri kategorik hale getir
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("\n🏗️ CNN Modeli oluşturuluyor...")

# CNN Modeli
model = keras.Sequential([
    # İlk Convolutional Katman
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # İkinci Convolutional Katman
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten ve Dense Katmanlar
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Model derleme
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Özeti:")
model.summary()

print("\n🚀 Model eğitimi başlıyor...")

# Model eğitimi
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

print("\n📈 Model test ediliyor...")

# Test doğruluğu
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Doğruluğu: {test_accuracy * 100:.2f}%")
print(f"✅ Test Kaybı: {test_loss:.4f}")

# Modeli kaydet
model.save('model/digit_model.h5')
print("\n💾 Model 'model/digit_model.h5' olarak kaydedildi!")

# Eğitim grafiklerini çiz
plt.figure(figsize=(12, 4))

# Doğruluk grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.title('Model Doğruluğu')
plt.grid(True)

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.title('Model Kaybı')
plt.grid(True)

plt.tight_layout()
plt.savefig('model/training_history.png', dpi=150, bbox_inches='tight')
print("📊 Eğitim grafikleri 'model/training_history.png' olarak kaydedildi!")

print("\n" + "="*50)
print("🎉 MODEL EĞİTİMİ TAMAMLANDI!")
print("="*50)

# Örnek tahmin
print("\n🔍 Örnek tahmin testi yapılıyor...")
sample_idx = np.random.randint(0, len(X_test))
sample_image = X_test[sample_idx:sample_idx+1]
sample_label = np.argmax(y_test[sample_idx])

prediction = model.predict(sample_image, verbose=0)
predicted_label = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(f"Gerçek Etiket: {sample_label}")
print(f"Tahmin: {predicted_label}")
print(f"Güven: {confidence:.2f}%")
print(f"Sonuç: {'✅ DOĞRU' if sample_label == predicted_label else '❌ YANLIŞ'}")