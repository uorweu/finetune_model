import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers

# --- 1. CẤU HÌNH HỆ THỐNG ---
DATA_PATH = 'audio_dataset/classes' # Đường dẫn ông vừa gửi
MODEL_NAME = 'my_audio_model_v1.keras'
BATCH_SIZE = 64         # Tối ưu cho tập dữ liệu ~1000 file
EPOCHS = 30             # Vừa đủ để đạt 0.9, tránh học vẹt (overfitting)
LEARNING_RATE = 0.0003  # Thông số "vàng" từ tài liệu ICTIS 2024

# --- 2. TẢI BỘ TRÍCH XUẤT ĐẶC TRƯNG YAMNet ---
print("--- Đang kết nối với bộ não YAMNet của Google... ---")
yamnet_layer = hub.KerasLayer('https://tfhub.dev/google/yamnet/1')

# --- 3. KHỐI TIỀN XỬ LÝ (AUDIO CONDITIONING) ---
def preprocess_audio(file_path):
    # Ép về 16kHz (YAMNet requirement)
    wav, _ = librosa.load(file_path, sr=16000)
    
    # Chuẩn hóa biên độ (Global Normalization từ tài liệu ResNet-18)
    if len(wav) > 0:
        wav = (wav - np.mean(wav)) / (np.std(wav) + 1e-7)
    
    # Cố định độ dài 1 giây (16000 samples)
    if len(wav) < 16000:
        wav = np.pad(wav, (0, 16000 - len(wav)))
    else:
        wav = wav[:16000]
    return wav

def get_embeddings(wav_data):
    # Chuyển sang tensor và chạy qua YAMNet
    wav_tensor = tf.convert_to_tensor(wav_data, dtype=tf.float32)
    _, embeddings, _ = yamnet_layer(wav_tensor)
    
    # MEAN POOLING: Gom các lát cắt thành 1 vector 1024 duy nhất
    return tf.reduce_mean(embeddings, axis=0)

# --- 4. KHỐI LOAD DỮ LIỆU ---
def load_dataset():
    X, y = [], []
    # Lấy tên các thư mục con làm nhãn (Labels)
    classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    
    print(f"Tìm thấy {len(classes)} lớp âm thanh: {classes}")
    
    for idx, label in enumerate(classes):
        folder = os.path.join(DATA_PATH, label)
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        print(f"-> Đang xử lý {len(files)} file trong nhóm: {label}")
        
        for f in files:
            try:
                wav = preprocess_audio(os.path.join(folder, f))
                emb = get_embeddings(wav)
                X.append(emb)
                y.append(idx)
            except Exception as e:
                print(f"Lỗi bỏ qua file {f}: {e}")
            
    return np.array(X), np.array(y), classes

# --- 5. KHỐI XÂY DỰNG MODEL PHÂN LOẠI (CLASSIFIER) ---
def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(1024,)), # Nhận 1024 đặc trưng từ YAMNet
        
        # Lớp ẩn 1 với BatchNormalization (Giúp học nhanh & ổn định)
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4), # Chống học vẹt 
        
        # Lớp ẩn 2
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Lớp đầu ra (Số lượng nơ-ron = Số lượng lớp của ông)
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Dùng Adam với Learning Rate nhỏ để hội tụ chính xác
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# --- 6. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # A. Chuẩn bị dữ liệu
    features, labels, class_names = load_dataset()
    
    # B. Chia Train/Test (80% để học, 20% để thi)
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # C. Khởi tạo và Train
    my_model = build_model(len(class_names))
    my_model.summary()
    
    
    
    print("\n🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN...")
    history = my_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # D. Lưu thành quả
    my_model.save(MODEL_NAME)
    
    
    
    print(f"\n✅ Xong! Model đã được lưu tại: {MODEL_NAME}")
    print(f"Nhớ dùng file này để nhận diện thực tế nhé ông!")
