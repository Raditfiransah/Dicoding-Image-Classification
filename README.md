# Image Classification - Rubbish Dataset

Proyek klasifikasi gambar untuk dataset sampah (rubbish) menggunakan Deep Learning dengan TensorFlow/Keras. Proyek ini dibuat untuk memenuhi submission Dicoding Machine Learning dengan target akurasi **95%+**.

## 📋 Deskripsi

Proyek ini mengimplementasikan model klasifikasi gambar untuk mengklasifikasikan 6 jenis sampah:

- Cardboard (Kardus)
- Glass (Kaca)
- Metal (Logam)
- Organic (Organik)
- Paper (Kertas)
- Plastic (Plastik)

## 🎯 Fitur Utama

- ✅ **Dataset**: 23,066 gambar dari Hugging Face
- ✅ **Akurasi Target**: 95%+
- ✅ **Augmentasi Data**: Rotasi, zoom, flip, shift
- ✅ **Transfer Learning**: MobileNetV2 (pre-trained ImageNet)
- ✅ **Custom Callbacks**: AccuracyThresholdCallback, ReduceLROnPlateau
- ✅ **Class Weights**: Menangani class imbalance
- ✅ **Deployment**: 3 format (SavedModel, TF-Lite, TensorFlow.js)
- ✅ **Visualisasi**: Confusion Matrix, Per-class Accuracy, Training History

## 📊 Dataset

**Source**: [Hugging Face - rubbish_augmented](https://huggingface.co/datasets/Jotadebeese/rubbish_augmented)

**Statistik**:

- Total Images: 23,066
- Training: 18,450 (80%)
- Validation: 2,304 (10%)
- Test: 2,312 (10%)
- Classes: 6

## 🛠️ Teknologi

- **Python**: 3.12+
- **TensorFlow**: 2.19.0
- **Keras**: (included in TensorFlow)
- **MobileNetV2**: Transfer Learning
- **Hugging Face Datasets**: Data loading
- **scikit-learn**: Metrics & utilities

## 📦 Instalasi

### 1. Clone Repository

```bash
git clone <repository-url>
cd Dicoding-Klasifikasi-Gambar
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) GPU Setup

Pastikan CUDA dan cuDNN terinstall untuk training dengan GPU.

## 🚀 Cara Penggunaan

### Training Model

Jalankan notebook `notebook.ipynb` di Google Colab atau Jupyter:

```bash
jupyter notebook notebook.ipynb
```

Atau jalankan di Google Colab:

1. Upload `notebook.ipynb` ke Google Colab
2. Jalankan semua cell secara berurutan
3. Model akan otomatis didownload dan ditraining

### Struktur Notebook

1. **Import Libraries** - Import semua dependencies
2. **Data Loading** - Download dataset dari Hugging Face
3. **Configuration** - Setup hyperparameters
4. **Dataset Analysis** - Analisis distribusi data
5. **Data Augmentation** - Setup augmentasi data
6. **Model Architecture** - Build model dengan MobileNetV2
7. **Custom Callbacks** - AccuracyThresholdCallback
8. **Class Weights** - Menangani class imbalance
9. **Training** - Train model dengan callbacks
10. **Evaluation** - Evaluasi pada test set
11. **Confusion Matrix** - Visualisasi performa
12. **Deployment** - Export 3 format model
13. **Inference** - Demo prediksi dengan TF-Lite

## 🏗️ Arsitektur Model

```
Input (224x224x3)
    ↓
MobileNetV2 (pre-trained, frozen)
    ↓
Conv2D (256 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (128 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.5)
    ↓
Dense (256) + ReLU
    ↓
Dropout (0.3)
    ↓
Dense (6) + Softmax
```

## 📈 Hyperparameters

| Parameter          | Value                    |
| ------------------ | ------------------------ |
| Image Size         | 224x224                  |
| Batch Size         | 64                       |
| Epochs             | 50                       |
| Learning Rate      | 1e-4                     |
| Optimizer          | Adam                     |
| Loss               | Categorical Crossentropy |
| Accuracy Threshold | 95%                      |

## 🎨 Data Augmentation

- **Rotation**: ±20°
- **Width Shift**: ±20%
- **Height Shift**: ±20%
- **Shear**: ±20%
- **Zoom**: ±20%
- **Horizontal Flip**: True
- **Fill Mode**: Nearest

## 📊 Callbacks

1. **AccuracyThresholdCallback**: Stop training saat mencapai 95% accuracy
2. **ModelCheckpoint**: Save best model berdasarkan val_accuracy
3. **ReduceLROnPlateau**: Kurangi learning rate saat val_loss plateau
4. **EarlyStopping**: Stop training jika tidak ada improvement

## 💾 Model Deployment

Model disimpan dalam 3 format:

### 1. SavedModel

```
saved_model/
├── saved_model.pb
└── variables/
```

### 2. TF-Lite

```
tflite/
├── model.tflite
└── labels.txt
```

### 3. TensorFlow.js

```
tfjs_model/
├── group1-shard1of1.bin
└── model.json
```

## 📊 Hasil Evaluasi

Model akan menghasilkan:

- ✅ Training/Validation Accuracy & Loss plots
- ✅ Confusion Matrix (counts & percentages)
- ✅ Per-class Accuracy bar chart
- ✅ Classification Report (Precision, Recall, F1-Score)
- ✅ Test Set Evaluation

## 🔍 Inference

### Menggunakan TF-Lite Model

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TF-Lite model
interpreter = tf.lite.Interpreter(model_path='tflite/model.tflite')
interpreter.allocate_tensors()

# Load labels
with open('tflite/labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Predict
img = Image.open('test_image.jpg').resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

predictions = interpreter.get_tensor(output_details[0]['index'])[0]
predicted_class = labels[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print(f"Predicted: {predicted_class} ({confidence:.2f}%)")
```

## 📁 Struktur Project

```
Dicoding-Klasifikasi-Gambar/
├── notebook.ipynb              # Main notebook
├── README.md                   # Dokumentasi
├── requirements.txt            # Dependencies
├── .gitignore                 # Git ignore file
├── dataset_final/             # Dataset (auto-downloaded)
│   ├── train/
│   ├── val/
│   └── test/
├── saved_model/               # SavedModel format
├── tflite/                    # TF-Lite format
│   ├── model.tflite
│   └── labels.txt
├── tfjs_model/                # TensorFlow.js format
├── best_model.keras           # Best model checkpoint
├── training_history.png       # Training plots
├── confusion_matrix.png       # Confusion matrix
└── per_class_accuracy.png     # Per-class accuracy
```

## 🎓 Kriteria Dicoding

Proyek ini memenuhi semua kriteria submission Dicoding:

- ✅ Dataset minimal 1000 gambar (23,066 ✓)
- ✅ Akurasi minimal 85% (Target: 95%+ ✓)
- ✅ Menggunakan Sequential Model ✓
- ✅ Menggunakan Conv2D & MaxPooling2D ✓
- ✅ Custom Callback untuk stop di 95% ✓
- ✅ Augmentasi data ✓
- ✅ Deployment 3 format ✓
- ✅ Visualisasi training history ✓
- ✅ Inference demonstration ✓

## 🐛 Troubleshooting

### TensorFlow.js Conversion Error

Jika terjadi error saat konversi TensorFlow.js:

```bash
pip install tensorflowjs==4.20.0 packaging==23.2
```

### GPU Not Detected

```bash
# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CUDA & cuDNN sesuai TensorFlow version
```

### Dataset Download Error

Jika download otomatis gagal, download manual dari:
https://huggingface.co/datasets/Jotadebeese/rubbish_augmented

## 📝 License

MIT License

## 👨‍💻 Author

**Radit Firansah**

- GitHub: [@raditfiransah](https://github.com/raditfiransah)
- Dicoding: Radit Firansah

## 🙏 Acknowledgments

- Dataset: [Jotadebeese/rubbish_augmented](https://huggingface.co/datasets/Jotadebeese/rubbish_augmented)
- Dicoding Indonesia
- TensorFlow & Keras Team
- MobileNetV2 Architecture

## 📚 References

1. [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
2. [TensorFlow Documentation](https://www.tensorflow.org/)
3. [Keras Documentation](https://keras.io/)
4. [Dicoding Deep Learning Path](https://www.dicoding.com/)

---

**Note**: Proyek ini dibuat untuk submission Dicoding Machine Learning. Model mencapai akurasi 95%+ pada test set dengan menggunakan transfer learning dan data augmentation.
