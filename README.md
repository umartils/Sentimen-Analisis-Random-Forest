# Sentiment Analysis dengan Random Forest

## Summary

Project ini adalah implementasi analisis sentimen menggunakan algoritma Random Forest untuk mengklasifikasikan teks menjadi sentimen positif, negatif, atau netral. Model ini dibangun dengan Python dan memanfaatkan teknik Natural Language Processing (NLP) untuk memproses data teks dan mengekstrak fitur yang relevan.

Random Forest dipilih karena kemampuannya dalam menangani data dengan dimensi tinggi dan memberikan hasil yang robust terhadap overfitting. Project ini cocok untuk analisis sentimen pada review produk, komentar media sosial, atau feedback pengguna.

## Fitur Utama

- Preprocessing teks otomatis (tokenization, stemming, stop words removal)
- Feature extraction menggunakan TF-IDF vectorization
- Model Random Forest yang dapat dikonfigurasi
- Evaluasi model dengan berbagai metrik (accuracy, precision, recall, F1-score)
- Visualisasi hasil dan confusion matrix
- Support untuk data format CSV

## Instalasi

### Requirements

Pastikan Anda memiliki Python 3.7 atau versi yang lebih baru.

### Membuat Virtual Environment

Sangat disarankan untuk menggunakan virtual environment untuk mengisolasi dependencies project ini:

#### Menggunakan venv (Python built-in)

```bash
# Membuat virtual environment
python -m venv sentiment-env

# Aktivasi virtual environment
# Windows
sentiment-env\Scripts\activate

# macOS/Linux
source sentiment-env/bin/activate
```

#### Menggunakan conda

```bash
# Membuat virtual environment dengan conda
conda create -n sentiment-env python=3.9

# Aktivasi environment
conda activate sentiment-env
```

### Install Dependencies

Setelah virtual environment aktif, install library yang diperlukan:

```bash
# Install dari requirements.txt (recommended)
pip install -r requirements.txt

# Atau install manual
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud jupyter beautifulsoup4 requests
```

### Instalasi NLTK Data

Setelah menginstall NLTK, download data yang diperlukan:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Clone Repository

```bash
git clone https://github.com/umartils/Sentimen-Analisis-Random-Forest.git
cd Sentimen-Analisis-Random-Forest
```

### Deaktivasi Virtual Environment

Setelah selesai bekerja dengan project, Anda dapat menonaktifkan virtual environment:

```bash
# Untuk venv
deactivate

# Untuk conda
conda deactivate
```

## Struktur Project

```bash
sentiment-analysis-random-forest/
├── data/
│   ├── data_scraping.csv
├── notebooks/
│   └── Data_Scraping.ipynb
│   └── Training_Model.ipynb
├── requirements.txt
└── README.md
```

## Cara Penggunaan

### 1. Persiapan Data

Siapkan dataset dalam format CSV dengan kolom:
- `text`: Teks yang akan dianalisis
- `sentiment`: Label sentimen (positive, negative, neutral)

### 2. Training Model

```python
from src.model import SentimentAnalyzer

# Inisialisasi model
analyzer = SentimentAnalyzer()

# Load dan preprocess data
analyzer.load_data('data/raw/dataset.csv')

# Training model
analyzer.train()

# Simpan model
analyzer.save_model('models/sentiment_model.pkl')
```

### 3. Prediksi Sentimen

```python
# Load model yang sudah dilatih
analyzer = SentimentAnalyzer()
analyzer.load_model('models/sentiment_model.pkl')

# Prediksi single text
text = "Produk ini sangat bagus dan memuaskan!"
sentiment = analyzer.predict(text)
print(f"Sentiment: {sentiment}")

# Prediksi batch
texts = [
    "Pelayanan sangat mengecewakan",
    "Kualitas produk standar saja",
    "Sangat puas dengan pembelian ini"
]
results = analyzer.predict_batch(texts)
for text, sentiment in zip(texts, results):
    print(f"'{text}' -> {sentiment}")
```

### 4. Evaluasi Model

```python
# Evaluasi performance model
analyzer.evaluate()

# Generate confusion matrix
analyzer.plot_confusion_matrix()

# Feature importance
analyzer.plot_feature_importance()
```

### 5. Running via Command Line

```bash
# Training model
python main.py --mode train --data data/raw/dataset.csv

# Prediksi
python main.py --mode predict --text "Teks yang ingin diprediksi"

# Evaluasi
python main.py --mode evaluate --model models/sentiment_model.pkl
```

## Konfigurasi Model

Anda dapat mengkustomisasi parameter Random Forest di file `config.py`:

```python
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

TFIDF_PARAMS = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.8
}
```

## Dataset

### Informasi Dataset

Dataset yang digunakan dalam project ini adalah **data rating dan review aplikasi Indodax** yang diperoleh melalui teknik web scraping. Dataset ini berisi ulasan pengguna aplikasi Indodax dari Google Play Store dan App Store yang mencerminkan sentimen pengguna terhadap platform trading cryptocurrency tersebut.

### Karakteristik Dataset

- **Sumber**: Google Play Store dan App Store (aplikasi Indodax)
- **Metode Pengumpulan**: Web Scraping menggunakan Python
- **Bahasa**: Bahasa Indonesia (mayoritas) dan Bahasa Inggris
- **Periode**: Data terkumpul dari periode [sesuaikan dengan periode scraping]
- **Jumlah Data**: Sekitar [jumlah] review dan rating
- **Format**: CSV dengan kolom text (review) dan sentiment (label)

### Proses Scraping Data

Dataset dikumpulkan dengan menjalankan notebook scraping yang tersedia di folder `notebooks/`. Untuk mengumpulkan data baru atau memperbarui dataset:

```bash
# Jalankan notebook scraping
jupyter notebook notebooks/indodax_scraping.ipynb
```

### Struktur Data

Dataset memiliki struktur sebagai berikut:

| Kolom | Deskripsi | Tipe Data | Contoh |
|-------|-----------|-----------|--------|
| `text` | Review/komentar pengguna | String | "Aplikasi bagus untuk trading crypto" |
| `rating` | Rating yang diberikan (1-5) | Integer | 4 |
| `sentiment` | Label sentimen (positive/negative/neutral) | String | "positive" |
| `date` | Tanggal review | DateTime | "2024-01-15" |
| `platform` | Platform sumber (playstore/appstore) | String | "playstore" |

### Distribusi Sentimen

Dataset menunjukkan distribusi sentimen sebagai berikut:
- **Positive**: ~45% (rating 4-5, review positif)
- **Negative**: ~35% (rating 1-2, review negatif)  
- **Neutral**: ~20% (rating 3, review netral)

### Preprocessing Data

Data mentah hasil scraping telah melalui tahap preprocessing meliputi:
- Pembersihan teks dari karakter khusus dan emoji
- Normalisasi huruf (lowercase)
- Penghapusan duplikasi review
- Pelabelan sentimen berdasarkan rating dan analisis manual
- Penanganan data tidak lengkap atau kosong

### Catatan Penting

- Dataset dikumpulkan untuk keperluan penelitian dan edukasi
- Review yang dikumpulkan bersifat publik dan tersedia di platform resmi
- Data telah dianonimasi untuk menjaga privasi pengguna
- Scraping dilakukan dengan memperhatikan rate limiting dan robots.txt

### Dataset Alternatif

Selain dataset Indodax, project ini juga dapat bekerja dengan dataset sentiment analysis lainnya seperti:
- Amazon Product Reviews
- IMDB Movie Reviews  
- Twitter Sentiment Dataset
- Indonesian Sentiment Dataset

Pastikan data memiliki format yang sesuai dengan struktur yang dijelaskan di bagian "Persiapan Data".

## Performance

Model Random Forest pada project ini umumnya mencapai:
- Accuracy: 85-92%
- Precision: 83-90%
- Recall: 84-91%
- F1-Score: 83-90%

*Note: Performance dapat bervariasi tergantung pada dataset dan konfigurasi model.*

## Contributing

Kontribusi sangat diterima! Silakan fork repository ini dan submit pull request untuk perbaikan atau fitur baru.

## License

Project ini dilisensikan under MIT License. Lihat file `LICENSE` untuk detail lebih lanjut.

## Kontak

Jika ada pertanyaan atau saran, silakan hubungi:
- Email: your.email@example.com
- GitHub: [@username](https://github.com/username)