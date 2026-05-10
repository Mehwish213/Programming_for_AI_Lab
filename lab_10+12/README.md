# Library Bot - Intelligent Book Recommendation System

<div align="center">

![Library Bot](https://img.shields.io/badge/Library%20Bot-Book%20Recommender-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An AI-powered book recommendation system that helps you discover your next favorite read.**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Dataset](#dataset)

</div>

---

## 📚 Overview

Library Bot is an intelligent book recommendation system that uses semantic search powered by state-of-the-art machine learning models. Instead of relying on traditional keyword matching, it understands the meaning behind your search queries and finds the most relevant books from a database of thousands of titles.

### Key Highlights

- **AI-Powered Search**: Uses Sentence Transformers for semantic understanding
- **Fast & Efficient**: FAISS indexing for instant search results
- **Extensive Database**: Access to 10,000+ books with detailed metadata
- **User-Friendly Interface**: Modern, responsive web UI
- **Easy to Deploy**: Built with Flask for simple deployment

---

## ✨ Features

### 1. **Semantic Search**
   - Understands natural language queries
   - Finds books by meaning, not just keywords
   - Example: Search "teenage adventure fantasy" instead of exact titles

### 2. **Quick Results**
   - Returns top 5 most relevant book recommendations
   - Displays author, title, and average rating
   - Real-time search performance

### 3. **Comprehensive Database**
   - Data sourced from Goodbooks-10k dataset
   - Includes millions of book reviews and ratings
   - Clean, preprocessed data for accurate results

### 4. **Responsive Design**
   - Works on desktop, tablet, and mobile
   - Modern gradient UI with smooth animations
   - Dark-themed professional interface

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Flask (Python) |
| **ML Models** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Search Engine** | FAISS (Facebook AI Similarity Search) |
| **Data Processing** | Pandas, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Deployment** | Flask Development Server |

---

## 📋 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd lab_10
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Data

The project includes pre-processed data files:
- `cleaned_books.csv` - Processed book data
- `book_embeddings.npy` - Pre-computed embeddings
- `faiss_index.index` - FAISS search index

If you need to regenerate these files, run the Jupyter notebook:

```bash
jupyter notebook library_bot.ipynb
```

### Step 5: Run the Application

```bash
python app.py
```

Visit `http://localhost:5000` in your web browser.

---

## 🚀 Usage

### Basic Search

1. **Open the Application**
   - Navigate to `http://localhost:5000`
   - You'll see the Library Bot homepage

2. **Enter Your Search Query**
   - Type a book title, author name, or genre
   - Example: "mystery thriller", "fantasy adventure", "science fiction"

3. **View Results**
   - Get top 5 recommendations instantly
   - See book titles, authors, and ratings
   - Click on results to learn more (future feature)

### Search Examples

| Query | Expected Results |
|-------|-----------------|
| "magical academy" | Fantasy books with magical settings |
| "detective noir" | Mystery/noir detective stories |
| "space exploration" | Science fiction space adventures |
| "romance love story" | Romance and love-themed books |

---

## 🏗️ Project Architecture

```
lab_10/
├── app.py                          # Flask application main file
├── library_bot.ipynb               # Data processing & embedding notebook
├── requirements.txt                 # Python dependencies
├── cleaned_books.csv               # Processed book metadata
├── book_embeddings.npy             # Pre-computed embeddings
├── faiss_index.index               # FAISS search index
│
├── templates/
│   └── index.html                  # Main HTML template
│
├── static/
│   ├── css/
│   │   └── style.css               # Stylesheet
│   └── js/
│       └── script.js               # JavaScript functionality
│
└── goodbooks-10k/                  # Dataset directory
    ├── books.csv
    ├── ratings.csv
    └── tags.csv
```

---

## 📊 How It Works

### 1. **Data Preparation Phase**
```
Raw Books CSV → Clean Text → Generate Embeddings → Build FAISS Index
```

### 2. **Search Phase**
```
User Query → Clean Query → Generate Embedding → FAISS Search → Return Results
```

### Data Processing Pipeline

```python
# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Combine title and author for better context
df['combined'] = df['title'] + " " + df['authors']

# Generate embeddings using Sentence Transformers
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(df['cleaned_text'].tolist())
```

### Search Algorithm

```python
def search_books(query, count=5):
    # Clean user input
    query = clean_text(query)
    
    # Generate embedding for query
    query_embedding = model.encode([query])
    
    # Search in FAISS index
    distance, indices = index.search(query_embedding, count)
    
    # Retrieve book details
    results = [df.iloc[idx] for idx in indices[0]]
    return results
```

---

## 📦 Dataset Information

### Goodbooks-10k Dataset

**Size**: ~10,000 books with detailed metadata

**Includes**:
- Book titles and authors
- ISBN and publication year
- Average ratings and review counts
- Book descriptions and genres
- User ratings and reviews

**Source**: https://github.com/zygmuntz/goodbooks-10k

---

## 🔧 Configuration

### Model Selection

Currently uses `all-MiniLM-L6-v2` from Sentence Transformers:
- **Pros**: Fast, accurate, good for semantic similarity
- **Size**: ~80MB
- **Performance**: ~100 inferences/second

To change the model, edit `app.py`:

```python
model = SentenceTransformer('model-name-here')
```

### Search Parameters

Modify search results count in `app.py`:

```python
def search_books(query, count=5):  # Change count here
    ...
```

---

## 🎨 Frontend Features

### UI Components

- **Navigation Bar**: Easy navigation between sections
- **Hero Section**: Eye-catching introduction
- **Search Form**: Intuitive search interface
- **Results Grid**: Beautiful book card display
- **About Section**: Feature highlights
- **Responsive Design**: Works on all devices

### Interactive Elements

- Search form with validation
- Book card hover effects
- Star rating display
- Smooth scroll animations
- Mobile-optimized layout

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Search Time** | <100ms |
| **Database Size** | ~10,000 books |
| **Embedding Dimensions** | 384 |
| **Index Type** | FAISS Flat |
| **Memory Usage** | ~150MB |

---

## 🐛 Troubleshooting

### Issue: Module not found error

**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: FAISS index not found

**Solution**: Regenerate index using the notebook
```bash
jupyter notebook library_bot.ipynb
```

### Issue: Slow search results

**Solution**: Restart Flask app
```bash
python app.py
```

---

## 🚀 Future Enhancements

- [ ] User authentication and bookmarks
- [ ] Advanced filtering (genre, rating, year)
- [ ] Book details modal with summaries
- [ ] User ratings and reviews integration
- [ ] Recommendation history
- [ ] Export recommendations to PDF
- [ ] Integration with online bookstores
- [ ] Mobile app version

---

## 📝 Requirements File

```
Flask==2.3.0
pandas==1.5.0
numpy==1.24.0
faiss-cpu==1.7.4
sentence-transformers==2.2.0
scikit-learn==1.2.0
python-docx==0.8.11
```

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Goodbooks-10k Dataset** by Zygmunt Zając
- **Sentence Transformers** by UKPLab
- **FAISS** by Facebook AI Research
- **Flask** for the lightweight framework

---

## 📧 Contact & Support

For questions or support, please open an issue in the repository.

**Created as part of PAI LAB Course**
*Semester 4 | University Project*

---

<div align="center">

**⭐ If you find this project helpful, please give it a star! ⭐**

Made with ❤️ by a student developer

</div>
