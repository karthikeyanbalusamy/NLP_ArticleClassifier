# News Article Classification using NLP & Deep Learning

## Project Overview
This project classifies news articles into predefined categories using **NLP and Deep Learning**.  
It demonstrates an end-to-end text classification workflow including data scraping, preprocessing, model experimentation, and evaluation.

## Objective
Build a system that automatically categorizes news articles, helping media & publishing platforms organize content and improve user experience.

## Dataset
- Articles collected using **web scraping** (Requests, BeautifulSoup)
- Text cleaned and structured for supervised learning
- Multiple categories representing news topics

> *Dataset not publicly shared due to web scraping constraints.*

## Techniques & Workflow
### **1️⃣ Data Pipeline**
- Web scraping
- Text cleaning (lowercasing, punctuation removal, stopwords)
- Tokenization & embeddings
- Sequence padding

### **2️⃣ Deep Learning Models**
Model | Technique
--- | ---
LSTM | Sequential text modeling
GRU | Gated recurrent variant with faster convergence
BERT | Transformer-based language model (best performance so far)

### **3️⃣ Training & Evaluation**
- Train/test split
- Loss, accuracy tracking with `tqdm`
- Model comparison across architectures

## Results
- **Current accuracy:** ~64% (baseline achieved)
- **BERT** models showed stronger performance vs RNN models
- Actively improving performance with hyperparameter tuning and training strategies

## Next Steps
- Hyperparameter tuning for BERT
- Experiment with **RoBERTa / DistilBERT**
- Add attention visualization & explainability (LIME/SHAP)
- Build an inference API / Streamlit UI

## Tech Stack
- Python
- Pandas, NumPy
- NLTK, Gensim
- TensorFlow / Keras, PyTorch
- HuggingFace Transformers
- tqdm
- Requests, BeautifulSoup
