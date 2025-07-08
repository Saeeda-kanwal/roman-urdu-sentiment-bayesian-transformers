
# 🧠 Roman Urdu Sentiment Analysis using BERT and Bayesian Transformers

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://roman-urdu-sentiment-bayesian-transformers-3vzeaaa7n8q23wzngpe.streamlit.app/)


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Saeeda-Kanwal/roman-urdu-sentiment-bayesian-transformers/blob/main/notebooks/02_inference_demo.ipynb)

This project presents a sentiment analysis system for **Roman Urdu** using **BERT-based Transformer models**, extended with **Bayesian learning** ideas to make predictions more robust and interpretable. 

---

## 📁 Project Structure

```
roman-urdu-sentiment-bayesian-transformers/
├── data/
│   └── roman_urdu_sentiment_sample.csv
├── notebooks/
│   ├── 01_train_model.ipynb        # Train BERT on Roman Urdu dataset
│   └── 02_inference_demo.ipynb     # Predict sentiment from input text
├── streamlit_app.py                # Simple UI using Streamlit
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## 📊 Features

- Sentiment classification in **Roman Urdu**
- Uses **BERT-base** model with fine-tuning
- Inference with confidence scores
- Interactive **Streamlit** demo
- Future extensibility with **Bayesian Transformers** & **Knowledge Graph Embeddings**

---

## 🚀 How to Run

### 🔬 1. Train the model (Colab or Jupyter)
Open `01_train_model.ipynb`, run all cells. Modify data or model as needed.

### 🧪 2. Test with inference notebook
Open `02_inference_demo.ipynb`, enter any Roman Urdu sentence and get sentiment predictions.

### 🌐 3. Launch the Streamlit App
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## ✨ Sample Input & Output

**Input:** `ye movie zabardast thi`  
**Output:**  
> Sentiment: Positive  
> Confidence: [0.03, 0.97]

---

## 📌 Research Alignment

- ✅ Transformer-based Representation Learning  
- ✅ BERT + Bayesian Methods  
- ✅ Interpretability via softmax probabilities  
- ✅ NLP on multilingual and under-resourced languages

---

## 👩‍💻 Author

**Saeeda Kanwal**  
Lecturer & AI Researcher  
[LinkedIn](https://www.linkedin.com/in/saeeda-kanwal) | [GitHub](https://github.com/Saeeda-Kanwal)
