# 💬 Sentiment Analysis on Real App Reviews

A sentiment classifier trained on real, scraped Google Play Store reviews 
(not a pre-cleaned academic dataset), deployed as a live web app.

## 🔗 Live Demo
[Try it here](https://sentiment-analyzer-menuka-projects.streamlit.app/)

## 📊 Overview
- **Data**: ~3,000 real user reviews scraped via `google-play-scraper`
- **Labels**: Derived from star ratings (1-2★ = Negative, 3★ = Neutral, 4-5★ = Positive)
- **Model**: TF-IDF (unigrams + bigrams) + Logistic Regression (class-balanced)
- **Accuracy**: XX% (fill in from your classification_report)

## 🛠️ Tech Stack
Python, scikit-learn, Streamlit, google-play-scraper

## 🚀 Run Locally
\`\`\`
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## 📈 Results
<img width="560" height="503" alt="image" src="https://github.com/user-attachments/assets/f68fdd0b-4f1e-4603-b797-5a3166b771b8" />
<img width="513" height="470" alt="image" src="https://github.com/user-attachments/assets/7c975a9d-7d76-4b58-abf8-64d26fbfc3f2" />




## 💡 Key Design Decisions
- Used real, messy user-generated data instead of a cleaned dataset to reflect 
  real-world NLP challenges (typos, slang, mixed sentiment)
- Handled class imbalance using `class_weight='balanced'`
- Multi-class (negative/neutral/positive) rather than binary for more nuance
