# 📧 SMS Spam Detection System

**Live Demo:** [https://my-sms-app-ersk.onrender.com](https://my-sms-app-ersk.onrender.com)

## 📌 Project Overview
This is a Machine Learning web application designed to classify SMS messages as either **Spam** or **Ham (Not Spam)**. 

The system uses Natural Language Processing (NLP) techniques to preprocess text data and a **Multinomial Naive Bayes** algorithm to make predictions with high accuracy. The web interface is built using **Flask** and deployed on **Render**.

## 🚀 Features
- **Real-time Prediction:** Instantly classifies user input as Spam or Safe.
- **Web Interface:** Clean and responsive UI built with HTML/CSS.
- **Machine Learning:** Trained on the SMS Spam Collection Dataset using Scikit-Learn.
- **Deployment:** Hosted live on the cloud using Render.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Frontend:** HTML, CSS
- **Backend:** Flask
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, NLTK
- **Deployment:** Gunicorn, Render Cloud

## 📂 Project Structure
├── app.py # Main Flask application ├── train.py # Script to train the ML model ├── requirements.txt # List of dependencies ├── templates │ └── index.html # Webpage HTML file ├── static # CSS and images (optional) ├── spam_model.pkl # Trained Model (Saved) └── vectorizer.pkl # TF-IDF Vectorizer (Saved)
## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Divyanshsharma1410/my-sms-app.git](https://github.com/Divyanshsharma1410/my-sms-app.git)
   cd my-sms-app
   Create a virtual environment:

Bash

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies:

Bash

pip install -r requirements.txt
Run the application:

Bash

python app.py
Open your browser and go to http://127.0.0.1:5000/

📊 Model Performance
The model was trained using the Multinomial Naive Bayes classifier, achieving an accuracy of approx 97% on the test dataset.

👤 Author
Divyansh Sharma - GitHub: Divyanshsharma1410

University: Shri Vishwakarma Skill University
