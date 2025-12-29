from flask import Flask, render_template, request
import pickle
import numpy as np

# This is the "app" variable that Render is looking for!
app = Flask(__name__)

# Load the trained model and vectorizer
try:
    model = pickle.load(open('spam_model.pkl', 'rb'))
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model files not found. Please run train.py first!")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        # Transform input and predict
        vect = tfidf.transform(data).toarray()
        my_prediction = model.predict(vect)
        
        # Return the result
        if my_prediction[0] == 1:
            res = "🚨 SPAM DETECTED"
            css_class = "spam"
        else:
            res = "✅ NOT SPAM (SAFE)"
            css_class = "ham"
            
        return render_template('index.html', prediction_text=res, prediction_class=css_class)

if __name__ == '__main__':
    app.run(debug=True)