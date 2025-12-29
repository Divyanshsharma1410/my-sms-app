from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# 1. Load the trained model and vectorizer
# Make sure these files exist in the same folder as app.py!
try:
    model = pickle.load(open('spam_model.pkl', 'rb'))
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model files not found. Please run train.py first!")
    exit()

@app.route('/')
def home():
    # This serves the index.html file from the 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        # Convert text to numbers using the vectorizer
        vect = tfidf.transform(data).toarray()
        
        # Make prediction
        my_prediction = model.predict(vect)
        
        # Determine result
        if my_prediction == 1:
            res = "🚨 SPAM DETECTED"
            css_class = "spam"
        else:
            res = "✅ NOT SPAM (SAFE)"
            css_class = "ham"
            
        return render_template('index.html', prediction_text=res, prediction_class=css_class)

if __name__ == '__main__':
    app.run(debug=True)