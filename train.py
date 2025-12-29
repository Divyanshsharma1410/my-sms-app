import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the dataset
# Note: encoding='latin-1' is often required for this dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# 2. Clean the data (Keep only necessary columns)
df = df[['v1', 'v2']]
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# 3. Convert labels to numbers (Spam=1, Ham=0)
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], 
    df['label_num'], 
    test_size=0.2, 
    random_state=42
)

# 5. Feature Extraction (Convert text to numbers using TF-IDF)
# TF-IDF highlights words that are unique to spam messages
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 6. Train the Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 7. Check Accuracy
predictions = model.predict(X_test_tfidf)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 8. Save the Model and Vectorizer for the App
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(tfidf, vec_file)

print("Success! Model and Vectorizer saved.")