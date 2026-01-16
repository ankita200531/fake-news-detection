from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

#  Load all models and vectorizer
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
models = {
    "Logistic Regression": joblib.load('models/logistic.pkl'),
    "Naive Bayes": joblib.load('models/naive_bayes.pkl'),
    "Random Forest": joblib.load('models/random_forest.pkl'),
    "SVM": joblib.load('models/svm.pkl')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']

    #  Transform input text
    transformed = vectorizer.transform([news])

    #  Store results of all models
    predictions = {}
    for name, model in models.items():
        pred = model.predict(transformed)[0]
        predictions[name] = " Real News" if pred == 1 else " Fake News"

    #  Calculate majority voting (optional)
    votes = [1 if "Real" in val else 0 for val in predictions.values()]
    final_result = " Real News" if np.mean(votes) >= 0.5 else " Fake News"

    return render_template('result.html', predictions=predictions, final_result=final_result)

if __name__ == '__main__':
    app.run(debug=True)