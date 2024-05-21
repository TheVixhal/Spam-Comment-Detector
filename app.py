from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load('spam.pkl')
vectorizer = joblib.load('vector.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the comment from the form
        comment = request.form['comment']
        
        # Debug: Print the received comment
        print(f"Received comment: {comment}")
        
        # Transform the comment using the loaded vectorizer
        comment_transformed = vectorizer.transform([comment])
        
        # Debug: Print the transformed comment
        print(f"Transformed comment: {comment_transformed.toarray()}")
        
        # Predict using the loaded model
        prediction = model.predict(comment_transformed)
        
        # Debug: Print the prediction
        print(f"Prediction: {prediction}")
        
        # Determine if it's spam or not
        result = 'SPAM' if prediction == 'Spam' else 'NOT-SPAM'
        
        return render_template('index.html', comment=comment, result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
