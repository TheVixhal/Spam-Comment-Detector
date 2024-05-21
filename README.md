# Spam-Comment-Detector
## Use:
```
git clone https://github.com/TheVixhal/Spam-Comment-Detector.git
```

```
cd Spam-Comment-Detector
```

```
python app.py
```
~ Ctrl+Click on `Running on http://xxx.x.x.1:5000`
# Model Training Code

# Import Libraries & Data 
```
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('/kaggle/input/spam-set/Comments - Sheet1 (1).csv', usecols= ['Comments', 'Label'])
```

# Re-Labelling
```
data['Label'] = data['Label'].map({
    0 : "Not Spam",
    1 : "Spam"
})
```

```
X = np.array(data['Comments'])
y = np.array(data['Label'])
```

# CountVectorizer
```
CV = CountVectorizer()
X = CV.fit_transform(X)
```

# Training Parameters
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

X_train.shape, y_train.shape
```

# MultinomialNB
```
MNB = MultinomialNB()
MNB.fit(X_train, y_train)
y_pred = MNB.predict(X_test)
print("MultinomialNB Results")
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
```

# Test
```
test = "nice content"
test_data = CV.transform([test]).toarray()
MNB.predict(test_data)[0]
```

# SAVE MODEL & VECTORIZER
```
joblib.dump(CV, 'vector.pkl') #save-vectorizer
joblib.dump(MNB, 'spam.pkl') #save-model
```
