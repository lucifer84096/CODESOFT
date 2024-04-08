import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC()
}

for name, classifier in classifiers.items():
    classifier.fit(X_train_tfidf, y_train)
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

best_model = LogisticRegression()
best_model.fit(X_train_tfidf, y_train)

def classify_sms(message):
    message_tfidf = tfidf_vectorizer.transform([message])
    prediction = best_model.predict(message_tfidf)
    return "spam" if prediction[0] == 1 else "legitimate"

spam_count = data['label'].value_counts()[1]
legitimate_count = data['label'].value_counts()[0]

plt.figure(figsize=(8, 6))
plt.bar(['Spam', 'Legitimate'], [spam_count, legitimate_count], color=['red', 'green'])
plt.title('Distribution of Spam vs Legitimate Messages')
plt.xlabel('Message Type')
plt.ylabel('Count')
plt.show()