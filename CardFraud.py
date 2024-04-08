# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the datasets
data_train = pd.read_csv('fraudTrain.csv')
data_test = pd.read_csv('fraudTest.csv')


# Data exploration for training data
print(data_train.head())
print(data_train.info())

# Check for missing values
print("Missing values:", data_train.isnull().sum().max())

# Check class distribution
print("Class Distribution:\n", data_train['is_fraud'].value_counts())

# Separate features and target for training data
X_train = data_train.drop('is_fraud', axis=1)
y_train = data_train['is_fraud']

# Separate features and target for testing data
X_test = data_test.drop('is_fraud', axis=1)
y_test = data_test['is_fraud']

# Exclude non-numeric columns
non_numeric_columns = ['trans_date_trans_time', 'merchant', 'category',
                       'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']

X_train_numeric = X_train.drop(non_numeric_columns, axis=1)
X_test_numeric = X_test.drop(non_numeric_columns, axis=1)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

# One-hot encode categorical columns
encoder = OneHotEncoder()
X_train_categorical = encoder.fit_transform(X_train[['merchant', 'category', 'gender', 'state']])
X_test_categorical = encoder.transform(X_test[['merchant', 'category', 'gender', 'state']])

# Concatenate scaled numeric and one-hot encoded categorical features
X_train_final = pd.concat([pd.DataFrame(X_train_scaled), pd.DataFrame(X_train_categorical.toarray())], axis=1)
X_test_final = pd.concat([pd.DataFrame(X_test_scaled), pd.DataFrame(X_test_categorical.toarray())], axis=1)

# Model Training and Evaluation
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    print("\nTraining", name)
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)
    
    # Evaluation
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
