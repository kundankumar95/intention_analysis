from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train_tfidf, y_train):
    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return model

def evaluate_model(model, X_test_tfidf, y_test, le):
    # Predict on the test data
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    from feature_extraction import extract_features
    from data_preprocessing import preprocess_data
    from data_collection import load_data
    
    df = load_data()
    X_train, X_test, y_train, y_test, le = preprocess_data(df)
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    
    model = train_model(X_train_tfidf, y_train)
    evaluate_model(model, X_test_tfidf, y_test, le)
