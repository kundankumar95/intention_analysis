from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(X_train, X_test):
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer

if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    from data_collection import load_data
    
    df = load_data()
    X_train, X_test, y_train, y_test, le = preprocess_data(df)
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    print(X_train_tfidf.shape)
