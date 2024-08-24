def predict_intention(model, vectorizer, le, text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    intention = le.inverse_transform(prediction)
    return intention[0]

if __name__ == "__main__":
    from model_development import train_model, evaluate_model
    from feature_extraction import extract_features
    from data_preprocessing import preprocess_data
    from data_collection import load_data
    
    # Load and preprocess the data
    df = load_data()
    X_train, X_test, y_train, y_test, le = preprocess_data(df)
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    
    # Train the model
    model = train_model(X_train_tfidf, y_train)
    
    # Loop to continuously get user input and predict intention
    while True:
        # Get user input
        user_input = input("Enter a sentence to predict its intention (or type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        
        # Predict the intention
        predicted_intention = predict_intention(model, vectorizer, le, user_input)
        print(f"Predicted Intention: {predicted_intention}")

