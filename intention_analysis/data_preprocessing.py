import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Downloading NLTK data
    nltk.download('punkt')

    # Tokenization
    df['tokens'] = df['text'].apply(nltk.word_tokenize)

    # Encode the labels (intentions)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['intention'])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le

if __name__ == "__main__":
    from data_collection import load_data
    df = load_data()
    X_train, X_test, y_train, y_test, le = preprocess_data(df)
    print(df.head())
