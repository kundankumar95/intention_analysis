import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    nltk.download('punkt')

    df['tokens'] = df['text'].apply(nltk.word_tokenize)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['intention'])

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le

if __name__ == "__main__":
    from data_collection import load_data
    df = load_data()
    X_train, X_test, y_train, y_test, le = preprocess_data(df)
    print(df.head())
