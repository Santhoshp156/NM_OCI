import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to clean text data
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text

def run_fake_news_detector(csv_path):
    try:
        df = pd.read_csv(csv_path)

        # Validate that the necessary columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            print("CSV must contain 'text' and 'label' columns.")
            return

        # Clean the text data
        df['text'] = df['text'].apply(clean_text)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

        # Initialize the TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        # Initialize the PassiveAggressiveClassifier
        model = PassiveAggressiveClassifier(max_iter=50)
        model.fit(X_train_vec, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {acc * 100:.2f}%")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # Prediction loop for user input
        while True:
            text = input("\nEnter a news headline (or 'exit' to quit): ")
            if text.lower() == 'exit':
                break
            text_cleaned = clean_text(text)  # Clean the input text
            pred = model.predict(tfidf.transform([text_cleaned]))
            print("Prediction:", pred[0])

    except FileNotFoundError:
        print("File not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Start
if __name__ == "__main__":
    path = input("Enter path to your dataset CSV file: ")
    run_fake_news_detector(path)
