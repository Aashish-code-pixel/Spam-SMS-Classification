import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv("data/spam_sms.csv")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r"[^\w\s]", "", text) 
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Apply text cleaning
df["cleaned_text"] = df["message"].apply(clean_text)

# Save cleaned data
df.to_csv("data/cleaned_data.csv", index=False)
print("Text preprocessing complete. Saved as 'cleaned_data.csv'")
