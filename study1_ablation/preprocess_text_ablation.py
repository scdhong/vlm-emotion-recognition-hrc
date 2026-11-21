import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK data once
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Set up preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    tokens = nltk.word_tokenize(text.lower())
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]
    return " ".join(cleaned)

def process_csv_and_wordcloud(input_csv, columns_to_process, processed_csv='ablation-gemini_flash_processed_cells.csv', wordcloud_image='gemini_flash_wordcloud.png'):
    df = pd.read_csv(input_csv)

    # Fill NaNs and ensure string type
    df[columns_to_process] = df[columns_to_process].fillna('').astype(str)

    # Process each cell individually and store back
    all_words = []
    for col in columns_to_process:
        df[col + "_processed"] = df[col].apply(preprocess_text)
        all_words.extend(df[col + "_processed"].str.split().sum())  # flatten list of words

    # Save processed cells to a new CSV
    df.to_csv(processed_csv, index=False)
    print(f"Processed CSV saved as '{processed_csv}'.")

    # Generate word cloud from all processed words
    final_text = " ".join(all_words)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(final_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(wordcloud_image)
    print(f"Word cloud saved as '{wordcloud_image}'.")

# === Example usage ===
if __name__ == "__main__":
    columns = ['Emotion Description'] #['P10_0004','S2P19_0000','P1_0000','P11_0003','S2P18_0005','S2P9_0004','P12_0002','S2P17_0005','P2_0023','P13_0011','P9_0003','S2P8_0040','P14_0012','P8_0001','P3_0019','P15_0000','P7_0000','S2P4_0021','P4_0001','S2P3_0021','P16_0018','P5_0001','S2P20_0011','P6_0001','S2P1_0020','S2P2_0034','P17_0022','S2P11_0010','S2P16_0007','P20_0006','S2P12_0032','S2P15_0014','P18_0006','S2P13_0027','S2P14_0014','P19_0007']  # Replace with your actual column names
    process_csv_and_wordcloud('ablation_gemini_flash_analysis_results.csv', columns)
