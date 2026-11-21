import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Simple formatting function
def format_text(text):
    return str(text)

# Load CSVs
ground_truth_df = pd.read_csv("Prolific_processed_cells.csv")
cnn_df = pd.read_csv("cnn_processed_cells.csv")
llm_df = pd.read_csv("gemini_flash_processed_cells.csv")

results = []

for _, row in cnn_df.iterrows():
    video = row['video_filename'].replace('.mp4', '')
    cnn_text_raw = row['total_processed']

    # Find corresponding LLM row
    llm_row = llm_df[llm_df['video_filename'] == row['video_filename']]
    if llm_row.empty:
        print(f"No LLM entry for {video}. Skipping.")
        continue
    llm_text_raw = llm_row.iloc[0]['Emotion Description_processed']

    # Build GT column name
    gt_column = f"{video}_processed"
    if gt_column not in ground_truth_df.columns:
        print(f"Column {gt_column} not found in ground truth. Skipping.")
        continue

    # Drop missing GT responses
    gt_texts = ground_truth_df[gt_column].dropna().astype(str)
    if gt_texts.empty:
        print(f"No ground truth responses for {video}. Skipping.")
        continue

    # Format texts
    formatted_gt_texts = [format_text(txt) for txt in gt_texts.tolist()]
    formatted_cnn_text = format_text(cnn_text_raw)
    formatted_llm_text = format_text(llm_text_raw)

    # Compute sentiment scores
    gt_sentiments = [sia.polarity_scores(txt)['compound'] for txt in formatted_gt_texts]
    gt_avg_sentiment = sum(gt_sentiments) / len(gt_sentiments)

    cnn_sentiment = sia.polarity_scores(formatted_cnn_text)['compound']
    llm_sentiment = sia.polarity_scores(formatted_llm_text)['compound']

    results.append({
        "video_name": video,
        "GT_avg_sentiment": gt_avg_sentiment,
        "CNN_sentiment": cnn_sentiment,
        "LLM_sentiment": llm_sentiment
    })

# Save results
sentiment_df = pd.DataFrame(results)
sentiment_df.to_csv("gemini_flash_sentiment_analysis_results.csv", index=False)

print(f"Sentiment scores saved to gemini_flash_sentiment_analysis_results.csv")
print(sentiment_df.head())
