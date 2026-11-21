import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the BGE large model from Hugging Face
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Apply prompt formatting for better embeddings
def format_text(text):
    return f"Represent this sentence for semantic similarity: {text}"

# Load processed LLM and ground truth data
ground_truth_df = pd.read_csv("Prolific_processed_cells.csv")
#cnn_df = pd.read_csv("cnn_processed_cells.csv")
llm_df = pd.read_csv("ablation-gemini_flash_processed_cells.csv")

similarity_results = []


for _, row in llm_df.iterrows(): #cnn_df.iterrows(): 
    video = row['Video Filename'].replace('.mp4', '')
    llm_text_raw = row['objects']
    #cnn_text_raw = row['objects']

    gt_column = f"{video}_processed"
    if gt_column not in ground_truth_df.columns:
        print(f"Column {gt_column} not found in ground truth. Skipping.")
        continue

    # Drop missing GT values
    gt_texts = ground_truth_df[gt_column].dropna().astype(str)
    if gt_texts.empty:
        print(f"No ground truth responses for {video}. Skipping.")
        continue

    # Format input text with semantic prompt
    formatted_llm_text = format_text(llm_text_raw)
    #formatted_cnn_text = format_text(cnn_text_raw)
    formatted_gt_texts = [format_text(txt) for txt in gt_texts.tolist()]

    # Encode with normalization for cosine similarity
    llm_embedding = model.encode(formatted_llm_text, normalize_embeddings=True)
    #cnn_embedding = model.encode(formatted_cnn_text, normalize_embeddings=True)
    gt_embeddings = model.encode(formatted_gt_texts, normalize_embeddings=True)

    # Average GT embeddings to get a single representation
    gt_avg_embedding = sum(gt_embeddings) / len(gt_embeddings)

    # Compute cosine similarity
    similarity = util.cos_sim(llm_embedding, gt_avg_embedding).item()
    #similarity = util.cos_sim(cnn_embedding, gt_avg_embedding).item()

    similarity_results.append({
        "video_name": video,
        "cosine_similarity": similarity
    })
    '''print(f"\n[Video: {video}]")
    print(f"LLM Description: {llm_text_raw}")
    print("GT Responses:", gt_texts.tolist())
    print("Formatted GT Examples:", formatted_gt_texts[:2])
    print("Cosine Similarity:", similarity)'''


# Save results
similarity_df = pd.DataFrame(similarity_results)
similarity_df.to_csv("objects-ablation-gemini_flash_similarity_results.csv", index=False)
'''import matplotlib.pyplot as plt

plt.hist(similarity_df['cosine_similarity'], bins=20)
plt.title("Cosine Similarity Distribution")
plt.xlabel("Similarity")
plt.ylabel("Frequency")
plt.show()
print("Model device:", model.device)
print("Test embedding shape:", model.encode("Hello world", normalize_embeddings=True).shape)

test1 = "She looks surprised and happy"
test2 = "She seems joyful and a bit shocked"

emb1 = model.encode(format_text(test1), normalize_embeddings=True)
emb2 = model.encode(format_text(test2), normalize_embeddings=True)
print("Manual similarity:", util.cos_sim(emb1, emb2).item())
print("Number of videos in result:", len(similarity_results))
print("Expected number (from LLM input):", len(llm_df))'''