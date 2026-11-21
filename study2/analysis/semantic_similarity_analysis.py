import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Load model
# -------------------------
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def format_text(text):
    """Apply semantic similarity prompt formatting."""
    return f"Represent this sentence for semantic similarity: {text}"

# -------------------------
# Load data
# -------------------------
participant_df = pd.read_excel("Emotion recognition in human robot collaboration (Responses).xlsx", engine="openpyxl")
llm_df = pd.read_csv("log.csv")

participant_df.columns = participant_df.columns.str.strip()
llm_df.columns = llm_df.columns.str.strip()

# --- CORRECTED: Get the actual Participant ID column (Column E, index 4) ---
PARTICIPANT_ID_COL_INDEX = 4

# -------------------------
# Participant response columns by index (0-indexed)
participant_col_indices = [33, 34, 67, 68, 69, 103, 104, 105]

# -------------------------
# LLM rows to skip (0-indexed, counting from first data row)
skip_indices = [104, 220, 221]

# -------------------------
# Compute semantic similarity
similarity_results = []

for i, row in llm_df.iterrows():
    if i in skip_indices:
        print(f"Skipping LLM row {i} due to known error.")
        continue

    participant_index = i // 8
    
    if participant_index >= len(participant_df):
        print(f"Warning: LLM row {i} has no corresponding participant. Skipping.")
        continue

    # --- CORRECTED: Get the full, unique Participant ID string (e.g., '1A', '1B') ---
    participant_id_str = str(participant_df.iloc[participant_index, PARTICIPANT_ID_COL_INDEX]).strip()

    participant_texts = participant_df.iloc[participant_index, participant_col_indices].dropna().astype(str)
    if participant_texts.empty:
        print(f"Warning: Participant {participant_id_str} has no responses. Skipping.")
        continue
    combined_participant_text = " ".join(participant_texts.tolist())

    llm_text_raw = str(row.iloc[1])

    formatted_llm_text = format_text(llm_text_raw)
    formatted_participant_text = format_text(combined_participant_text)

    llm_embedding = model.encode(formatted_llm_text, normalize_embeddings=True)
    participant_embedding = model.encode(formatted_participant_text, normalize_embeddings=True)

    similarity = util.cos_sim(llm_embedding, participant_embedding).item()

    # --- CORRECTED: Save the full Participant ID, not an index ---
    similarity_results.append({
        "ParticipantID": participant_id_str,
        "cosine_similarity": similarity
    })

# -------------------------
# Save results to a new file to avoid confusion
similarity_df = pd.DataFrame(similarity_results)
similarity_df.to_csv("llm_participant_semantic_similarity_corrected.csv", index=False)
print(f"Done! Saved corrected similarity scores to llm_participant_semantic_similarity_corrected.csv")