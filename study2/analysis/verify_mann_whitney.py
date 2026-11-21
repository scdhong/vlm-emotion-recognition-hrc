import pandas as pd
from scipy import stats
import sys

def run_mann_whitney_test():
    """
    Loads participant similarity and preference data, performs a Mann-Whitney U test
    to compare cosine similarity between preference groups, and prints the results.
    """
    print("--- Verifying Mann-Whitney U Test for Preference Groups ---")

    # --- 1. Load and Prepare Data ---
    try:
        # Load corrected similarity scores
        sim_df = pd.read_csv("llm_participant_semantic_similarity_corrected.csv")
        
        # Load preference data from the correct file and sheet
        obs_df = pd.read_excel("User study notes and observations.xlsx", sheet_name="Study observations", engine="openpyxl")

    except FileNotFoundError as e:
        print(f"\nERROR: A required data file was not found: {e.filename}")
        print("Please ensure all data files are in the same folder as the script.")
        sys.exit()

    # --- Process Similarity Data ---
    participant_similarity = sim_df.groupby('ParticipantID')['cosine_similarity'].mean().reset_index()

    # --- Process Preference Data ---
    PREFERENCE_COL_INDEX = 6  # Column G
    PARTICIPANT_ID_COL_INDEX = 0 # Column A

    participant_ids_raw = obs_df.iloc[:, PARTICIPANT_ID_COL_INDEX].astype(str).str.strip()
    preferences_raw = obs_df.iloc[:, PREFERENCE_COL_INDEX].astype(str).str.strip().str.upper()

    pref_df = pd.DataFrame({'ParticipantID': participant_ids_raw, 'Preference': preferences_raw})
    pref_df['Preference'] = pref_df['Preference'].replace({'NE': 'NEA'})
    pref_df = pref_df[pref_df['Preference'].isin(['EA', 'NEA'])]
    pref_df['ParticipantID'] = pref_df['ParticipantID'].str.replace(r'\.0$', '', regex=True)

    # --- Merge Data ---
    merged_df = pd.merge(participant_similarity, pref_df, on='ParticipantID', how='inner')

    if len(merged_df) < 40:
        print(f"\nWARNING: Merged DataFrame contains only {len(merged_df)} participants. Check for ID mismatches.")
    
    # --- 2. Separate Data into Two Groups ---
    group_ea = merged_df[merged_df['Preference'] == 'EA']['cosine_similarity']
    group_nea = merged_df[merged_df['Preference'] == 'NEA']['cosine_similarity']

    if group_ea.empty or group_nea.empty:
        print("\nERROR: One or both preference groups are empty. Cannot perform the test.")
        print(f"EA group size: {len(group_ea)}, NEA group size: {len(group_nea)}")
        sys.exit()
        
    # --- 3. Perform the Mann-Whitney U Test ---
    # The 'alternative'='two-sided' tests if the distributions are different (non-directional)
    stat, p_value = stats.mannwhitneyu(group_ea, group_nea, alternative='two-sided')

    # --- 4. Print a Detailed Report ---
    print("\n--- Descriptive Statistics ---")
    print(f"EA Preference Group (N={len(group_ea)}):")
    print(f"  Mean Cosine Similarity: {group_ea.mean():.4f}")
    print(f"  Standard Deviation:     {group_ea.std():.4f}")
    print(f"  Median Similarity:      {group_ea.median():.4f}")

    print(f"\nNEA Preference Group (N={len(group_nea)}):")
    print(f"  Mean Cosine Similarity: {group_nea.mean():.4f}")
    print(f"  Standard Deviation:     {group_nea.std():.4f}")
    print(f"  Median Similarity:      {group_nea.median():.4f}")

    print("\n--- Mann-Whitney U Test Results ---")
    print(f"U-statistic: {stat:.2f}")
    print(f"P-value:     {p_value:.4f}")

    # --- 5. Interpret the Result ---
    alpha = 0.05
    if p_value < alpha:
        print(f"\nConclusion: The result is statistically significant (p < {alpha}).")
        print("There is a significant difference in the cosine similarity scores between")
        print("participants who preferred the EA condition and those who preferred the NEA condition.")
    else:
        print(f"\nConclusion: The result is not statistically significant (p >= {alpha}).")
        print("There is not enough evidence to claim a difference in cosine similarity")
        print("scores between the two preference groups.")

if __name__ == "__main__":
    run_mann_whitney_test()