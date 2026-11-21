import pandas as pd
import numpy as np
import sys

# --- Configuration ---
HRC_FILE = "HRC_Friedman_results.xlsx"
GODSPEED_FILE = "ancova_results.xlsx"
PREFERENCE_FILE = "User study notes and observations.xlsx"
OUTPUT_FILE = "preference_group_comparison.xlsx"

# Define all metrics
HRC_METRICS = ['Trust', 'Fluency', 'WorkingAlliance']
GODSPEED_METRICS = ['Anthropomorphism', 'Animacy', 'Likeability', 'Intelligence', 'Safety']
ALL_METRICS = HRC_METRICS + GODSPEED_METRICS

# --- 1. Load All Necessary Data ---
print("--- Loading All Data Files ---")
try:
    hrc_df = pd.read_excel(HRC_FILE, sheet_name="LongData")
    godspeed_df = pd.read_excel(GODSPEED_FILE, sheet_name="LongData")
    
    # Load Preference Data
    obs_df = pd.read_excel(PREFERENCE_FILE, sheet_name="Study observations", engine="openpyxl")
    participant_ids_raw = obs_df.iloc[:, 0].astype(str).str.strip()
    preferences_raw = obs_df.iloc[:, 6].astype(str).str.strip().str.upper()

    pref_df = pd.DataFrame({'ParticipantID': participant_ids_raw, 'Preference': preferences_raw})
    pref_df['Preference'] = pref_df['Preference'].replace({'NE': 'NEA'})
    pref_df = pref_df[pref_df['Preference'].isin(['EA', 'NEA'])]
    pref_df['ParticipantID'] = pref_df['ParticipantID'].str.replace(r'\.0$', '', regex=True)
    
    print("✅ All data files loaded successfully.")

except FileNotFoundError as e:
    print(f"\n❌ ERROR: File not found: {e.filename}")
    sys.exit()
except Exception as e:
    print(f"\n❌ An unexpected error occurred during file loading: {e}")
    sys.exit()

# --- 2. Create a Unified DataFrame ---
print("\n--- Merging data sources ---")
hrc_df['Condition'] = hrc_df['Condition'].replace({'NE': 'NEA'})
godspeed_df['Condition'] = godspeed_df['Condition'].replace({'NE': 'NEA'})
metrics_df = pd.merge(hrc_df, godspeed_df, on=['ParticipantID', 'Condition'], how='outer')
final_df = pd.merge(metrics_df, pref_df, on='ParticipantID', how='inner')
print("✅ Data successfully merged.")


# --- 3. Calculate Grouped Statistics (as before) ---
print("\n--- Calculating statistics by preference group ---")
grouped_stats = final_df.groupby(['Preference', 'Condition'])[ALL_METRICS].agg(['mean', 'std', 'count'])


# --- 4. Reshape Data for Easy Comparison ---
print("\n--- Reshaping data for easy eyeballing ---")
results_list = []

for metric in ALL_METRICS:
    for condition in ['EA', 'NEA']:
        try:
            # Stats for the group that PREFERRED EA
            ea_pref_mean = grouped_stats.loc[('EA', condition), (metric, 'mean')]
            ea_pref_std = grouped_stats.loc[('EA', condition), (metric, 'std')]
            ea_pref_n = grouped_stats.loc[('EA', condition), (metric, 'count')]
            
            # Stats for the group that PREFERRED NEA
            nea_pref_mean = grouped_stats.loc[('NEA', condition), (metric, 'mean')]
            nea_pref_std = grouped_stats.loc[('NEA', condition), (metric, 'std')]
            nea_pref_n = grouped_stats.loc[('NEA', condition), (metric, 'count')]
            
            # Calculate the key difference
            mean_diff = ea_pref_mean - nea_pref_mean
            
            results_list.append({
                'Subscale': metric,
                'Rated_Condition': condition,
                'Prefers_EA_Mean': ea_pref_mean,
                'Prefers_NEA_Mean': nea_pref_mean,
                'Mean_Diff (PrefEA - PrefNEA)': mean_diff,
                'Prefers_EA_Std': ea_pref_std,
                'Prefers_NEA_Std': nea_pref_std,
                'N_Prefers_EA': ea_pref_n,
                'N_Prefers_NEA': nea_pref_n,
            })
        except KeyError:
            print(f"⚠️ Warning: Could not find data for {metric} under {condition}. Skipping.")
            continue

# Create the final, flat DataFrame
comparison_df = pd.DataFrame(results_list)

# Round for clarity
numeric_cols = [col for col in comparison_df.columns if 'Mean' in col or 'Std' in col]
comparison_df[numeric_cols] = comparison_df[numeric_cols].round(2)


# --- 5. Display and Save the Final Table ---
print("\n" + "="*110)
print("COMPARISON OF PREFERENCE GROUPS (EA-Preferring Group vs. NEA-Preferring Group)")
print("="*110)

# Define a clean column order for printing
display_cols = [
    'Subscale', 'Rated_Condition', 'Prefers_EA_Mean', 'Prefers_NEA_Mean', 
    'Mean_Diff (PrefEA - PrefNEA)', 'N_Prefers_EA', 'N_Prefers_NEA'
]
print(comparison_df[display_cols].to_string(index=False))
print("="*110)
print("\n'Rated_Condition' is the agent they were rating.")
print("'Prefers_EA_Mean' is the average score from the 31 people who preferred the EA agent.")
print("'Prefers_NEA_Mean' is the average score from the 9 people who preferred the NEA agent.")
print("A positive 'Mean_Diff' means the EA-preferring group rated that item higher.")


print(f"\n--- Saving full results to Excel ---")
try:
    comparison_df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    print(f"✅ Successfully saved detailed comparison to '{OUTPUT_FILE}'")
except Exception as e:
    print(f"❌ Error saving Excel file: {e}")