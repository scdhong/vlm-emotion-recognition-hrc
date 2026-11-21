import pandas as pd
import pymc as pm
import arviz as az
import numpy as np

# Helper function to summarize posterior samples
def summarize_posterior(samples):
    """Computes mean, 95% HDI, and P(>0) for a posterior distribution."""
    mean = np.mean(samples)
    hdi = az.hdi(samples, hdi_prob=0.95)
    prob_gt_zero = np.mean(samples > 0)
    return {
        'Mean': mean,
        '95% HDI Lower': hdi[0],
        '95% HDI Upper': hdi[1],
        'P(>0)': prob_gt_zero
    }

def run_bayesian_hrc(infile, outfile, draws=2000, tune=1000, chains=4):
    df = pd.read_excel(infile, sheet_name=0, engine="openpyxl")

    def col_letter_to_index(letter):
        letter = letter.upper().strip()
        idx = 0
        for ch in letter:
            idx = idx * 26 + (ord(ch) - ord('A') + 1)
        return idx - 1

    c_participant = col_letter_to_index('E')
    participant_ids = df.iloc[:, c_participant].astype(str).str.strip()

    control_cols = df.iloc[:, 60:68].apply(pd.to_numeric, errors='coerce')
    part2_cols = df.iloc[:, 96:103].apply(pd.to_numeric, errors='coerce')
    part3_cols = df.iloc[:, 130:137].apply(pd.to_numeric, errors='coerce')

    subscores = {
        "Fluency": slice(0, 2),
        "Trust": slice(2, 4),
        "WorkingAlliance": slice(4, 7)
    }

    # Create lists to hold results for each sheet
    mean_results = []
    contrast_results = []

    for subscale, srange in subscores.items():
        print(f"Running Bayesian mixed model for {subscale} ...")

        control_mean = control_cols.iloc[:, srange].mean(axis=1)
        part2_mean = part2_cols.iloc[:, srange].mean(axis=1)
        part3_mean = part3_cols.iloc[:, srange].mean(axis=1)

        records = []
        for i, pid in enumerate(participant_ids):
            if pd.isna(control_mean.iloc[i]) and pd.isna(part2_mean.iloc[i]) and pd.isna(part3_mean.iloc[i]):
                continue
            order = 'A' if pid.endswith('A') else 'B'
            part2_cond, part3_cond = ('NE', 'EA') if order == 'A' else ('EA', 'NE')
            records.append({'ParticipantID': i, 'Condition': 'Control', 'Score': control_mean.iloc[i]})
            records.append({'ParticipantID': i, 'Condition': part2_cond, 'Score': part2_mean.iloc[i]})
            records.append({'ParticipantID': i, 'Condition': part3_cond, 'Score': part3_mean.iloc[i]})

        long_df = pd.DataFrame(records).dropna().reset_index()

        participants = pd.Categorical(long_df["ParticipantID"]).codes
        n_subj = len(np.unique(participants))
        condition_cats = pd.Categorical(long_df["Condition"], categories=["Control", "NE", "EA"])
        cond_NE = (condition_cats == 'NE').astype(int)
        cond_EA = (condition_cats == 'EA').astype(int)

        scores_raw = long_df["Score"].values
        scores_mean = scores_raw.mean()
        scores_sd = scores_raw.std(ddof=1) if scores_raw.std(ddof=1) > 0 else 1.0
        scores = (scores_raw - scores_mean) / scores_sd

        with pm.Model() as model:
            mu_a = pm.Normal("mu_a", mu=0.0, sigma=1.0)
            sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
            a_subj = pm.Normal("a_subj", mu=mu_a, sigma=sigma_a, shape=n_subj)
            beta_NE = pm.Normal("beta_NE", mu=0.0, sigma=1.0)
            beta_EA = pm.Normal("beta_EA", mu=0.0, sigma=1.0)
            nu = pm.Exponential("nu_minus_one", 1/29.) + 1
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            mu = a_subj[participants] + beta_NE * cond_NE + beta_EA * cond_EA
            y_obs = pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=scores)
            trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.98,
                              return_inferencedata=True, progressbar=True)

        post = trace.posterior
        mu_a_samps = post['mu_a'].values.ravel()
        b_NE_samps = post['beta_NE'].values.ravel() # Standardized diff: NE - Control
        b_EA_samps = post['beta_EA'].values.ravel() # Standardized diff: EA - Control

        # --- PART 1: Calculate Raw Means (for the first sheet) ---
        est_mean_std_control = mu_a_samps
        est_mean_std_ne      = mu_a_samps + b_NE_samps
        est_mean_std_ea      = mu_a_samps + b_EA_samps
        
        est_mean_orig_control = (est_mean_std_control * scores_sd) + scores_mean
        est_mean_orig_ne      = (est_mean_std_ne * scores_sd) + scores_mean
        est_mean_orig_ea      = (est_mean_std_ea * scores_sd) + scores_mean

        for cond_name, samples in { "Control": est_mean_orig_control, "NE": est_mean_orig_ne, "EA": est_mean_orig_ea }.items():
            summary = summarize_posterior(samples)
            mean_results.append({'Subscale': subscale, 'Condition': cond_name, **summary})

        # --- PART 2: Calculate Pairwise Contrasts (for the second sheet) ---
        # First, calculate the third contrast (EA - NE) on the standardized scale
        diff_std_ea_ne = b_EA_samps - b_NE_samps

        # Now, un-standardize all differences to the original score scale
        # Formula: raw_diff = std_diff * scores_sd
        diff_orig_ne_control = b_NE_samps * scores_sd
        diff_orig_ea_control = b_EA_samps * scores_sd
        diff_orig_ea_ne      = diff_std_ea_ne * scores_sd
        
        for name, samples in { "NE - Control": diff_orig_ne_control, "EA - Control": diff_orig_ea_control, "EA - NE": diff_orig_ea_ne }.items():
            summary = summarize_posterior(samples)
            contrast_results.append({'Subscale': subscale, 'Contrast': name, **summary})

    # Save both results to a single Excel file with two sheets
    with pd.ExcelWriter(outfile, engine='openpyxl') as writer:
        pd.DataFrame(mean_results).to_excel(writer, sheet_name="Estimated_Means", index=False)
        pd.DataFrame(contrast_results).to_excel(writer, sheet_name="Pairwise_Contrasts", index=False)

    print(f"✅ Combined results with means and contrasts saved to {outfile}")

if __name__ == "__main__":
    run_bayesian_hrc(
        infile="Emotion recognition in human robot collaboration (Responses).xlsx",
        outfile="HRC_Bayesian_Results_Combined.xlsx",
        draws=2000,
        tune=1000,
        chains=4
    )