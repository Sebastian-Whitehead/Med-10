import os
import json
import pandas as pd
from utils.utilities import get_root_dir
from scipy.stats import ttest_ind_from_stats

def load_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def make_result_file(folder):
    df = pd.DataFrame()
    file_folder = os.path.join(get_root_dir(), folder)
    for file in os.listdir(file_folder):
        file_path = os.path.join(file_folder, file)
        if not file.endswith('.json'):
            continue
        data = load_file(file_path)
        temp_df = pd.DataFrame(data)
        df = pd.concat([df, temp_df], ignore_index=True)
    df.drop(['batch_path', 'model', 'time_per_image', 'time_total'], axis=1, inplace=True, errors='ignore')
    df.to_csv(os.path.join(file_folder, 'result.csv'), index=False)

def get_p(row, reference_row, res, std):
    n = 653
    ref_map = reference_row[res]
    row_map = row[res]
    ref_std = reference_row[std]
    row_std = row[std]
    t_stat, p_value = ttest_ind_from_stats(
        mean1=ref_map,
        std1=ref_std,
        nobs1=n,
        mean2=row_map,
        std2=row_std,
        nobs2=n
    )
    return t_stat, p_value

def calc_p(folder, results_file):
    df = pd.read_csv(results_file)
    reference_row = df[df["test"] == "0"]
    if reference_row.empty:
        raise ValueError("No row found where test == 0")
    reference_row = reference_row.iloc[0]
    df["p_stat_map50"] = None
    df["p_stat_recall"] = None
    df["p_stat_precision"] = None
    for index, row in df.iterrows():
        res, std = 'map50', 'std_dev_50'
        _, p_stat_50 = get_p(row, reference_row, res, std)
        df.at[index, "p_stat_map50"] = p_stat_50
        res, std = 'recall', 'std_recall'
        _, p_stat_r = get_p(row, reference_row, res, std)
        df.at[index, "p_stat_recall"] = p_stat_r
        res, std = 'precision', 'std_precision'
        _, p_stat_p = get_p(row, reference_row, res, std)
        df.at[index, "p_stat_precision"] = p_stat_p
    df = df.sort_values(by="test", ascending=True)
    df.to_csv(results_file, index=False)
    print(f"Updated results saved to {results_file}")

if __name__ == "__main__":
    folder = os.path.join(get_root_dir(), 'fine_tune_sets', 'results')
    print(f"Processing folder: {folder}")
    make_result_file(folder)
    print(f"Result file created in {folder} folder.")
    results_file = os.path.join(folder, 'result.csv')
    calc_p(folder, results_file)
