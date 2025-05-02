# Read json file from folder 'result_cls_bad'
# put data into dataframe under right names
# save dataframe to csv file

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
            continue  # Skip non-JSON files
        data = load_file(file_path)
        temp_df = pd.DataFrame(data)  # Convert the loaded data to a DataFrame
        df = pd.concat([df, temp_df], ignore_index=True)  # Append the new DataFrame

    df.drop(['batch_path', 'model', 'time_per_image', 'time_total'], axis=1, inplace=True, errors='ignore')
    df.to_csv(os.path.join(file_folder, 'result.csv'), index=False)

def get_p(row, reference_row, res, std):
    columns_to_compare = [res, std]
    n = 653
    # Iterate and compare
    for col in columns_to_compare:
        if col == res:
            ref_map = reference_row[col]
            row_map = row[col]
        if col == std:
            ref_std = reference_row[col]
            row_std = row[col]


    t_stat, p_value = ttest_ind_from_stats(
        mean1=ref_map,
        std1=ref_std,
        nobs1=n,
        mean2=row_map,
        std2=row_std,
        nobs2=n
    )
    '''
    print(f"Comparing {col}:")
    print(f'map50: {ref_map} vs {row_map}')
    print(f'std_dev_50: {ref_std} vs {row_std}')
    print(f"t-statistic: {t_stat}, p-value: {p_value}")
    print()
    '''
    return t_stat, p_value


def calc_p(folder, results_file):
    print(folder)
    if 'rev' in folder:
        print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    # Load the CSV file
    df = pd.read_csv(results_file)  # Replace with your actual file path

    reference_row = df[df["test"] == "test_set-og"]
    if reference_row.empty:
        raise ValueError("No row found where test == 2")

    # Extract the first (and assuming only) reference row
    reference_row = reference_row.iloc[0]

    # Add new columns for p-values
    df["p_stat_map50"] = None
    df["p_stat_recall"] = None
    df["p_stat_precision"] = None

    # Iterate and compare
    for index, row in df.iterrows():
        print(f"Comparing test == {row['test']}:")
        
        # Calculate p-value for map50
        res = 'map50'
        std = 'std_dev_50'
        _, p_stat_50 = get_p(row, reference_row, res, std)
        df.at[index, "p_stat_map50"] = p_stat_50

        # Calculate p-value for recall
        res = 'recall'
        std = 'std_recall'
        _, p_stat_r = get_p(row, reference_row, res, std)
        df.at[index, "p_stat_recall"] = p_stat_r

        # Calculate p-value for precision
        res = 'precision'
        std = 'std_precision'
        _, p_stat_p = get_p(row, reference_row, res, std)
        df.at[index, "p_stat_precision"] = p_stat_p
    df = df.sort_values(by="test", ascending=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(results_file, index=False)
    print(f"Updated results saved to {results_file}")


if __name__ == "__main__":
    #path = os.path.join(get_root_dir(), 'result')
    #print(f"Path: {path}")
    #for folder in os.listdir(path):
    folder = os.path.join(get_root_dir(), 'results_rev_cls')
    #folder = os.path.join('result', folder)
    print(f"Processing folder: {folder}")
    make_result_file(folder)
    print(f"Result file created in {folder} folder.")
    results_file = os.path.join(folder, 'result.csv')
    calc_p(folder, results_file)
