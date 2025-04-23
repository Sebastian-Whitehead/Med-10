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

def calc_p(results_file):

    # Load the CSV file
    df = pd.read_csv(results_file)  # Replace with your actual file path

    # Extract the row where test == 2
    reference_row = df[df["test"] == 2]
    if reference_row.empty:
        raise ValueError("No row found where test == 2")

    # Extract the first (and assuming only) reference row
    reference_row = reference_row.iloc[0]

    # Drop the test == 2 row from comparison
    comparison_rows = df[df["test"] != 2]

    # Specify the columns you want to compare
    columns_to_compare = ["map50", "std_dev_50"]
    n = 201
    # Iterate and compare
    for index, row in comparison_rows.iterrows():
        print(f"Comparing test == {row['test']}:")
        for col in columns_to_compare:
            if col == "map50":
                ref_map = reference_row[col]
                row_map = row[col]
            if col == "std_dev_50":
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
        print(f"Comparing {col}:")
        print(f'map50: {ref_map} vs {row_map}')
        print(f'std_dev_50: {ref_std} vs {row_std}')
        print(f"t-statistic: {t_stat}, p-value: {p_value}")
        print()


if __name__ == "__main__":
    folder = 'result_cls_bad'
    #make_result_file(folder)
    #print(f"Result file created in {folder} folder.")
    results_file = os.path.join(folder, 'result.csv')
    calc_p(results_file)