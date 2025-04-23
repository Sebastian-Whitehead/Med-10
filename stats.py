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
    results_path = os.path.join(get_root_dir(), results_file)
    pd = pd.read_csv(results_path)
    pd = pd.dropna()
    pd_og = pd.DataFrame()
    pd_results = pd.DataFrame()
    for row in pd.iterrows():
        if row[1]['test'] == '2':
            pd_og = pd_og.append(row[1])
            print("row 2 added")
        else:
            pd_results = pd_results.append(row[1])
            print("other row added")


if __name__ == "__main__":
    folder = 'result_cls_bad'
    #make_result_file(folder)
    #print(f"Result file created in {folder} folder.")
    results_file = os.path.join(folder, 'result.csv')
    calc_p(results_file)