import pandas as pd
import sys
import os
from utils.utilities import get_root_dir
import pandas as pd

'''
Creates overleaf tables from the csv files in the result folder
'''

def calculate_avg_map50(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Calculate the average of the 'map50' column
    avg_map50 = df["map50"].mean()
    
    print(f"Average mAP@50: {avg_map50}")
    return avg_map50

def to_tab(filepath, name):
    file = os.path.join(filepath, name)
    print("")
    print(file)
    print(filepath)
    print("")
    # Load your CSV file
    df = pd.read_csv(file)
    calculate_avg_map50(file)

    # Start writing LaTeX lines
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{c|l|c|c|c|c|c}",
        r"\textbf{Test \#} & \textbf{Name} & \textbf{mAP@50} & \textbf{mAP@50-95} & \textbf{Recall} & \textbf{Precision} & \textbf{p\_stat\_50} \\",
        r"\hline"
    ]

    # Add rows dynamically
    for _, row in df.iterrows():
        latex_lines.append(
            f"{int(row['test'])} &  & {row['map50']:.3f} & {row['map50_95']:.3f} & {row['recall']:.3f} & {row['precision']:.3f} & {row['p_stat_map50']:.3f} \\\\"
        )

    # Finish the LaTeX table
    latex_lines += [
        r"\end{tabular}",
        r"\caption{Evaluation Results}",
        r"\label{tab:evaluation-results}",
        r"\end{table}"
    ]

    # Write to .txt file
    save_path = os.path.join(filepath, "evaluation_table.txt")
    print(save_path)
    with open(save_path, "w") as f:
        f.write("\n".join(latex_lines))

    print("LaTeX table saved to evaluation_table.txt")


if __name__ == "__main__":
    folder = "result"
    path = os.path.join(get_root_dir(), folder)
    for file in os.listdir(path): #this is a folder, i need to check if theres a csv file in it
        print(file)
        if "plot" in file:
            continue
        if not os.path.isdir(os.path.join(path, file)):  # Check if it's not a folder
            continue
        for x in os.listdir(os.path.join(path, file)):
            if x.endswith(".csv"):
                l = os.path.join(path, file)
                print(l)
                print(x)
                to_tab(l, x)
            