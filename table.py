import pandas as pd

# Load your CSV file
df = pd.read_csv("result.csv")

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
with open("evaluation_table.txt", "w") as f:
    f.write("\n".join(latex_lines))

print("LaTeX table saved to evaluation_table.txt")
