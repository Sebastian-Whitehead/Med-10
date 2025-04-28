from matplotlib import axis
from matplotlib.pylab import f
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.utilities import get_root_dir
import json
import sys

def plot(results_file, folder):
    # Define the categories and their corresponding test numbers
    plots = {
        "lighting": [0, 1, 2, 3],
        "mip map": [0, 4, 5, 6, 7, 8, 9],
        "poly count wos": [0, 10, 11, 12, 13],
        "poly count ws": [0, 14, 15, 16, 17]
    }

    df = pd.read_csv(results_file)

    # Create a figure with 4 subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    # Define colors for each category
    colors = ["red", "blue", "green", "orange"]

    # Iterate over each category and create a subplot
    for i, (category, test_numbers) in enumerate(plots.items()):
        # Filter the DataFrame for the current category's test numbers
        filtered_df = df[df["test"].isin(test_numbers)]

        # Sort by test numbers to ensure proper order
        filtered_df = filtered_df.sort_values(by="test")

        # Convert test numbers to strings for even spacing
        filtered_df["test"] = filtered_df["test"].astype(str)

        # Plot the data as scatter points
        axes[i].scatter(filtered_df["test"], filtered_df["map50"], color=colors[i], s=100, label=category)  # `s=100` makes dots bigger

        # Set labels, title, and limits
        axes[i].set_xlabel("Test Numbers")
        axes[i].set_ylabel("mAP@50")
        axes[i].set_title(f"Category: {category}")
        axes[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1

        # Treat x-axis as categorical labels
        axes[i].set_xticks(filtered_df["test"])  # Set the test numbers as ticks
        axes[i].set_xticklabels(filtered_df["test"])  # Use string labels for even spacing

        # Add grid lines only on the y-axis
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)  # Dashed grid lines for better readability
        #axes[i].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    name = f"plots_{folder}_sep.png"
    print(f"Saving plot as: {name}")
    print(f"Saving plot in: {os.path.join(get_root_dir(), 'result', 'plots', name)}")
    plt.savefig(os.path.join(get_root_dir(), 'result', 'plots', name), dpi=300, bbox_inches='tight')

    # Show all plots in the same window
    #plt.show()

def big_plot(results_file, folder):
    df = pd.read_csv(results_file)

    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(df["test"], df["map50"], color='blue', s=100, label='mAP@50')
    plt.xlabel("Test Numbers")  # Corrected method
    plt.ylabel("mAP@50")        # Corrected method
    plt.title(f"Category: {folder}")  # Corrected method
    plt.ylim(0, 1)  # Set y-axis range from 0 to 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Dashed grid lines for better readability
    plt.xticks(df["test"])  # Set the test numbers as ticks


    name = f"plots_{folder}.png"
    print(f"Saving plot as: {name}")
    print(f"Saving plot in: {os.path.join(get_root_dir(), 'result', 'plots', name)}")
    plt.savefig(os.path.join(get_root_dir(), 'result', 'plots', name), dpi=300, bbox_inches='tight')
    # Show the plot
    #plt.show()

if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "big":
        run_big_plot = True
    else:
        run_big_plot = False

    for folder in os.listdir(os.path.join(get_root_dir(), 'result')):
        for file in os.listdir(os.path.join(get_root_dir(), 'result', folder)):
            if 'result.csv' in file:
                result_file = os.path.join(get_root_dir(), 'result', folder, file)
                print(f"Plotting results from: {result_file}")

                if run_big_plot:
                    big_plot(result_file, folder)
                else:
                    plot(result_file, folder)