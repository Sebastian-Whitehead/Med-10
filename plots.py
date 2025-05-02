from matplotlib import axis
from matplotlib.pylab import f
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.utilities import get_root_dir
import json
import sys
import numpy as np
import ast
import yaml

def plot(results_file, folder, cls=None):
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

        # Generate sequential x-values (1, 2, 3, ...)
        x = np.arange(1, len(filtered_df) + 1)

        if cls is not None:
            # Get the mAP@50 values for the specified class
            y = []
            for _, row in filtered_df.iterrows():
                matched_classes = ast.literal_eval(row["matched_classes"])
                ap_per_class = ast.literal_eval(row["AP_per_class"])
                if cls in matched_classes:
                    cls_index = matched_classes.index(cls)  # Find the index of the class in matched_classes
                    y.append(ap_per_class[cls_index][0])  # Use the index to get the correct mAP@50 value
                else:
                    y.append(0)  # If the class is not matched, append 0
        else:
            # Use overall mAP@50 values
            y = filtered_df["map50"]

        # Plot the data as scatter points
        axes[i].scatter(x, y, color=colors[i], s=100, label=category)

        # Add a fitted line
        if len(x) > 1:  # Ensure there are enough points to fit a line
            coeffs = np.polyfit(x, y, 1)  # Fit a linear regression line (degree=1)
            poly = np.poly1d(coeffs)  # Create a polynomial function
            fitted_y = poly(x)  # Generate y-values for the fitted line
            axes[i].plot(x, fitted_y, color=colors[i], linestyle="--", label=f"{category} (fit)")

        # Set labels, title, and limits
        axes[i].set_xlabel("Data Points")
        axes[i].set_ylabel("mAP@50")
        axes[i].set_title(f"Category: {category}")
        axes[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1

        # Treat x-axis as categorical labels
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(filtered_df["test"])  # Use the original test numbers as labels

        # Add grid lines only on the y-axis
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Determine the base folder (cls or total)
    base_folder = "cls" if cls is not None else "total"
    sub_folder = "combined_plots"  # Subfolder for combined plots

    # Add class name to the title and filename if cls is specified
    class_name = f"_class_{cls}" if cls is not None else ""
    plot_folder = os.path.join(get_root_dir(), 'result', 'plots', base_folder, sub_folder, folder)  # Save in cls/combined_plots/ or total/combined_plots/
    os.makedirs(plot_folder, exist_ok=True)  # Create the folder if it doesn't exist
    name = f"plots_{folder}{class_name}.png"
    if cls is not None:
        fig.suptitle(f"Class {cls} mAP@50", fontsize=16)  # Add class name to the top of the graph
    print(f"Saving plot as: {name}")
    print(f"Saving plot in: {os.path.join(plot_folder, name)}")
    plt.savefig(os.path.join(plot_folder, name), dpi=300, bbox_inches='tight')

    # Show all plots in the same window
    # plt.show()


def big_plot(results_file, folder, cls=None):
    df = pd.read_csv(results_file)

    # Create the plot
    fig = plt.figure(figsize=(8, 6))

    if cls is not None:
        # Get the mAP@50 values for the specified class
        y = []
        for _, row in df.iterrows():
            matched_classes = ast.literal_eval(row["matched_classes"])
            ap_per_class = ast.literal_eval(row["AP_per_class"])
            if cls in matched_classes:
                cls_index = matched_classes.index(cls)  # Find the index of the class in matched_classes
                y.append(ap_per_class[cls_index][0])  # Use the index to get the correct mAP@50 value
            else:
                y.append(0)  # If the class is not matched, append 0
    else:
        # Use overall mAP@50 values
        y = df["map50"]

    x = df["test"].astype(float)  # Convert test numbers to float for fitting

    plt.scatter(x, y, color='blue', s=100, label='mAP@50')

    # Add a fitted line
    if len(x) > 1:  # Ensure there are enough points to fit a line
        coeffs = np.polyfit(x, y, 1)  # Fit a linear regression line (degree=1)
        poly = np.poly1d(coeffs)  # Create a polynomial function
        fitted_y = poly(x)  # Generate y-values for the fitted line
        plt.plot(x, fitted_y, color=(0.2, 0.2, 0.4), linestyle="--", label="Fitted Line")

    # Set labels, title, and limits
    plt.xlabel("Test Numbers")
    plt.ylabel("mAP@50")
    plt.title(f"Category: {folder}")
    plt.ylim(0, 1)  # Set y-axis range from 0 to 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(df["test"])
    plt.legend()

    # Determine the base folder (cls or total)
    base_folder = "cls" if cls is not None else "total"
    sub_folder = "bigplots"  # Subfolder for big plots

    # Add class name to the title and filename if cls is specified
    class_name = f"_class_{cls}" if cls is not None else ""
    plot_folder = os.path.join(get_root_dir(), 'result', 'plots', base_folder, sub_folder, folder)  # Save in cls/bigplots/ or total/bigplots/
    os.makedirs(plot_folder, exist_ok=True)  # Create the folder if it doesn't exist
    name = f"plots_{folder}{class_name}.png"
    if cls is not None:
        plt.title(f"Class {cls} mAP@50", fontsize=16)  # Add class name to the top of the graph
    print(f"Saving plot as: {name}")
    print(f"Saving plot in: {os.path.join(plot_folder, name)}")
    plt.savefig(os.path.join(plot_folder, name), dpi=300, bbox_inches='tight')
    # plt.show()


def combined_plot(results_file, folder, yaml_file):
    # Load class names from the YAML file
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data['names']  # List of class names

    plots = {
        "lighting": [0, 1, 2, 3],
        "mip map": [0, 4, 5, 6, 7, 8, 9],
        "poly count wos": [0, 10, 11, 12, 13],
        "poly count ws": [0, 14, 15, 16, 17]
    }
    label = {
        "lighting": ['raytracing', 'high', 'medium', 'low'],
        "mip map": [0, 1, 2, 3, 4, 6, 8],
        "poly count wos": [0, 0.8, 0.6, 0.4, 0.2],
        "poly count ws": [0, 0.8, 0.6, 0.4, 0.2]
    }

    df = pd.read_csv(results_file)

    # Collect all unique classes from the dataset
    all_classes = set()
    for matched in df["matched_classes"]:
        all_classes.update(ast.literal_eval(matched))
    all_classes = sorted(all_classes)  # Sort classes for consistency

    # Assign a unique color to each class
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_classes)))  # Use a colormap for distinguishable colors
    class_color_map = {cls: colors[i] for i, cls in enumerate(all_classes)}

    # Create a figure with 4 subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    # Iterate over each category and create a subplot
    for i, (category, test_numbers) in enumerate(plots.items()):
        # Filter the DataFrame for the current category's test numbers
        filtered_df = df[df["test"].isin(test_numbers)]

        # Sort by test numbers to ensure proper order
        filtered_df = filtered_df.sort_values(by="test")

        # Generate sequential x-values (1, 2, 3, ...)
        x = np.arange(1, len(filtered_df) + 1)

        # Plot a line for each class
        for cls in all_classes:
            y = []
            for _, row in filtered_df.iterrows():
                matched_classes = ast.literal_eval(row["matched_classes"])
                ap_per_class = ast.literal_eval(row["AP_per_class"])
                if cls in matched_classes:
                    cls_index = matched_classes.index(cls)  # Find the index of the class in matched_classes
                    y.append(ap_per_class[cls_index][0])  # Use the index to get the correct mAP@50 value
                else:
                    y.append(0)  # If the class is not matched, append 0

            # Plot the line for the current class
            class_name = class_names[cls]  # Get the class name from the YAML file
            axes[i].plot(x, y, label=f"{class_name}", color=class_color_map[cls], linewidth=1)

        # Set labels, title, and limits
        axes[i].set_xlabel("Degredation amount")
        axes[i].set_ylabel("mAP@50")
        axes[i].set_title(f"Category: {category}")
        axes[i].set_ylim(0, 1)  # Set y-axis range from 0 to 1
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(label[category])  # Use the labels from the `label` dictionary
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Add a single legend outside the plots
    handles = [plt.Line2D([0], [0], color=class_color_map[cls], linewidth=2, label=f"{class_names[cls]}") for cls in all_classes]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=10)

    # Determine the base folder (always "combined_plots" for this function)
    base_folder = "combined_plots"
    plot_folder = os.path.join(get_root_dir(), 'result', 'plots', base_folder, folder)
    os.makedirs(plot_folder, exist_ok=True)  # Create the folder if it doesn't exist
    name = f"combined_plots_{folder}.png"
    print(f"Saving plot as: {name}")
    print(f"Saving plot in: {os.path.join(plot_folder, name)}")
    plt.savefig(os.path.join(plot_folder, name), dpi=300, bbox_inches='tight')

    # Show the plot
    # plt.show()


if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "big":
        run_big_plot = True
    elif len(sys.argv) > 1 and sys.argv[1] == "combined":
        run_combined_plot = True
    else:
        run_big_plot = False
        run_combined_plot = False

    cls_mode = True if len(sys.argv) > 2 and sys.argv[2] == "cls" else False
    yaml_file = os.path.join(get_root_dir(), 'batch_images', 'test_set', 'data.yaml')

    for folder in os.listdir(os.path.join(get_root_dir(), 'result')):
        for file in os.listdir(os.path.join(get_root_dir(), 'result', folder)):
            if 'result.csv' in file:
                result_file = os.path.join(get_root_dir(), 'result', folder, file)
                print(f"Plotting results from: {result_file}")

                # Load the CSV to determine the classes dynamically
                df = pd.read_csv(result_file)
                all_classes = set()
                for matched in df["matched_classes"]:
                    all_classes.update(ast.literal_eval(matched))  # Collect all unique classes

                # Generate plots based on the selected mode
                if run_combined_plot:
                    combined_plot(result_file, folder, yaml_file)
                elif cls_mode:
                    for cls in sorted(all_classes):  # Iterate over all unique classes
                        if run_big_plot:
                            big_plot(result_file, folder, cls=cls)
                        else:
                            plot(result_file, folder, cls=cls)
                else:
                    # If cls_mode is False, generate overall plots
                    if run_big_plot:
                        big_plot(result_file, folder)
                    else:
                        plot(result_file, folder)