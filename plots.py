import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import ast
import yaml
from utils.utilities import get_root_dir


# Helper function to determine result type based on folder name
def get_result_type(folder):
    if "rev" in folder and "cls" in folder:
        return "Restricted With Classes"
    elif "rev" in folder:
        return "Restricted Class Agnostic"
    elif "cls" in folder:
        return "Wild With Classes"
    else:
        return "Wild Class Agnostic"


# Helper function to save plots
def save_plot(fig, folder, name, base_folder="plots"):
    plot_folder = os.path.join(get_root_dir(), 'result', base_folder, folder)
    os.makedirs(plot_folder, exist_ok=True)  # Create the folder if it doesn't exist
    save_path = os.path.join(plot_folder, name)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")


# Helper function to set up plot labels, titles, and limits
def setup_plot(ax, x, y, title, x_labels, xlabel="Degradation Amount", ylabel="mAP@50"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)  # Set y-axis range from 0 to 1
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(axis='y', linestyle='--', alpha=0.7)


# Helper function to extract class-specific data
def get_class_data(df, cls):
    y = []
    for _, row in df.iterrows():
        matched_classes = ast.literal_eval(row["matched_classes"])
        ap_per_class = ast.literal_eval(row["AP_per_class"])
        if cls in matched_classes:
            cls_index = matched_classes.index(cls)
            y.append(ap_per_class[cls_index][0])  # Use the index to get the correct mAP@50 value
        else:
            y.append(0)  # If the class is not matched, append 0
    return y


# Helper function to darken a color
def darken_color(color, factor=0.3):
    """
    Darkens a given color by multiplying its RGB values by a factor.
    """
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    return tuple([max(0, min(1, c * factor)) for c in rgb])


# Helper function to add a fitted line to a plot
def add_fitted_line(ax, x, y, color, label="Fitted Line"):
    coefficients = np.polyfit(x, y, 1)  # Linear fit (degree=1)
    fitted_line = np.polyval(coefficients, x)
    darker_color = darken_color(color)  # Darken the color for the fitted line
    ax.plot(x, fitted_line, color=darker_color, linestyle="--", label=label)


def get_class_name(cls_index):
    """
    Returns the class name based on the index.
    """
    class_names = [
        'bottle-glass', 'bottle-plastic', 'cup-disposable', 'cup-handle',
        'glass-mug', 'glass-normal', 'glass-wine', 'gym bottle', 'tin can'
    ]
    return class_names[cls_index]


# Plot function
def plot(results_file, folder, cls=None, result_type="Wild Class Agnostic"):
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Define unique colors for each category
    colors = ["red", "blue", "green", "orange"]

    for i, (category, test_numbers) in enumerate(plots.items()):
        filtered_df = df[df["test"].isin(test_numbers)].sort_values(by="test")
        x = np.arange(1, len(filtered_df) + 1)
        y = get_class_data(filtered_df, cls) if cls is not None else filtered_df["map50"]

        # Plot the data as scatter points
        axes[i].scatter(x, y, color=colors[i], s=100, label=category)

        # Add a fitted line with a slightly darker color
        add_fitted_line(axes[i], x, y, color=colors[i])

        # Set up the plot (only the category name as the title for each subplot)
        setup_plot(axes[i], x, y, title=category, x_labels=label[category])

    # Determine if the folder is class-specific
    is_class_specific = "cls" in folder

    # Add a single title for the entire figure
    class_name = f" - {get_class_name(cls)}" if is_class_specific and cls is not None else ""
    fig.suptitle(f"{result_type}{class_name}", fontsize=16, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the suptitle
    save_plot(fig, folder, f"plots_{folder}{class_name}.png")


# Big plot function
def big_plot(results_file, folder, cls=None, result_type="Wild Class Agnostic"):
    df = pd.read_csv(results_file)
    x = np.arange(1, len(df) + 1)
    y = get_class_data(df, cls) if cls is not None else df["map50"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the data as scatter points
    ax.scatter(x, y, color="blue", s=100, label="mAP@50")

    # Add a fitted line with a slightly darker color
    add_fitted_line(ax, x, y, color="blue")

    # Set up the plot
    setup_plot(ax, x, y, title="", x_labels=x)  # No title for the single plot

    # Determine if the folder is class-specific
    is_class_specific = "cls" in folder

    # Add a single title for the entire figure
    class_name = f" - {get_class_name(cls)}" if is_class_specific and cls is not None else ""
    fig.suptitle(f"{result_type}{class_name}", fontsize=16, fontweight="bold")

    save_plot(fig, folder, f"big_plot_{folder}_class_{cls if cls else 'overall'}.png")


# Combined plot function
def combined_plot(results_file, folder):
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

    # Define markers for each class
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', '+']

    df = pd.read_csv(results_file)
    all_classes = sorted(set(cls for matched in df["matched_classes"] for cls in ast.literal_eval(matched)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_classes)))
    class_color_map = {cls: colors[i] for i, cls in enumerate(all_classes)}
    class_marker_map = {cls: markers[i % len(markers)] for i, cls in enumerate(all_classes)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (category, test_numbers) in enumerate(plots.items()):
        filtered_df = df[df["test"].isin(test_numbers)].sort_values(by="test")
        x = np.arange(1, len(filtered_df) + 1)

        for cls in all_classes:
            y = get_class_data(filtered_df, cls)

            # Plot the line for the class
            axes[i].plot(
                x, y,
                label=get_class_name(cls),
                color=class_color_map[cls],
                linewidth=1
            )

            # Add scatter points with unique markers
            axes[i].scatter(
                x, y,
                color=class_color_map[cls],
                marker=class_marker_map[cls],
                s=50  # Marker size
            )

        # Set up the plot (only the category name as the title for each subplot)
        axes[i].set_xlabel("Degradation Amount")
        axes[i].set_ylabel("mAP@50")
        axes[i].set_title(category)  # Only the category name as the title
        axes[i].set_ylim(0, 1)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(label[category])
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # Add a single title for the entire figure
    fig.suptitle(f"Combined Plot - {folder}", fontsize=16, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the suptitle
    handles = [
        plt.Line2D([0], [0], color=class_color_map[cls], marker=class_marker_map[cls], markersize=8, label=get_class_name(cls))
        for cls in all_classes
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=10)
    save_plot(fig, folder, f"combined_plots_{folder}.png")


# Main execution block
if __name__ == "__main__":
    args = sys.argv[1:]  # Get all arguments after the script name
    run_big_plot = "big" in args  # Check if "big" is in the arguments
    cls_mode = "cls" in args  # Check if "cls" is in the arguments
    combined_mode = "combined" in args  # Check if "combined" is in the arguments

    for folder in os.listdir(os.path.join(get_root_dir(), 'result')):
        for file in os.listdir(os.path.join(get_root_dir(), 'result', folder)):
            if 'result.csv' in file:
                result_file = os.path.join(get_root_dir(), 'result', folder, file)
                result_type = get_result_type(folder)

                # Load the DataFrame and extract all unique classes
                df = pd.read_csv(result_file)
                all_classes = sorted(set(cls for matched in df["matched_classes"] for cls in ast.literal_eval(matched)))

                if combined_mode:
                    # Run combined plot
                    yaml_file = os.path.join(get_root_dir(), 'result', folder, 'classes.yaml')
                    combined_plot(result_file, folder)
                elif cls_mode:
                    # Generate class-specific plots
                    for cls in all_classes:
                        if run_big_plot:
                            big_plot(result_file, folder, cls=cls, result_type=result_type)
                        else:
                            plot(result_file, folder, cls=cls, result_type=result_type)
                else:
                    # Generate overall plots
                    if run_big_plot:
                        big_plot(result_file, folder, result_type=result_type)
                    else:
                        plot(result_file, folder, result_type=result_type)