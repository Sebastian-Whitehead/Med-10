import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import ast
import yaml
from utils.utilities import get_root_dir


def get_result_type(folder):
    """Determine result type string based on folder name."""
    if "rev" in folder and "cls" in folder:
        return "Restricted With Classes"
    elif "rev" in folder:
        return "Restricted Class Agnostic"
    elif "cls" in folder:
        return "Wild With Classes"
    else:
        return "Wild Class Agnostic"


def save_plot(fig, folder, name, base_folder="plots"):
    """Save a matplotlib figure to the appropriate result folder."""
    plot_folder = os.path.join(get_root_dir(), 'result', base_folder, folder)
    os.makedirs(plot_folder, exist_ok=True)
    save_path = os.path.join(plot_folder, name)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")


def setup_plot(ax, x, y, title, x_labels, xlabel="Degradation Amount", ylabel="mAP@50"):
    """Set up axis labels, title, limits, and grid for a plot."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def get_class_data(df, cls):
    """Extract mAP@50 values for a specific class from a DataFrame."""
    y = []
    for _, row in df.iterrows():
        matched_classes = ast.literal_eval(row["matched_classes"])
        ap_per_class = ast.literal_eval(row["AP_per_class"])
        if cls in matched_classes:
            cls_index = matched_classes.index(cls)
            y.append(ap_per_class[cls_index][0])
        else:
            y.append(0)
    return y


def darken_color(color, factor=0.3):
    """Darken a matplotlib color by a given factor."""
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    return tuple([max(0, min(1, c * factor)) for c in rgb])


def add_fitted_line(ax, x, y, color, label="Fitted Line"):
    """Add a linear fitted line to a plot."""
    coefficients = np.polyfit(x, y, 1)
    fitted_line = np.polyval(coefficients, x)
    darker_color = darken_color(color)
    ax.plot(x, fitted_line, color=darker_color, linestyle="--", label=label)


def get_class_name(cls_index):
    """Return the class name for a given index."""
    class_names = [
        'bottle-glass', 'bottle-plastic', 'cup-disposable', 'cup-handle',
        'glass-mug', 'glass-normal', 'glass-wine', 'gym bottle', 'tin can'
    ]
    return class_names[cls_index]


def plot(results_file, folder, cls=None, result_type="Wild Class Agnostic"):
    """Plot mAP@50 for each category, optionally for a specific class."""
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
    colors = ["red", "blue", "green", "orange"]

    for i, (category, test_numbers) in enumerate(plots.items()):
        filtered_df = df[df["test"].isin(test_numbers)].sort_values(by="test")
        x = np.arange(1, len(filtered_df) + 1)
        y = get_class_data(filtered_df, cls) if cls is not None else filtered_df["map50"]
        axes[i].scatter(x, y, color=colors[i], s=100, label=category)
        add_fitted_line(axes[i], x, y, color=colors[i])
        setup_plot(axes[i], x, y, title=category, x_labels=label[category])

    is_class_specific = "cls" in folder
    class_name = f" - {get_class_name(cls)}" if is_class_specific and cls is not None else ""
    fig.suptitle(f"{result_type}{class_name}", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(fig, folder, f"plots_{folder}{class_name}.png")


def big_plot(results_file, folder, cls=None, result_type="Wild Class Agnostic"):
    """Plot a single big mAP@50 plot for all tests, optionally for a specific class."""
    df = pd.read_csv(results_file)
    x = np.arange(0, len(df))
    y = get_class_data(df, cls) if cls is not None else df["map50"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color="blue", s=100, label="mAP@50")
    add_fitted_line(ax, x, y, color="blue")
    setup_plot(ax, x, y, title="", x_labels=x)
    is_class_specific = "cls" in folder
    class_name = f" - {get_class_name(cls)}" if is_class_specific and cls is not None else ""
    fig.suptitle(f"{result_type}{class_name}", fontsize=16, fontweight="bold")
    print(folder)
    save_plot(fig, folder, f"big_plot_{folder}_class_{cls if cls else 'overall'}.png")


def combined_plot(results_file, folder):
    """Plot all classes together with unique colors and markers for each class."""
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
    all_classes = set()
    for matched in df["matched_classes"]:
        all_classes.update(ast.literal_eval(matched))
    all_classes = sorted(all_classes)
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_classes)))
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p']
    class_color_map = {cls: colors[i] for i, cls in enumerate(all_classes)}
    class_marker_map = {cls: markers[i % len(markers)] for i, cls in enumerate(all_classes)}
    result_type = get_result_type(folder)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (category, test_numbers) in enumerate(plots.items()):
        filtered_df = df[df["test"].isin(test_numbers)].sort_values(by="test")
        x = np.arange(len(filtered_df))
        for cls in all_classes:
            y = []
            for _, row in filtered_df.iterrows():
                matched_classes = ast.literal_eval(row["matched_classes"])
                ap_per_class = ast.literal_eval(row["AP_per_class"])
                if cls in matched_classes:
                    cls_index = matched_classes.index(cls)
                    y.append(ap_per_class[cls_index][0])
                else:
                    y.append(0)
            class_name = get_class_name(cls)
            axes[i].plot(x, y, label=f"{class_name}", color=class_color_map[cls], linewidth=1)
            axes[i].scatter(x, y, color=class_color_map[cls], marker=class_marker_map[cls], s=30, alpha=0.7)
        axes[i].set_xlabel("Degradation Amount")
        axes[i].set_ylabel("mAP@50")
        axes[i].set_title(f"{category}")
        axes[i].set_ylim(0, 1)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(label[category])
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle(f"{result_type} - {folder}", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    handles = [
        plt.Line2D([0], [0], color=class_color_map[cls], marker=class_marker_map[cls], markersize=6, linewidth=0, label=get_class_name(cls))
        for cls in all_classes
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=10)
    save_plot(fig, folder, f"combined_plots_{folder}.png")


# Main execution block
if __name__ == "__main__":
    args = sys.argv[1:]
    run_big_plot = "big" in args
    cls_mode = "cls" in args
    combined_mode = "combined" in args
    folder_name = "fine_tune_sets"

    for folder in os.listdir(os.path.join(get_root_dir(), folder_name)):
        folder_path = os.path.join(get_root_dir(), folder_name, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if 'result.csv' in file:
                result_file = os.path.join(folder_path, file)
                print(result_file)
                result_type = get_result_type(folder)
                df = pd.read_csv(result_file)
                all_classes = sorted(set(cls for matched in df["matched_classes"] for cls in ast.literal_eval(matched)))
                if combined_mode:
                    combined_plot(result_file, folder)
                elif cls_mode:
                    for cls in all_classes:
                        if run_big_plot:
                            big_plot(result_file, folder, cls=cls, result_type=result_type)
                        else:
                            plot(result_file, folder, cls=cls, result_type=result_type)
                else:
                    if run_big_plot:
                        big_plot(result_file, folder, result_type=result_type)
                    else:
                        plot(result_file, folder, result_type=result_type)