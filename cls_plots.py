import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

def plot_map_per_class(results_file):
    # Load the results CSV
    df = pd.read_csv(results_file)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Parse matched_classes and AP_per_class
        matched_classes = ast.literal_eval(row["matched_classes"])
        ap_per_class = ast.literal_eval(row["AP_per_class"])

        # Flatten the AP_per_class array to get mAP@50 values (first value for each class)
        map_50_per_class = [ap[0] for ap in ap_per_class]

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(matched_classes, map_50_per_class, color="skyblue")
        plt.xlabel("Class ID")
        plt.ylabel("mAP@50")
        plt.title(f"mAP@50 per Class (Test {row['test']})")
        plt.xticks(matched_classes)  # Set x-axis ticks to class IDs
        plt.ylim(0, 1)  # Set y-axis range from 0 to 1
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Save the plot
        #plt.savefig(f"map_per_class_test_{row['test']}.png", dpi=300, bbox_inches="tight")
        plt.close()

def plot_combined_map_per_class(results_file):
    # Define the test categories
    plots = {
        "lighting": [0, 1, 2, 3],
        "mip map": [0, 4, 5, 6, 7, 8, 9],
        "poly count wos": [0, 10, 11, 12, 13],
        "poly count ws": [0, 14, 15, 16, 17]
    }

    # Load the results CSV
    df = pd.read_csv(results_file)

    # Initialize a dictionary to store aggregated mAP values per category and class
    category_map = {category: {} for category in plots.keys()}

    # Iterate over each category in the plots dictionary
    for category, test_numbers in plots.items():
        # Filter the DataFrame for the current category's test numbers
        filtered_df = df[df["test"].isin(test_numbers)]

        # Aggregate mAP values for each class
        for _, row in filtered_df.iterrows():
            # Parse matched_classes and AP_per_class
            matched_classes = ast.literal_eval(row["matched_classes"])
            ap_per_class = ast.literal_eval(row["AP_per_class"])

            # Flatten the AP_per_class array to get mAP@50 values (first value for each class)
            map_50_per_class = [ap[0] for ap in ap_per_class]

            # Store mAP values for each class in the current category
            for cls, map_50 in zip(matched_classes, map_50_per_class):
                if cls not in category_map[category]:
                    category_map[category][cls] = []
                category_map[category][cls].append(map_50)

    # Compute the average mAP@50 for each class in each category
    avg_map_per_category = {
        category: {cls: sum(values) / len(values) for cls, values in cls_map.items()}
        for category, cls_map in category_map.items()
    }

    # Get a sorted list of all unique class IDs
    all_classes = sorted(set(cls for cls_map in avg_map_per_category.values() for cls in cls_map.keys()))

    # Create a grouped bar plot
    x = np.arange(len(all_classes))  # X positions for the classes
    width = 0.2  # Width of each bar

    plt.figure(figsize=(12, 6))

    # Plot bars for each category
    for i, (category, cls_map) in enumerate(avg_map_per_category.items()):
        # Get mAP values for all classes (fill missing classes with 0)
        y = [cls_map.get(cls, 0) for cls in all_classes]
        plt.bar(x + i * width, y, width, label=category)

    # Set plot labels and title
    plt.xlabel("Class ID")
    plt.ylabel("Average mAP@50")
    plt.title("Average mAP@50 per Class Across Test Categories")
    plt.xticks(x + width * (len(plots) - 1) / 2, all_classes)  # Center x-ticks
    plt.ylim(0, 1)  # Set y-axis range from 0 to 1
    plt.legend(title="Test Categories")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig("combined_map_per_class.png", dpi=300, bbox_inches="tight")
    plt.show()

# Example usage
results_file = "c:/Users/rebec/OneDrive/Documents/GitHub/Med-10/result/results_cls/result.csv"
plot_map_per_class(results_file)
plot_combined_map_per_class(results_file)