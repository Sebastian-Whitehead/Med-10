import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

def plot_map_per_class(results_file):
    df = pd.read_csv(results_file)

    for _, row in df.iterrows():
        matched_classes = ast.literal_eval(row["matched_classes"])
        ap_per_class = ast.literal_eval(row["AP_per_class"])

        map_50_per_class = [ap[0] for ap in ap_per_class]

        plt.figure(figsize=(10, 6))
        plt.bar(matched_classes, map_50_per_class, color="skyblue")
        plt.xlabel("Class ID")
        plt.ylabel("mAP@50")
        plt.title(f"mAP@50 per Class (Test {row['test']})")
        plt.xticks(matched_classes) 
        plt.ylim(0, 1) 
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.close()

def plot_combined_map_per_class(results_file):
    plots = {
        "lighting": [0, 1, 2, 3],
        "mip map": [0, 4, 5, 6, 7, 8, 9],
        "poly count wos": [0, 10, 11, 12, 13],
        "poly count ws": [0, 14, 15, 16, 17]
    }

    df = pd.read_csv(results_file)
    category_map = {category: {} for category in plots.keys()}
    for category, test_numbers in plots.items():
        filtered_df = df[df["test"].isin(test_numbers)]
        for _, row in filtered_df.iterrows():
            matched_classes = ast.literal_eval(row["matched_classes"])
            ap_per_class = ast.literal_eval(row["AP_per_class"])
            map_50_per_class = [ap[0] for ap in ap_per_class]
            for cls, map_50 in zip(matched_classes, map_50_per_class):
                if cls not in category_map[category]:
                    category_map[category][cls] = []
                category_map[category][cls].append(map_50)

    avg_map_per_category = {
        category: {cls: sum(values) / len(values) for cls, values in cls_map.items()}
        for category, cls_map in category_map.items()
    }

    all_classes = sorted(set(cls for cls_map in avg_map_per_category.values() for cls in cls_map.keys()))
    x = np.arange(len(all_classes))  
    width = 0.2
    plt.figure(figsize=(12, 6))
    for i, (category, cls_map) in enumerate(avg_map_per_category.items()):
        y = [cls_map.get(cls, 0) for cls in all_classes]
        plt.bar(x + i * width, y, width, label=category)

    plt.xlabel("Class ID")
    plt.ylabel("Average mAP@50")
    plt.title("Average mAP@50 per Class Across Test Categories")
    plt.xticks(x + width * (len(plots) - 1) / 2, all_classes)  
    plt.ylim(0, 1)  
    plt.legend(title="Test Categories")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("combined_map_per_class.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    results_file = "c:/Users/rebec/OneDrive/Documents/GitHub/Med-10/result/results_cls/result.csv"
    plot_map_per_class(results_file)
    plot_combined_map_per_class(results_file)