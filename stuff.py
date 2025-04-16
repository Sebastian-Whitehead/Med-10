import numpy as np

def evaluate_batch_11(batch_path, log=False):
    individual_map_values_50 = []  # List to store mAP@50 
    individual_map_values_95 = []  # List to store mAP@50:95
values for each image

    for idx, result in enumerate(results):
        annotations = annotations_list[idx]
        
        # Update the metric for the current image
        map_metric.update(result, [annotations])
        
        # Compute mAP for the current image
        eval_metric = map_metric.compute()
        individual_map_values_50.append(eval_metric.map50) 
        individual_map_values_95.append(eval_metric.map50_95) # Store mAP@50 for the image
        
        # Reset the metric for the next image
        map_metric.reset()

    # Compute standard deviation of mAP values
    map_std_dev_50 = np.std(individual_map_values_50)
    map_std_dev_ =95 np.std(individual_map_values_95)
    print(f"Standard Deviation of mAP@50: {map_std_dev_50}")
    print(f"Standard Deviation of mAP@50:95: {map_std_dev_95}")

    # Compute overall metrics for the batch
    for idx, result in enumerate(results):
        annotations = annotations_list[idx]
        map_metric.update(result, [annotations])
        precision_metric.update(result, [annotations])
        recall_metric.update(result, [annotations])

    eval_metric = map_metric.compute()
    pre = precision_metric.compute()
    rec = recall_metric.compute()


    print(f"Precision: {pre.precision_at_50}")
    print(f"Recall: {rec.recall_at_50}")
    print(f"Mean Average Precision for batch: {eval_metric.map50}")
