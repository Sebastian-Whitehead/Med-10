from map import evaluate_batch, evaluate_single, update_batch_id

# Evaluate a batch of images
batch_folder = "batch_images/test_set"
evaluate_batch(batch_folder)

# Uncomment the following lines to evaluate a single image
#single_image_path = "batch_images/test_set/images/0a20baf628fa9be7_jpg.rf.6ce55e30ca2cac3c5e01b3dacedc7b11.jpg"
#single_annotation_path = "batch_images/test_set/labels/0a20baf628fa9be7_jpg.rf.6ce55e30ca2cac3c5e01b3dacedc7b11.txt"
#evaluate_single(single_image_path, single_annotation_path, true)