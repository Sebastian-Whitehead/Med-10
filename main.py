from map import evaluate_batch_8, evaluate_single, update_batch_id
from map11 import evaluate_batch_11
from sali import evaluate_single

# Evaluate a batch of images
batch_folder = "batch_images/test_set"
#evaluate_batch_8(batch_folder)

#Precision: 0.8864187348978007
#Recall: 0.8950617283950617
#Mean Average Precision for batch: 0.8772768020901894  


evaluate_batch_11(batch_folder)
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
#print(os.getcwd())
# Uncomment the following lines to evaluate a single image
#single_image_path = "batch_images/test_set/images/0a20baf628fa9be7_jpg.rf.6ce55e30ca2cac3c5e01b3dacedc7b11.jpg"
#single_annotation_path = "batch_images/test_set/labels/0a20baf628fa9be7_jpg.rf.6ce55e30ca2cac3c5e01b3dacedc7b11.txt"

image = "2dark5_jpg.rf.195ff3de514bdb9771d4100d2e00f7ce.jpg"
#image = "000000253834_jpg.rf.c0bb81a9ec8fa4c6a3eecc34235e50f0.jpg"
#image = "uc-352-_jpg.rf.cc0da01a3bd0eac428567cc344cf0dac.jpg"


#image = "2dark5_jpg.rf.195ff3de514bdb9771d4100d2e00f7ce.jpg"

base_name, extension = image.rsplit(".", 1)

txt_image = f"{base_name}.txt"

single_image_path = os.path.join("batch_images", "test_set", "images", image)
single_annotation_path = os.path.join("batch_images", "test_set", "labels", txt_image)

print(single_annotation_path)
print(single_image_path)

#evaluate_single(single_image_path, single_annotation_path, True, False)