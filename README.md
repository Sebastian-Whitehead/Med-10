# MED 10 Thesis - Python Implementation

This project includes the python files used for our master thesis, including training and running YOLO models, processing results and plotting them.

## Setup
- Create an environment of your choice.
- Install dependencies from *"requirements.txt"*
- Use terminal to excecute scripts

## Structure
- *main.py* uses two arguments: *single* or *batch* - Insert folder or image placement in the script under *"image_name"* or *"batch_folder"*. To view images for *single*, include *"verbose"* as an argument. 
- *stats.py* calculates std for mAP, precision and recall, and adds them into the results file.
- *"plot_scripts"* folder contains two scripts for making plots. *plots.py* has 3 methods: *normal* (no argument), *big* and *combined*. *"table.py"* takes a results file and converts it into a *latex* table.
- *train.py* finetunes a model on a selected dataset. 
- *"utils"* folder contains a *logger.py* script to log results, and *utilities.py* with general utility functions used in various scripts. 
- *main_old.py* has functions for running both v8 and v11 models for both batch and single. 