import argparse
from test_scripts.test_batch_8 import evaluate_batch_8
from test_scripts.test_batch_11 import evaluate_batch_11    
from test_scripts.test_single_8 import evaluate_single_8
from test_scripts.test_single_11 import evaluate_single_11

def parse_arguments():
    """Create and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run evaluation scripts based on input arguments.")
    parser.add_argument("version", choices=["8", "11"], help="Specify the version: '8' or '11'.")
    parser.add_argument("type", choices=["batch", "single"], help="Specify the type: 'batch' or 'single'.")
    return parser.parse_args()

def main(args, image=None, batch_folder=None):
    """Run the appropriate evaluation script based on parsed arguments."""
    if args.version == "8" and args.type == "batch":
        evaluate_batch_8(batch_path=batch_folder, log=True, std = True)
    elif args.version == "8" and args.type == "single":
        evaluate_single_8(image_name=image, show_results=True, log=False)
    elif args.version == "11" and args.type == "batch":
        evaluate_batch_11(batch_path=batch_folder, log=True, std = True)
    elif args.version == "11" and args.type == "single":
        evaluate_single_11(image_name=image, show_results=True, log=False, sali=False)
    else:
        print("Invalid combination of arguments.")

if __name__ == "__main__":
    image = "2dark5_jpg.rf.195ff3de514bdb9771d4100d2e00f7ce.jpg"
    #image = "09a245b6-498e-4169-9acd-73f09ac4e04b_jpeg_jpg.rf.4a1a07dc1571709c7e7995f48ac6804e.jpg"
    batch_folder = "batch_images/test_set"
    batch_folder = "data/3"

    image = "551cb335-dfff-4516-9070-b2a3fadfe28b_jpeg_jpg.rf.2188f2c02fca6fa14c6f851f89c1ac5e.jpg"
    image = "camera_103.png"
    args = parse_arguments()
    main(args, image=image, batch_folder=batch_folder)

