import os
import json
import logging

# Setup logging for debugging pysolotools
logging.basicConfig(level=logging.DEBUG)

def check_metadata(folder_path):
    print("\n[INFO] Checking folder contents...")
    files = os.listdir(folder_path)
    print(f"All files in folder: {files}")

    # Look for metadata files
    metadata_files = [f for f in files if f.endswith(".metadata.json")]
    print(f"Found metadata files: {metadata_files}")
    print(f"Number of metadata files: {len(metadata_files)}")

    # Verify the metadata file is valid JSON
    if metadata_files:
        metadata_path = os.path.join(folder_path, metadata_files[0])
        print(f"\n[INFO] Trying to load metadata from: {metadata_path}")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print("[INFO] Metadata loaded successfully!")
                print(f"Metadata (first 3 lines): {str(metadata)[:300]}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to decode JSON: {e}")
    else:
        print("[ERROR] No valid metadata files found!")

def check_file_permissions(file_path):
    print(f"\n[INFO] Checking file permissions for {file_path}...")
    try:
        with open(file_path, 'r'):
            print(f"[INFO] File {file_path} is accessible.")
    except PermissionError:
        print(f"[ERROR] Permission denied for {file_path}.")
    except Exception as e:
        print(f"[ERROR] Unable to access file: {e}")

# Main script execution
folder_path = r"C:\Users\rebec\Downloads\Default\solo_2"
check_metadata(folder_path)

# Check file permissions for the metadata file (if it exists)
metadata_file_path = r"C:\Users\rebec\Downloads\Default\solo_2\sequence.0.metadata.json"
check_file_permissions(metadata_file_path)

# Try running the Solo class to see if we get any additional debugging output
try:
    from pysolotools.consumers import Solo
    print("\n[INFO] Trying to initialize Solo...")
    solo = Solo(data_path = folder_path, metadata_file_path = metadata_file_path)  # Initialize Solo with folder path
    print("[INFO] Solo initialized successfully!")
except Exception as e:
    print(f"[ERROR] Error initializing Solo: {e}")
