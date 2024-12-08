import os

def create_data_directory(base_dir="data"):
    """Creates a directory structure for dataset organization."""
    subdirs = ["train/images", "train/annotations", "val/images", "val/annotations"]

    try:
        for subdir in subdirs:
            path = os.path.join(base_dir, subdir)
            os.makedirs(path, exist_ok=True)
        print(f"Data directory structure created successfully under '{base_dir}'")
    except Exception as e:
        print(f"An error occurred while creating the directory structure: {e}")

# Run the function to create the directory structure
create_data_directory()
