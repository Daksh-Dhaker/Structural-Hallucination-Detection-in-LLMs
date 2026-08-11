import os
import shutil

parent_dir = "Non_hallucinating_circuits/code_dataset"
output_dir = "Non_hallucinating_circuits_bootstrapped/code_dataset"

# Create output folder if not exists
os.makedirs(output_dir, exist_ok=True)

# List all subfolders inside parent_dir
subfolders = [f for f in os.listdir(parent_dir)
              if os.path.isdir(os.path.join(parent_dir, f))]

for folder in subfolders:
    folder_path = os.path.join(parent_dir, folder)

    # Extract the batch number (e.g., "batch_1")
    batch_suffix = folder.replace("code_dataset_", "")  # -> batch_1

    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        if os.path.isfile(old_path):
            # Split base name and extension
            name, ext = os.path.splitext(filename)

            # New name for merged folder
            new_filename = f"{name}_{batch_suffix}{ext}"

            new_path = os.path.join(output_dir, new_filename)

            # Copy (not move) to keep original files safe
            shutil.copy2(old_path, new_path)

print("All files copied successfully into:", output_dir)
