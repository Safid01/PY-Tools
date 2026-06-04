import os

def rename_files_sequentially(folder_path, start_num=1):
    # Get all files (ignore subfolders)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()

    # First, rename all files to a temporary name to avoid conflicts
    for i, filename in enumerate(files):
        old_path = os.path.join(folder_path, filename)
        temp_name = f"__temp_{i}__{os.path.splitext(filename)[1]}"
        temp_path = os.path.join(folder_path, temp_name)
        os.rename(old_path, temp_path)

    # Now, rename them from temp names to final numeric names
    temp_files = [f for f in os.listdir(folder_path) if f.startswith("__temp_")]
    temp_files.sort()

    for i, filename in enumerate(temp_files, start=start_num):
        ext = os.path.splitext(filename)[1]
        new_name = f"{i}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

# Example usage:
folder = r"C:\Users\safid\Downloads\Learning-Backpack\This is RAW Kpola Dorja Khol"
start_number = 1
rename_files_sequentially(folder, start_number)