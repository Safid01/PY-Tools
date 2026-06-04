import os
import random
import time
import pywintypes
import win32file
import win32con

def set_creation_time_windows(file_path, timestamp):
    # Convert timestamp to Windows FILETIME format
    wintime = pywintypes.Time(timestamp)

    handle = win32file.CreateFile(
        file_path,
        win32con.GENERIC_WRITE,
        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
        None,
        win32con.OPEN_EXISTING,
        0,
        None
    )

    # Set creation, access, and modified times
    win32file.SetFileTime(handle, wintime, wintime, wintime)
    handle.close()


def randomize_rename_and_update_all_times(folder_path, start_num=1):
    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f))]

    random.shuffle(files)

    # Temporary rename
    for i, filename in enumerate(files):
        old_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1]
        temp_name = f"__temp_{i}__{ext}"
        temp_path = os.path.join(folder_path, temp_name)
        os.rename(old_path, temp_path)

    temp_files = [f for f in os.listdir(folder_path) if f.startswith("__temp_")]
    temp_files.sort()

    current_time = time.time()

    for i, filename in enumerate(temp_files, start=start_num):
        ext = os.path.splitext(filename)[1]
        new_name = f"{i}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

        # Update modified & access time
        os.utime(new_path, (current_time, current_time))

        # Update creation time (Windows only)
        set_creation_time_windows(new_path, current_time)

        print(f"Renamed & Fully Updated: {filename} → {new_name}")

# Example usage
folder = r"C:\Users\safid\Downloads\Memes 3"
start_number = 1
randomize_rename_and_update_all_times(folder, start_number)