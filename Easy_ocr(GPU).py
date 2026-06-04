import easyocr
import os
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import threading
import warnings

# ---------------- CONFIGURATION ----------------
IMAGE_FOLDER = r"C:\Users\safid\Downloads\Memes 3"  # 🔹 Change this path
OUTPUT_FILE = "Memes_3.xlsx" # Output Excel file
LANGUAGES = ['en', 'bn']  # English + Bangla
MAX_WORKERS = 6  # Adjust threads (based on CPU cores)
SAVE_INTERVAL = 1 # Save after every image processed

# ---------------- SILENCE WARNINGS ----------------
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------- INITIALIZE READER ----------------
print("🔍 Initializing EasyOCR reader (English + Bangla)...")
reader = easyocr.Reader(['en', 'bn'], gpu=True)
print("🧩 EasyOCR backend:", "GPU" if reader.device == 'cuda' else "CPU")

# ---------------- THREAD LOCK ----------------
lock = threading.Lock()

# ---------------- LOAD EXISTING RESULTS ----------------
if os.path.exists(OUTPUT_FILE):
    try:
        processed_results_df = pd.read_excel(OUTPUT_FILE)
    except Exception:
        processed_results_df = pd.DataFrame(columns=["image_name", "extracted_text"])
else:
    processed_results_df = pd.DataFrame(columns=["image_name", "extracted_text"])

processed_names = set(processed_results_df['image_name'].tolist())
print(f"🔁 Resuming from previous progress. Already processed: {len(processed_names)} images.")

# ---------------- OCR FUNCTION ----------------
def process_image(image_path):
    file_name = os.path.basename(image_path)
    if file_name in processed_names:
        return None, None, None  # Skip already processed
    
    try:
        results = reader.readtext(image_path, detail=1, paragraph=True)
        extracted_text = " ".join([res[1] for res in results])
    except Exception as e:
        extracted_text = f"Error: {str(e)}"

    return image_path, file_name, extracted_text


def natural_sort_key(value):
    name = os.path.splitext(os.path.basename(str(value)))[0]
    return (0, int(name)) if name.isdigit() else (1, name.lower())


# ---------------- MAIN EXECUTION ----------------
def main():
    all_images = [
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    # ✅ Natural sort: 1.jpg, 2.jpg, 10.jpg
    all_images.sort(key=natural_sort_key)

    to_process = [img for img in all_images if os.path.basename(img) not in processed_names]
    print(f"📂 Found {len(all_images)} total images.")
    print(f"🚀 Processing remaining {len(to_process)} images...\n")

    new_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_image, img): img for img in to_process}
        with tqdm(total=len(to_process), desc="🧠 OCR Progress", unit="image") as pbar:
            for future in concurrent.futures.as_completed(futures):
                img_path, file_name, extracted_text = future.result()
                if file_name is None:
                    continue

                # Show current filename on the progress bar
                pbar.set_postfix_str(f"📄 {file_name[:40]}")
                pbar.update(1)

                with lock:
                    new_results.append((file_name, extracted_text))
                    if len(new_results) % SAVE_INTERVAL == 0:
                        temp_df = pd.DataFrame(new_results, columns=["image_name", "extracted_text"])
                        combined = pd.concat([processed_results_df, temp_df], ignore_index=True)
                        combined.sort_values(
                            "image_name",
                            key=lambda s: s.map(natural_sort_key),
                            inplace=True
                        )
                        combined.to_excel(OUTPUT_FILE, index=False)

    # 🔹 Final save
    final_df = pd.concat([processed_results_df, pd.DataFrame(new_results, columns=["image_name", "extracted_text"])],
                         ignore_index=True)
    final_df.sort_values(
        "image_name",
        key=lambda s: s.map(natural_sort_key),
        inplace=True
    )
    final_df.to_excel(OUTPUT_FILE, index=False)

    print(f"\n✅ OCR completed. Total images processed: {len(final_df)}")
    print(f"📁 Results saved to: {OUTPUT_FILE}")


# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    main()