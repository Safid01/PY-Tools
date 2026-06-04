import concurrent.futures
import os
import re
import threading
import warnings

import pandas as pd
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

# ---------------- CONFIGURATION ----------------
IMAGE_FOLDER = r"C:\Users\safid\Downloads\Memes 2"
OUTPUT_FILE = "ocr_results.xlsx"
LANGUAGES = "ben+eng"
MAX_WORKERS = 6
SAVE_INTERVAL = 1

# Optional: Hide Tesseract warnings
warnings.filterwarnings("ignore")

# ---------------- SETUP TESSERACT PATH ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- THREAD LOCK ----------------
lock = threading.Lock()


def natural_sort_key(filename):
    parts = re.split(r"(\d+)", filename.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def clean_text(text):
    return re.sub(r"\s+\n", "\n", text).strip()


def build_ocr_variants(img):
    grayscale = ImageOps.grayscale(img)
    upscaled = grayscale.resize(
        (max(1, grayscale.width * 2), max(1, grayscale.height * 2)),
        Image.Resampling.LANCZOS,
    )
    autocontrast = ImageOps.autocontrast(upscaled)
    sharpened = autocontrast.filter(ImageFilter.SHARPEN)
    thresholded = sharpened.point(lambda x: 0 if x < 160 else 255, "1").convert("L")

    return [img, grayscale, upscaled, autocontrast, sharpened, thresholded]


def extract_text_from_image(img):
    best_text = ""
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 3",
    ]

    for variant in build_ocr_variants(img):
        for config in configs:
            text = clean_text(
                pytesseract.image_to_string(variant, lang=LANGUAGES, config=config)
            )
            if len(text) > len(best_text):
                best_text = text

    return best_text if best_text else "No text detected"


# ---------------- LOAD EXISTING RESULTS ----------------
if os.path.exists(OUTPUT_FILE):
    try:
        processed_results_df = pd.read_excel(OUTPUT_FILE)
    except Exception:
        processed_results_df = pd.DataFrame(columns=["image_name", "extracted_text"])
else:
    processed_results_df = pd.DataFrame(columns=["image_name", "extracted_text"])

if "serial" in processed_results_df.columns:
    processed_results_df = processed_results_df.drop(columns=["serial"])

processed_results_df = processed_results_df.reindex(
    columns=["image_name", "extracted_text"], fill_value=""
)
processed_names = set(processed_results_df["image_name"].tolist())
print(f"Resuming from previous progress. Already processed: {len(processed_names)} images.")


def save_results():
    global processed_results_df, processed_names

    if processed_results_df.empty:
        output_df = pd.DataFrame(columns=["serial", "image_name", "extracted_text"])
    else:
        processed_results_df = (
            processed_results_df
            .drop_duplicates(subset=["image_name"], keep="last")
            .sort_values("image_name", key=lambda col: col.map(natural_sort_key))
            .reset_index(drop=True)
        )
        processed_names = set(processed_results_df["image_name"].tolist())
        output_df = processed_results_df.copy()
        output_df.insert(0, "serial", range(1, len(output_df) + 1))

    output_df.to_excel(OUTPUT_FILE, index=False)


def process_image(image_path):
    file_name = os.path.basename(image_path)
    if file_name in processed_names:
        return None

    try:
        with Image.open(image_path) as img:
            extracted_text = extract_text_from_image(img)
    except Exception as e:
        extracted_text = f"Error: {str(e)}"

    return {"image_name": file_name, "extracted_text": extracted_text}


# ---------------- MAIN EXECUTION ----------------
def main():
    allowed_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    try:
        files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(allowed_exts)]
    except FileNotFoundError:
        print(f"Folder not found: {IMAGE_FOLDER}")
        return

    files_sorted = sorted(files, key=natural_sort_key)
    all_images = [os.path.join(IMAGE_FOLDER, f) for f in files_sorted]

    to_process = [img for img in all_images if os.path.basename(img) not in processed_names]
    print(f"Found {len(all_images)} total images.")
    print(f"Processing remaining {len(to_process)} images...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for index, result in enumerate(
            tqdm(executor.map(process_image, to_process), total=len(to_process)),
            start=1,
        ):
            if result is None:
                continue

            with lock:
                processed_results_df.loc[len(processed_results_df)] = result
                processed_names.add(result["image_name"])
                if index % SAVE_INTERVAL == 0:
                    save_results()

    save_results()
    print(f"\nOCR completed. Total images processed: {len(processed_results_df)}")
    print(f"Results saved to: {OUTPUT_FILE}")


# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    main()
