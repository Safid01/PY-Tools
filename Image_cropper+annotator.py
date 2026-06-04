import cv2
import os
import pytesseract
import pandas as pd
import re
import matplotlib.pyplot as plt
import sys

# -------- CONFIG --------
INPUT_DIR = "images"
OUTPUT_DIR_TRUE = "output/true_news"
OUTPUT_DIR_FALSE = "output/false_news"

TRUE_XLSX = "true_news.xlsx"
FALSE_XLSX = "fake_news.xlsx"

# Tesseract OCR Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Create output directories
os.makedirs(OUTPUT_DIR_TRUE, exist_ok=True)
os.makedirs(OUTPUT_DIR_FALSE, exist_ok=True)

# Load existing excel files if they exist
true_df = pd.read_excel(TRUE_XLSX) if os.path.exists(TRUE_XLSX) else pd.DataFrame(columns=["ocr_text", "image_file"])
false_df = pd.read_excel(FALSE_XLSX) if os.path.exists(FALSE_XLSX) else pd.DataFrame(columns=["ocr_text", "image_file", "manipulation_type"])

# Mapping numbers to manipulation types
MANIPULATION_TYPES = {
    "1": "Disinformation",
    "2": "Nonsense",
    "3": "Defamation",
    "4": "Propaganda",
    "5": "Fabricated Claim"
}

# -------- FUNCTIONS --------
def numeric_sort_key(fname):
    """Sort filenames by their numeric part instead of alphabetically"""
    nums = re.findall(r'\d+', fname)
    return int(nums[0]) if nums else float('inf')

def interactive_crop(image_path, label_text="Select region"):
    """Open image fullscreen, crop interactively with 2 clicks.
       If window closed → stop program."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mng = plt.get_current_fig_manager()
    try:
        mng.full_screen_toggle()  # Works on most backends
    except Exception:
        try:
            mng.window.state('zoomed')  # TkAgg backend
        except Exception:
            pass

    plt.imshow(img_rgb)
    plt.title(f"{label_text}\n{os.path.basename(image_path)}")
    
    # ✅ wait indefinitely until 2 clicks or window close
    pts = plt.ginput(2, timeout=0)  
    plt.close()

    if len(pts) < 2:
        print("🛑 Program stopped by user (crop window closed).")
        sys.exit(0)  # Stop if window closed manually

    (x1, y1), (x2, y2) = [(int(x), int(y)) for x, y in pts]
    crop = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
    return crop

def run_ocr(image):
    """Run OCR on a cropped region"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="ben+hin+eng")
    return text.strip()

def save_true_entry(crop, ocr_text, fname):
    """Save true crop + update Excel"""
    save_path = os.path.join(OUTPUT_DIR_TRUE, fname)
    cv2.imwrite(save_path, crop)
    global true_df
    true_df.loc[len(true_df)] = [ocr_text, fname]
    true_df.to_excel(TRUE_XLSX, index=False)

def save_false_entry(crop, ocr_text, manipulation, fname):
    """Save false crop + update Excel"""
    save_path = os.path.join(OUTPUT_DIR_FALSE, fname)
    cv2.imwrite(save_path, crop)
    global false_df
    false_df.loc[len(false_df)] = [ocr_text, fname, manipulation]
    false_df.to_excel(FALSE_XLSX, index=False)

# -------- MAIN LOOP --------
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
files = sorted(files, key=numeric_sort_key)  # ✅ natural numeric sorting

# Track already processed images
processed_files = set(true_df["image_file"].tolist()) | set(false_df["image_file"].tolist())

for fname in files:
    if fname in processed_files:
        print(f"⏭ Skipping already processed: {fname}")
        continue

    fpath = os.path.join(INPUT_DIR, fname)
    print(f"\n📌 Processing: {fname}")

    # Step 1: Fake crop
    fake_crop = interactive_crop(fpath, label_text="Select FAKE claim region")
    ocr_text = run_ocr(fake_crop)

    print("\nChoose Manipulation Type:")
    for k, v in MANIPULATION_TYPES.items():
        print(f"{k} → {v}")
    choice = input("Enter number [1-5]: ").strip()
    manipulation = MANIPULATION_TYPES.get(choice, "Disinformation")

    save_false_entry(fake_crop, ocr_text, manipulation, fname)

    # Step 2: True crop
    true_crop = interactive_crop(fpath, label_text="Select TRUE claim region")
    ocr_text = run_ocr(true_crop)

    save_true_entry(true_crop, ocr_text, fname)

print("\n🎉 Done! Crops saved with original filenames + Excel files updated.")