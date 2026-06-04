import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, pipeline, AutoModelForSequenceClassification, AutoTokenizer
from langdetect import detect
import warnings
warnings.filterwarnings("ignore")

# --------------------
# CONFIG (defaults)
# --------------------
DEFAULT_EXCEL = r"C:\Users\Bolod Manus\Downloads\Code\Memes_Ocr.xlsx"
DEFAULT_IMAGES = r"C:\Users\Bolod Manus\Downloads\Memes"
DEFAULT_OUT = r"C:\Users\Bolod Manus\Downloads\Code\Meme_annotations_multilingual_clean.xlsx"
LIMIT = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "openai/clip-vit-large-patch14"
XLM_R_NLI = "joeddav/xlm-roberta-large-xnli"  # multilingual NLI for zero-shot

# Performance / batch & fp16
BATCH_SIZE = 8               # safe for 12GB RTX 3060; increase carefully
SAVE_EVERY = 500             # autosave every N processed items (0 disables)
USE_FP16 = True              # enable half precision on GPU (recommended for 12GB)

# Fusion weights (tune for best accuracy)
WEIGHT_TEXT = 0.6
WEIGHT_IMAGE = 0.4
CONFIDENCE_THRESHOLD = 0.45

# --------------------
# Bilingual label prompts (English + Bangla)
# --------------------
LABEL_PROMPTS = {
    "Context_Type": {
        "Personal": ["Personal or relationship context", "ব্যক্তিগত বা সম্পর্ক ভিত্তিক বিষয়"],
        "Domestic": ["Domestic or household context", "গার্হস্থ্য বা পরিবারের বিষয়"],
        "Professional": ["Workplace or professional context", "পেশাগত বা কর্মক্ষেত্র বিষয়ে"],
        "Societal": ["Societal or cultural commentary", "সামাজিক বা সাংস্কৃতিক মন্তব্য"],
        "Political": ["Political or ideological context", "রাজনৈতিক বা নীতিগত বিষয়"],
        "National": ["Nationality or country comparison", "জাতীয় বা দেশের তুলনা"],
        "Entertainment": ["Entertainment, celebrities or media", "বিনোদন বা সেলিব্রিটি বিষয়"],
        "Humor": ["Humorous or joke-based content", "রসিক বা মজার কন্টেন্ট"]
    },
    "Image_Type": {
        "Screenshot": ["A screenshot of chat, tweet or post", "চ্যাট/টুইট/পোস্টের স্ক্রিনশট"],
        "Photo": ["A real photograph", "বাস্তব ছবির ফটো"],
        "Cartoon": ["A cartoon or illustration", "কার্টুন বা চিত্রকলা"],
        "Edited": ["An edited or photoshopped image", "সম্পাদিত বা ফটোশপ করা ইমেজ"],
        "Template": ["A common meme template", "একটি প্রচলিত মিম টেমপ্লেট"],
        "VideoFrame": ["A frame from a video or movie", "ভিডিও বা সিনেমার ফ্রেম"]
    },
    "Misogyny": {"Present": ["Contains misogyny or hostile statements towards women", "মহিলাদের বিরুদ্ধে বিদ্বেষ বা অপবাদ আছে"], "Absent": ["No misogyny", "মহিলা বিরোধী মন্তব্য নেই"]},
    "Objectification": {"Present": ["Objectifies or sexualizes women", "মহিলাদের যৌনভাবে বস্তু করে দেখানো হয়েছে"], "Absent": ["No objectification", "কোনো বস্তুবান্ধবী নেই"]},
    "Prejudice": {"Present": ["Contains stereotypes or prejudice", "ভেদাভেদ বা রূঢ় ধারণা আছে"], "Absent": ["No prejudice", "পূর্বাগ্রহ নেই"]},
    "Humiliation": {"Present": ["Humiliates, shames, or mocks someone", "অপমান বা উপহাস করে"], "Absent": ["No humiliation", "অপমান নেই"]},
    "Misogyny_Severity": {"None": ["No misogyny"], "Moderate": ["Moderate or subtle misogyny"], "Severe": ["Severe or explicit misogyny"]},
    "Sarcasm_Present": {"Sarcastic": ["Sarcastic or ironic tone", "বিদ্রূপাত্মক বা ব্যঙ্গাত্মক স্বর"], "Not Sarcastic": ["Not sarcastic", "বিদ্রূপ নয়"]}
}

LABEL_OPTIONS = {
    "Context_Type": list(LABEL_PROMPTS['Context_Type'].keys()),
    "Image_Type": list(LABEL_PROMPTS['Image_Type'].keys()),
    "Misogyny": ['Present', 'Absent'],
    "Objectification": ['Present', 'Absent'],
    "Prejudice": ['Present', 'Absent'],
    "Humiliation": ['Present', 'Absent'],
    "Misogyny_Severity": ['None', 'Moderate', 'Severe'],
    "Sarcasm_Present": ['Sarcastic', 'Not Sarcastic']
}

# --------------------
# Helpers
# --------------------

def safe_open_image(path):
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        return None

def truncate_text(text, max_tokens=75, max_chars=300):
    if not isinstance(text, str):
        return ""
    t = text.strip()
    words = t.split()
    if len(words) > max_tokens:
        t = ' '.join(words[:max_tokens])
    if len(t) > max_chars:
        t = t[:max_chars]
    return t

def detect_language_safe(text):
    try:
        if not isinstance(text, str) or text.strip()=="":
            return 'unknown'
        return detect(text)
    except Exception:
        return 'unknown'

# --------------------
# Model loaders (FP16 + GPU aware)
# --------------------

def load_models(device=DEVICE, use_fp16=USE_FP16):
    print('Loading CLIP and multilingual NLI model...')
    # CLIP
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model = clip_model.to(device)
    if use_fp16 and device == "cuda":
        clip_model = clip_model.half()

    # XLM-R NLI pipeline (tokenizer + model)
    tokenizer = AutoTokenizer.from_pretrained(XLM_R_NLI)
    nli_model = AutoModelForSequenceClassification.from_pretrained(XLM_R_NLI)
    nli_model = nli_model.to(device)
    if use_fp16 and device == "cuda":
        nli_model = nli_model.half()

    xlm_pipe = pipeline('zero-shot-classification', model=nli_model, tokenizer=tokenizer, device=0 if device=='cuda' else -1)
    return clip_model, clip_processor, xlm_pipe

# --------------------
# Build label embeddings for CLIP (store on device)
# --------------------

def build_label_embeddings(clip_model, clip_processor, device):
    clip_model.eval()
    label_embs = {}
    for group, opts in LABEL_PROMPTS.items():
        label_embs[group] = {}
        for label, paraphrases in opts.items():
            # paraphrases may be list - pass them all to processor
            inputs = clip_processor(text=paraphrases, return_tensors='pt', padding=True, truncation=True)
            # move only relevant tensors to device (input_ids/attention_mask)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                t_embs = clip_model.get_text_features(**inputs)
            mean_emb = t_embs.mean(dim=0, keepdim=True)
            norm_emb = F.normalize(mean_emb, p=2, dim=-1)
            # keep embeddings on device (and half if model is half)
            label_embs[group][label] = norm_emb.detach()
    return label_embs

# --------------------
# Get combined multimodal embeddings in batch (device-aware + fp16 safe)
# --------------------

def batch_get_combined_embs(clip_model, clip_processor, device, image_paths, texts):
    """
    Returns combined embeddings tensor on device of shape (batch, d).
    If an image is missing, combined[i] = text embedding.
    """
    # Prepare safe texts
    texts_short = [truncate_text(t) for t in texts]

    # Text inputs (batch)
    text_inputs = clip_processor(text=texts_short, return_tensors='pt', padding=True, truncation=True, max_length=77)
    # Move text inputs to device (input_ids, attention_mask)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # Compute text features
    with torch.no_grad():
        txt_embs = clip_model.get_text_features(**text_inputs)
    txt_embs = F.normalize(txt_embs, p=2, dim=-1)  # (batch, d)

    # Images: build list of PIL images; track missing
    images = []
    image_mask = []
    for p in image_paths:
        img = safe_open_image(p)
        if img is None:
            # placeholder image (neutral)
            images.append(Image.new('RGB', (224,224), (127,127,127)))
            image_mask.append(False)
        else:
            images.append(img)
            image_mask.append(True)

    image_inputs = clip_processor(images=images, return_tensors='pt', padding=True)
    # Move pixel_values to device and cast to fp16 if model is half
    # Keep other inputs (none for images) unchanged
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    # If model is half precision, convert pixel_values to half
    if hasattr(clip_model, "dtype") and clip_model.dtype == torch.float16:
        if 'pixel_values' in image_inputs:
            image_inputs['pixel_values'] = image_inputs['pixel_values'].half()

    with torch.no_grad():
        img_embs = clip_model.get_image_features(**image_inputs)
    img_embs = F.normalize(img_embs, p=2, dim=-1)

    # Combine per-item (weighted)
    combined = F.normalize(WEIGHT_TEXT * txt_embs + WEIGHT_IMAGE * img_embs, p=2, dim=-1)

    # For items without real image, fallback to text-only
    for i, has_img in enumerate(image_mask):
        if not has_img:
            combined[i] = txt_embs[i]

    return combined  # tensor (batch, d) on device

# --------------------
# CLIP-based label scoring (single item) - device safe
# --------------------

def clip_scores_for_group_single(combined, embs):
    """
    combined: 1-D tensor on device (d,) or (1,d)
    embs: dict label -> tensor (1,d) stored on device
    returns dict label->prob (numpy)
    """
    # ensure combined is 1-d tensor on same device as embs
    if combined.dim() == 2 and combined.shape[0] == 1:
        combined = combined.squeeze(0)

    device = combined.device

    # Stack embeddings and move to same device (they are already on device from build step)
    all_embs = torch.stack([v.squeeze(0).to(device) for v in embs.values()])  # (num_labels, d)

    # Compute cosine similarity
    # expand combined to (num_labels, d) or use cosine_similarity with dim=1
    combined_exp = combined.unsqueeze(0).repeat(all_embs.shape[0], 1)  # (num_labels, d)
    sims = F.cosine_similarity(combined_exp, all_embs, dim=1)  # (num_labels,)
    probs = F.softmax(sims / 0.07, dim=0)

    # Map back to labels in same order
    labels = list(embs.keys())
    return {labels[i]: float(probs[i].detach().cpu().item()) for i in range(len(labels))}

# --------------------
# XLM-R (text) signals via zero-shot (per-text)
# --------------------

def xlm_text_scores(xlm_pipe, text, candidate_labels):
    if not isinstance(text, str) or text.strip() == '':
        return {lbl: 0.0 for lbl in candidate_labels}
    try:
        out = xlm_pipe(text, candidate_labels=candidate_labels, multi_label=True)
        scores = {lbl: 0.0 for lbl in candidate_labels}
        for lab, sc in zip(out['labels'], out['scores']):
            scores[lab] = float(sc)
        return scores
    except Exception:
        return {lbl: 0.0 for lbl in candidate_labels}

# --------------------
# Keyword heuristics for image/context fallback
# --------------------

IMAGE_KEYWORDS = {
    "Screenshot": ["screenshot", "screen shot", "tweet", "fb", "facebook", "instagram", "ig", "snap", "whatsapp", "telegram"],
    "Template": ["template", "me when", "tag someone", "starter pack", "x be like", "meme template"],
    "Photo": ["photo", "pic", "picture", "photograph", "selfie"],
    "Cartoon": ["cartoon", "comic", "illustration", "drawn", "sketch"],
    "Edited": ["photoshop", "edited", "edit", "deepfake", "ps", "manipulated"],
    "VideoFrame": ["video", "clip", "scene", "movie", "film", "frame"]
}

CONTEXT_KEYWORDS = {
    "Political": ["vote", "election", "minister", "president", "pm", "politic", "রাজ", "সরকার", "মন্ত্রি", "ভোট"],
    "Domestic": ["mom", "dad", "mother", "father", "home", "house", "family", "বউ", "মা", "বাবা", "পরিবার"],
    "Professional": ["boss", "office", "work", "job", "career", "পেশা", "কাজ"],
    "Societal": ["society", "culture", "community", "social", "সামাজিক", "সংস্কৃতি"],
    "National": ["country", "nation", "national", "জাতীয়", "দেশ"],
    "Entertainment": ["movie", "song", "celebrity", "actor", "singer", "বিনোদন", "সেলিব্রিটি"],
    "Humor": ["joke", "lol", "haha", "funny", "meme", "হাস্য", "মজার"],
    "Personal": ["relationship", "girlfriend", "boyfriend", "wife", "husband", "lover", "বন্ধু"]
}

def keyword_detect_image(text):
    if not isinstance(text, str):
        return None
    t = text.lower()
    for label, kws in IMAGE_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return label
    return None

def keyword_detect_context(text):
    if not isinstance(text, str):
        return None
    t = text.lower()
    for label, kws in CONTEXT_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return label
    return None

# --------------------
# Fusion & decision logic (improved)
# --------------------

def fusion_and_decide(clip_probs, xlm_probs, text, combined_emb):
    res = {}

    # ---------- Image_Type decision ----------
    img_group = clip_probs.get('Image_Type', {})

    # Handle dict or numpy array safely
    if isinstance(img_group, dict) and len(img_group) > 0:
        top_img, top_img_p = max(img_group.items(), key=lambda x: x[1])
    elif isinstance(img_group, np.ndarray) and img_group.size > 0:
        top_idx = int(np.argmax(img_group))
        label_list = list(LABEL_OPTIONS['Image_Type'])
        if 0 <= top_idx < len(label_list):
            top_img = label_list[top_idx]
            top_img_p = float(img_group[top_idx])
        else:
            top_img, top_img_p = ("Photo", 0.0)
    else:
        top_img, top_img_p = ("Photo", 0.0)

    kw_img = keyword_detect_image(text)
    if kw_img:
        chosen_image_type = kw_img if top_img_p < 0.7 else top_img
    else:
        chosen_image_type = top_img

    if combined_emb is None:
        chosen_image_type = kw_img if kw_img else ("Template" if "meme" in str(text).lower() else "Photo")

    res['Image_Type'] = chosen_image_type

    # ---------- Context_Type decision ----------
    ctx_group = clip_probs.get('Context_Type', {})

    if isinstance(ctx_group, dict) and len(ctx_group) > 0:
        ctx_top, ctx_top_p = max(ctx_group.items(), key=lambda x: x[1])
    elif isinstance(ctx_group, np.ndarray) and ctx_group.size > 0:
        top_idx = int(np.argmax(ctx_group))
        label_list = list(LABEL_OPTIONS['Context_Type'])
        if 0 <= top_idx < len(label_list):
            ctx_top = label_list[top_idx]
            ctx_top_p = float(ctx_group[top_idx])
        else:
            ctx_top, ctx_top_p = ("Humor", 0.0)
    else:
        ctx_top, ctx_top_p = ("Humor", 0.0)

    xlm_context_map = {
        'Political': 'Political',
        'Humor': 'Humor',
        'Domestic': 'Domestic',
        'Personal': 'Personal',
        'Professional': 'Professional',
        'Societal': 'Societal'
    }

    xlm_candidate_best = None
    xlm_candidate_score = 0.0
    for cand, mapped in xlm_context_map.items():
        sc = xlm_probs.get(cand, 0.0)
        if sc > xlm_candidate_score:
            xlm_candidate_score = sc
            xlm_candidate_best = mapped

    kw_ctx = keyword_detect_context(text)

    if ctx_top_p >= 0.45:
        chosen_context = ctx_top
    elif xlm_candidate_best and xlm_candidate_score >= 0.55:
        chosen_context = xlm_candidate_best
    elif kw_ctx:
        chosen_context = kw_ctx
    else:
        chosen_context = ctx_top

    if combined_emb is None and kw_ctx:
        chosen_context = kw_ctx

    res['Context_Type'] = chosen_context

    # ---------- Binary label fusion ----------
    for lab in ['Misogyny', 'Objectification', 'Prejudice', 'Humiliation']:
        clip_present = 0.0
        clip_group = clip_probs.get(lab, {})
        if isinstance(clip_group, dict):
            clip_present = clip_group.get('Present', 0.0)
        elif isinstance(clip_group, np.ndarray):
            arr = clip_group
            if arr.size > 0:
                clip_present = float(arr.flatten()[0])
            else:
                clip_present = 0.0

        xlm_val = xlm_probs.get(lab, 0.0)
        fused = WEIGHT_TEXT * xlm_val + WEIGHT_IMAGE * clip_present
        res[lab] = 1 if fused >= 0.5 else 0

    # ---------- Sarcasm ----------
    sarcasm_score = xlm_probs.get('Sarcasm', 0.0)
    res['Sarcasm_Present'] = 1 if sarcasm_score >= 0.5 else 0

    # ---------- Misogyny_Severity ----------
    avg = np.mean([
        (WEIGHT_TEXT * xlm_probs.get('Misogyny', 0.0) + WEIGHT_IMAGE * (clip_probs.get('Misogyny', {}).get('Present', 0.0) if isinstance(clip_probs.get('Misogyny', {}), dict) else 0.0)),
        (WEIGHT_TEXT * xlm_probs.get('Objectification', 0.0) + WEIGHT_IMAGE * (clip_probs.get('Objectification', {}).get('Present', 0.0) if isinstance(clip_probs.get('Objectification', {}), dict) else 0.0)),
        (WEIGHT_TEXT * xlm_probs.get('Humiliation', 0.0) + WEIGHT_IMAGE * (clip_probs.get('Humiliation', {}).get('Present', 0.0) if isinstance(clip_probs.get('Humiliation', {}), dict) else 0.0))
    ])
    if avg < 0.33:
        sev = 0
    elif avg < 0.66:
        sev = 1
    else:
        sev = 2

    res['Misogyny_Severity'] = int(sev)

    return res

# --------------------
# Main
# --------------------

def main(excel_path, image_dir, output_path, limit=LIMIT, batch_size=BATCH_SIZE, save_every=SAVE_EVERY):
    # load models
    clip_model, clip_processor, xlm_pipe = load_models(DEVICE, use_fp16=USE_FP16)
    label_embs = build_label_embeddings(clip_model, clip_processor, DEVICE)

    df = pd.read_excel(excel_path)
    if 'image_name' not in df.columns or 'extracted_text' not in df.columns:
        raise ValueError("Excel must contain 'image_name' and 'extracted_text' columns")

    if limit is not None and limit > 0:
        df = df.head(limit)

    n = len(df)
    results = []

    candidate_labels_for_xlm = [
        'Misogyny', 'Objectification', 'Prejudice', 'Humiliation',
        'Sarcasm', 'Humor', 'Political', 'Domestic', 'Personal', 'Professional', 'Societal'
    ]

    for start in tqdm(range(0, n, batch_size), desc="Batches"):
        end = min(start + batch_size, n)
        batch = df.iloc[start:end]
        image_paths = [os.path.join(image_dir, str(r['image_name'])) for _, r in batch.iterrows()]
        texts = [str(r['extracted_text']) if pd.notna(r['extracted_text']) else '' for _, r in batch.iterrows()]

        # compute combined embeddings in batch (on device)
        with torch.no_grad():
            combined_embs = batch_get_combined_embs(clip_model, clip_processor, DEVICE, image_paths, texts)
        # combined_embs is on device

        # For each item, compute clip_probs (using label_embs) and xlm_probs (text)
        for i, (_, row) in enumerate(batch.iterrows()):
            fname = str(row['image_name'])
            text = texts[i]
            image_path = image_paths[i]

            lang = detect_language_safe(text) if text.strip() else 'unknown'

            emb = combined_embs[i]  # keep on device

            # clip_probs per group (dicts)
            clip_probs = {}
            for group in LABEL_OPTIONS.keys():
                clip_probs[group] = clip_scores_for_group_single(emb, label_embs[group])

            # xlm text probs (per item)
            xlm_probs = xlm_text_scores(xlm_pipe, text, candidate_labels_for_xlm)

            fused = fusion_and_decide(clip_probs, xlm_probs, text, emb)

            out = {
                'Image_Name': fname,
                'Extracted_Text': text,
                'Language': lang,
                'Context_Type': fused['Context_Type'],
                'Image_Type': fused['Image_Type'],
                'Misogyny': int(fused['Misogyny']),
                'Objectification': int(fused['Objectification']),
                'Prejudice': int(fused['Prejudice']),
                'Humiliation': int(fused['Humiliation']),
                'Misogyny_Severity': int(fused['Misogyny_Severity']),
                'Sarcasm_Present': int(fused['Sarcasm_Present'])
            }
            results.append(out)

        # optional: free some GPU memory and autosave periodically
        if save_every and save_every > 0 and (end % save_every == 0 or end == n):
            partial_df = pd.DataFrame(results)
            keep_cols = [
                'Image_Name', 'Extracted_Text', 'Language', 'Context_Type', 'Image_Type',
                'Misogyny', 'Objectification', 'Prejudice', 'Humiliation', 'Misogyny_Severity', 'Sarcasm_Present'
            ]
            partial_df = partial_df[[c for c in keep_cols if c in partial_df.columns]]
            partial_path = f"{os.path.splitext(output_path)[0]}_part_{end}.xlsx"
            partial_df.to_excel(partial_path, index=False)
            print(f"✅ Autosaved checkpoint up to {end} rows -> {partial_path}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    out_df = pd.DataFrame(results)
    keep_cols = [
        'Image_Name', 'Extracted_Text', 'Language', 'Context_Type', 'Image_Type',
        'Misogyny', 'Objectification', 'Prejudice', 'Humiliation', 'Misogyny_Severity', 'Sarcasm_Present'
    ]
    out_df = out_df[[c for c in keep_cols if c in out_df.columns]]
    out_df.to_excel(output_path, index=False)
    print('Saved:', output_path)

# --------------------
# CLI
# --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel', type=str, default=DEFAULT_EXCEL, help='Path to input Excel (must have image_name and extracted_text)')
    parser.add_argument('--images', type=str, default=DEFAULT_IMAGES, help='Folder containing images')
    parser.add_argument('--out', type=str, default=DEFAULT_OUT, help='Output Excel path')
    parser.add_argument('--limit', type=int, default=LIMIT, help='Limit number of rows (0 for all)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for CLIP embedding extraction')
    parser.add_argument('--save_every', type=int, default=SAVE_EVERY, help='Autosave every N rows (0 to disable)')
    args = parser.parse_args()

    # if limit==0 => process all
    limit = None if (args.limit == 0 or args.limit is None) else args.limit
    main(args.excel, args.images, args.out, limit=limit, batch_size=args.batch_size, save_every=args.save_every)
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# --- CONFIGURATION ---
image_folder = r"C:\Users\safid\Downloads\Memes"
excel_path = r"C:\Users\safid\Downloads\Memes\Meme_annotations_Binar.xlsx"

# --- Load Excel ---
df = pd.read_excel(excel_path)

required_columns = [
    "Image_Name", "Misogyny", "Objectification", "Prejudice",
    "Humiliation", "Misogyny_Severity", "Sarcasm_Present"
]

for col in required_columns:
    if col not in df.columns:
        df[col] = None

df = df.sort_values(
    by="Image_Name",
    key=lambda x: x.astype(str).str.extract(r"(\d+)")[0].astype(float)
).reset_index(drop=True)

mask_unannotated = df[[
    "Misogyny", "Objectification", "Prejudice",
    "Humiliation", "Misogyny_Severity", "Sarcasm_Present"
]].isna().any(axis=1)

rows_to_annotate = df[mask_unannotated].index.tolist()
if not rows_to_annotate:
    print("🎉 All images are already annotated!")
    exit()

current_index = 0

# --- GUI SETUP ---
root = tk.Tk()
root.title("Meme Annotation Tool")
root.geometry("900x700")
root.configure(bg="#1e1e1e")

image_label = tk.Label(root, bg="#1e1e1e")
image_label.pack(pady=20)

filename_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="white", bg="#1e1e1e")
filename_label.pack(pady=5)

entry_values = {}
field_widgets = {}   # store frames for highlight
fields = [
    ("Misogyny", [0, 1]),
    ("Objectification", [0, 1]),
    ("Prejudice", [0, 1]),
    ("Humiliation", [0, 1]),
    ("Misogyny_Severity", [0, 1, 2]),
    ("Sarcasm_Present", [0, 1])
]

frame_buttons = tk.Frame(root, bg="#1e1e1e")
frame_buttons.pack(pady=10)

field_order = []
current_field_idx = 0

# Create UI fields
for field, values in fields:
    subframe = tk.Frame(frame_buttons, bg="#1e1e1e")
    subframe.pack(pady=8, fill="x")
    
    label = tk.Label(subframe, text=field, fg="white", bg="#1e1e1e", font=("Arial", 14))
    label.pack(side="left", padx=10)

    var = tk.IntVar(value=-1)
    entry_values[field] = var
    field_order.append(field)
    field_widgets[field] = (subframe, label)

    for v in values:
        tk.Radiobutton(
            subframe,
            text=str(v),
            variable=var,
            value=v,
            indicatoron=False,
            width=3,
            font=("Arial", 11),
            bg="#2d2d2d",
            fg="white",
            selectcolor="#3b82f6",
            relief="ridge",
        ).pack(side="left", padx=4)

# Highlight logic
def update_highlight():
    for f in field_order:
        frame, label = field_widgets[f]
        frame.configure(bg="#1e1e1e")
        label.configure(bg="#1e1e1e", fg="white")

    current_field = field_order[current_field_idx]
    frame, label = field_widgets[current_field]
    frame.configure(bg="#3b82f6")
    label.configure(bg="#3b82f6", fg="black")

# Save & Next
def save_and_next():
    global current_index

    idx = rows_to_annotate[current_index]
    for f in entry_values:
        if entry_values[f].get() == -1:
            messagebox.showwarning("Missing Value", f"Please select a value for {f}")
            return

    for f in entry_values:
        df.at[idx, f] = entry_values[f].get()

    df.to_excel(excel_path, index=False)
    print(f"Saved: {df.at[idx, 'Image_Name']}")

    current_index += 1
    if current_index >= len(rows_to_annotate):
        messagebox.showinfo("Done", "🎉 All images annotated!")
        root.destroy()
        return

    load_image()

# Load image
def load_image():
    global current_field_idx
    idx = rows_to_annotate[current_index]
    image_name = str(df.at[idx, "Image_Name"])
    image_path = os.path.join(image_folder, image_name)

    img = Image.open(image_path)
    img.thumbnail((800, 500))
    tk_img = ImageTk.PhotoImage(img)
    image_label.configure(image=tk_img)
    image_label.image = tk_img
    filename_label.configure(text=image_name)

    for f in entry_values:
        entry_values[f].set(-1)

    current_field_idx = 0
    update_highlight()

# ------------------ ARROW KEY CONTROLS ------------------
def on_key(event):
    global current_field_idx

    key = event.keysym

    # Navigate fields
    if key == "Down":
        current_field_idx = (current_field_idx + 1) % len(field_order)
        update_highlight()
        return

    if key == "Up":
        current_field_idx = (current_field_idx - 1) % len(field_order)
        update_highlight()
        return

    # Change values
    current_field = field_order[current_field_idx]
    possible_values = [rb["value"] for rb in entry_values[current_field].trace_vinfo()] \
        if hasattr(entry_values[current_field], "trace_vinfo") else None

    values_list = fields[current_field_idx][1]
    current_value = entry_values[current_field].get()

    if key == "Right":
        if current_value == -1:
            entry_values[current_field].set(values_list[0])
        else:
            idx = values_list.index(current_value)
            entry_values[current_field].set(values_list[(idx + 1) % len(values_list)])
        return

    if key == "Left":
        if current_value == -1:
            entry_values[current_field].set(values_list[-1])
        else:
            idx = values_list.index(current_value)
            entry_values[current_field].set(values_list[(idx - 1) % len(values_list)])
        return

    # Next
    if key == "Return":
        save_and_next()

root.bind("<Key>", on_key)
# --------------------------------------------------------

btn_next = tk.Button(
    root, text="Save & Next ➡️", command=save_and_next,
    font=("Arial", 14, "bold"), bg="#3b82f6", fg="white",
    width=20, height=2
)
btn_next.pack(pady=20)

load_image()
root.mainloop()