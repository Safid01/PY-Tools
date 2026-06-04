import os
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# --- CONFIGURATION ---
image_folder = r"C:\Users\safid\Downloads\Memes"   # ← change this
excel_path = r"C:\Users\safid\Downloads\Memes\Meme_annotations_context.xlsx"

# --- Load Excel ---
df = pd.read_excel(excel_path)

# Ensure columns exist
if "Context_Type" not in df.columns:
    df["Context_Type"] = None
if "Last_Annotated_Index" not in df.columns:
    df["Last_Annotated_Index"] = None

# --- Sort numerically by filename ---
df = df.sort_values(
    by="Image_Name",
    key=lambda x: x.astype(str).str.extract(r"(\d+)")[0].astype(float)
).reset_index(drop=True)

# --- Determine Resume Point ---
resume_index = 0
if df["Last_Annotated_Index"].notna().any():
    last_saved = int(df["Last_Annotated_Index"].dropna().iloc[-1])
    resume_index = min(last_saved + 1, len(df) - 1)
else:
    unannotated = df.index[df["Context_Type"].isna()].tolist()
    resume_index = unannotated[0] if unannotated else 0

rows_to_annotate = df.index.tolist()
current_index = resume_index
current_img = None
current_tk_img = None

# --- GUI SETUP ---
root = tk.Tk()
root.title("Context Type Annotator")
root.configure(bg="#1e1e1e")
root.minsize(900, 750)
root.rowconfigure(1, weight=1)
root.columnconfigure(0, weight=1)

filename_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="white", bg="#1e1e1e")
filename_label.grid(row=0, column=0, pady=10)

image_label = tk.Label(root, bg="#1e1e1e")
image_label.grid(row=1, column=0, sticky="nsew")

# Label to show previous selection text
prev_label = tk.Label(root, text="", font=("Arial", 13), fg="#00ffaa", bg="#1e1e1e")
prev_label.grid(row=2, column=0, pady=(5, 10))

frame_buttons = tk.Frame(root, bg="#1e1e1e")
frame_buttons.grid(row=3, column=0, pady=10)

# --- Navigation Buttons Frame ---
frame_nav = tk.Frame(root, bg="#1e1e1e")
frame_nav.grid(row=4, column=0, pady=10)

context_var = tk.StringVar(value="")

context_types = [
    "Personal", "Domestic", "Professional", "Societal",
    "Political", "National", "Entertainment", "Humor"
]

shortcut_map = {
    "1": "Personal", "2": "Domestic", "3": "Professional", "4": "Societal",
    "5": "Political", "6": "National", "7": "Entertainment", "8": "Humor"
}

# Create buttons (manual selection only)
for idx, c in enumerate(context_types, start=1):
    btn = tk.Radiobutton(
        frame_buttons,
        text=f"{c} ({idx})",
        variable=context_var,
        value=c,
        indicatoron=False,
        width=14,
        font=("Arial", 12),
        bg="#2d2d2d",
        fg="white",
        selectcolor="#3b82f6",
        relief="ridge"
    )
    btn.grid(row=0, column=idx-1, padx=5, pady=5, sticky="ew")

frame_buttons.grid_columnconfigure(tuple(range(len(context_types))), weight=1)


# --- FUNCTIONS ---
def show_image(resize_only=False):
    """Display or resize current image."""
    global current_img, current_tk_img

    if not resize_only:
        idx = rows_to_annotate[current_index]
        image_name = str(df.at[idx, "Image_Name"])
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image not found: {image_name}")
            return

        current_img = Image.open(image_path)
        filename_label.configure(
            text=f"{image_name} ({current_index + 1}/{len(rows_to_annotate)})"
        )

        # --- Show previous annotation text ---
        prev_val = df.at[idx, "Context_Type"]
        if pd.notna(prev_val):
            prev_label.configure(text=f"Previously selected: {prev_val}")
            context_var.set(prev_val)
        else:
            prev_label.configure(text="No previous annotation.")
            context_var.set("")

    if current_img:
        w = root.winfo_width() - 100
        h = root.winfo_height() - 350
        resized = current_img.copy()
        resized.thumbnail((w, h))
        current_tk_img = ImageTk.PhotoImage(resized)
        image_label.configure(image=current_tk_img)
        image_label.image = current_tk_img


def save_annotation():
    """Save current annotation without moving automatically."""
    idx = rows_to_annotate[current_index]
    selected = context_var.get()

    if not selected:
        messagebox.showwarning("Missing Selection", "Please select a Context_Type before saving.")
        return False

    df.at[idx, "Context_Type"] = selected
    df["Last_Annotated_Index"] = None
    df.at[idx, "Last_Annotated_Index"] = idx
    df.to_excel(excel_path, index=False)
    print(f"✅ Saved: {df.at[idx, 'Image_Name']} → {selected}")
    return True


def go_next(event=None):
    """Move forward one image."""
    global current_index, current_img
    if not save_annotation():
        return
    if current_index < len(rows_to_annotate) - 1:
        current_index += 1
        current_img = None
        show_image()
    else:
        messagebox.showinfo("Done", "🎉 All images annotated!")
        root.destroy()


def go_previous(event=None):
    """Move backward one image."""
    global current_index, current_img
    if current_index > 0:
        current_index -= 1
        current_img = None
        show_image()
    else:
        messagebox.showinfo("Start", "You're already at the first image!")


def handle_shortcut(event):
    key = event.char
    if key in shortcut_map:
        context_var.set(shortcut_map[key])
        print(f"🔹 Selected via key {key}: {shortcut_map[key]}")
    elif event.keysym == "Return":
        go_next()


def on_resize(event):
    show_image(resize_only=True)


# --- Navigation Buttons ---
btn_prev = tk.Button(
    frame_nav, text="⬅️ Previous",
    font=("Arial", 13, "bold"), bg="#6366f1", fg="white",
    width=15, height=2, relief="ridge", command=go_previous
)
btn_prev.pack(side="left", padx=10)

btn_next = tk.Button(
    frame_nav, text="Save & Next ➡️",
    font=("Arial", 13, "bold"), bg="#3b82f6", fg="white",
    width=15, height=2, relief="ridge", command=go_next
)
btn_next.pack(side="right", padx=10)


# --- Bindings ---
root.bind("<Key>", handle_shortcut)
root.bind("<Return>", go_next)
root.bind("<Left>", go_previous)
root.bind("<Right>", go_next)
root.bind("<Configure>", on_resize)

# --- Start ---
root.after(300, show_image)
root.mainloop()