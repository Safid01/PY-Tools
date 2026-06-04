import os
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# ------------------- CONFIG -------------------
image_folder = r"C:\Users\safid\Downloads\Memes"
excel_path = r"C:\Users\safid\Downloads\Memes\Meme_annotations_Binar.xlsx"

FIELDS = [
    ("Misogyny", [0, 1]),
    ("Misogyny_Severity", [0, 1, 2])
]

# ------------------- LOAD EXCEL -------------------
df = pd.read_excel(excel_path)

for name, _ in FIELDS:
    if name not in df.columns:
        df[name] = None

df = df.sort_values(
    by="Image_Name",
    key=lambda x: x.astype(str).str.extract(r"(\d+)")[0].astype(float)
).reset_index(drop=True)

# All rows where ANY of the two columns is unfilled
mask_unanno = df[[f[0] for f in FIELDS]].isna().any(axis=1)
rows_to_annotate = df[mask_unanno].index.tolist()

if not rows_to_annotate:
    print("🎉 All images annotated!")
    exit()

current_index = 0
selected_field = 0  # 0 = Misogyny, 1 = Severity

# ------------------- GUI -------------------
root = tk.Tk()
root.title("Misogyny + Misogyny Severity Annotator")
root.geometry("900x700")
root.configure(bg="#1e1e1e")

image_label = tk.Label(root, bg="#1e1e1e")
image_label.pack(pady=20)

filename_label = tk.Label(root, text="", fg="white", bg="#1e1e1e", font=("Arial", 14, "bold"))
filename_label.pack(pady=5)

# Field Frames + Variables
field_frames = []
field_vars = {}

fields_container = tk.Frame(root, bg="#1e1e1e")
fields_container.pack(pady=15)

for fname, values in FIELDS:
    frame = tk.Frame(fields_container, bg="#1e1e1e", highlightthickness=0)
    frame.pack(pady=10)

    tk.Label(frame, text=fname, fg="white", bg="#1e1e1e", font=("Arial", 12)).pack()

    var = tk.IntVar(value=-1)
    field_vars[fname] = (var, values)

    row_frame = tk.Frame(frame, bg="#1e1e1e")
    row_frame.pack()

    for v in values:
        tk.Radiobutton(
            row_frame,
            text=str(v),
            variable=var,
            value=v,
            indicatoron=False,
            width=4,
            font=("Arial", 12),
            bg="#333",
            fg="white",
            selectcolor="#3b82f6",
        ).pack(side="left", padx=4)

    field_frames.append(frame)

# ------------------- FUNCTIONS -------------------
def update_highlight():
    for i, frame in enumerate(field_frames):
        if i == selected_field:
            frame.config(highlightbackground="#3b82f6", highlightcolor="#3b82f6", highlightthickness=2)
        else:
            frame.config(highlightthickness=0)


def load_image():
    idx = rows_to_annotate[current_index]
    img_name = df.at[idx, "Image_Name"]
    img_path = os.path.join(image_folder, img_name)

    if not os.path.exists(img_path):
        messagebox.showerror("Missing Image", img_name)
        return

    img = Image.open(img_path)
    img.thumbnail((800, 500))
    tk_img = ImageTk.PhotoImage(img)
    image_label.configure(image=tk_img)
    image_label.image = tk_img

    filename_label.configure(text=img_name)

    # Reset values
    for fname, _ in FIELDS:
        field_vars[fname][0].set(-1)

    update_highlight()


def save_and_next():
    global current_index

    idx = rows_to_annotate[current_index]

    # All fields must be filled
    for fname, _ in FIELDS:
        if field_vars[fname][0].get() == -1:
            messagebox.showwarning("Missing Value", f"Select a value for {fname}")
            return

    for fname, _ in FIELDS:
        df.at[idx, fname] = field_vars[fname][0].get()

    df.to_excel(excel_path, index=False)

    current_index += 1
    if current_index >= len(rows_to_annotate):
        messagebox.showinfo("Done", "🎉 All annotations complete!")
        root.destroy()
        return

    load_image()


def on_key(event):
    global selected_field

    key = event.keysym

    # Up / Down navigation
    if key == "Up":
        selected_field = max(0, selected_field - 1)
        update_highlight()
        return

    if key == "Down":
        selected_field = min(len(FIELDS) - 1, selected_field + 1)
        update_highlight()
        return

    # Left / Right changes
    fname, values = FIELDS[selected_field]
    var, options = field_vars[fname]

    if key == "Left":
        if var.get() == -1:
            var.set(options[-1])
        else:
            idx = options.index(var.get())
            var.set(options[(idx - 1) % len(options)])
        return

    if key == "Right":
        if var.get() == -1:
            var.set(options[0])
        else:
            idx = options.index(var.get())
            var.set(options[(idx + 1) % len(options)])
        return

    if key == "Return":
        save_and_next()

root.bind("<Key>", on_key)

save_btn = tk.Button(
    root, text="Save & Next ➡️",
    command=save_and_next,
    font=("Arial", 14),
    bg="#3b82f6",
    fg="white",
    width=18, height=2
)
save_btn.pack(pady=20)

load_image()
root.mainloop()