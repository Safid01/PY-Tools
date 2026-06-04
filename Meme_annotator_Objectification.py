import os
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# --- CONFIG ---
image_folder = r"C:\Users\safid\Downloads\Memes"
excel_path = r"C:\Users\safid\Downloads\Memes\Meme_annotations_Binar.xlsx"
field = "Objectification"
values = [0, 1]

# Load Excel
df = pd.read_excel(excel_path)
if field not in df.columns:
    df[field] = None

mask = df[field].isna()
rows_to_annotate = df[mask].index.tolist()

if not rows_to_annotate:
    print("All annotated!")
    exit()

current_index = 0

# GUI
root = tk.Tk()
root.title(f"{field} Annotation")
root.geometry("900x700")
root.configure(bg="#1e1e1e")

image_label = tk.Label(root, bg="#1e1e1e")
image_label.pack(pady=20)

filename_label = tk.Label(root, text="", font=("Arial", 15),
                          fg="white", bg="#1e1e1e")
filename_label.pack(pady=5)

# highlight variable
var = tk.IntVar(value=-1)

# Create frame row
row = tk.Frame(root, bg="#1e1e1e")
row.pack(pady=20)

tk.Label(row, text=field, fg="white", bg="#1e1e1e",
         font=("Arial", 14)).pack(side="left", padx=10)

# store button refs for highlight
buttons = {}

def update_highlight():
    """Updates the highlight effect based on selected value."""
    selected = var.get()
    for val, btn in buttons.items():
        if val == selected:
            btn.config(bg="#3b82f6", fg="white")
        else:
            btn.config(bg="#2d2d2d", fg="white")

def set_value(v):
    var.set(v)
    update_highlight()

# Radio buttons with highlight box style
for v in values:
    btn = tk.Label(row, text=str(v),
                   width=4,
                   font=("Arial", 14),
                   bg="#2d2d2d",
                   fg="white",
                   padx=10, pady=5)
    btn.pack(side="left", padx=8)
    btn.bind("<Button-1>", lambda e, val=v: set_value(val))
    buttons[v] = btn


def save_and_next():
    global current_index
    idx = rows_to_annotate[current_index]

    if var.get() == -1:
        messagebox.showwarning("Missing", f"Select: {field}")
        return

    df.at[idx, field] = var.get()
    df.to_excel(excel_path, index=False)

    current_index += 1
    if current_index >= len(rows_to_annotate):
        messagebox.showinfo("Done", "All annotated!")
        root.destroy()
        return
    load_image()

def load_image():
    idx = rows_to_annotate[current_index]
    name = df.at[idx, "Image_Name"]
    path = os.path.join(image_folder, name)

    img = Image.open(path)
    img.thumbnail((800, 500))
    tk_img = ImageTk.PhotoImage(img)
    image_label.configure(image=tk_img)
    image_label.image = tk_img

    filename_label.configure(text=name)
    var.set(-1)
    update_highlight()

def on_key(event):
    key = event.keysym
    current = var.get()

    if key == "Return":
        save_and_next()

    elif key == "Right":
        if current == -1:
            new_val = values[0]
        else:
            new_val = values[(values.index(current) + 1) % len(values)]
        set_value(new_val)

    elif key == "Left":
        if current == -1:
            new_val = values[-1]
        else:
            new_val = values[(values.index(current) - 1) % len(values)]
        set_value(new_val)

root.bind("<Key>", on_key)

# Save button
save_btn = tk.Button(
    root,
    text="Save & Next ➡️",
    command=save_and_next,
    font=("Arial", 14),
    bg="#3b82f6",
    fg="white",
    width=18,
    height=2
)
save_btn.pack(pady=20)

load_image()
root.mainloop()