### DOESN'T WORK (because the black threshold is never updated from 0) ###

import tkinter as tk
from tkinter import filedialog, messagebox
import rasterio
import numpy as np
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Click anywhere on image to pick a color")

def from_rgb(rgb):
    return "#%02x%02x%02x" % rgb

def check_picked_color(rgb):
    return all(value == rgb[0] for value in rgb)

def prompt_pick_color():
    messagebox.showinfo("Pick a Color", "Please pick a shade of gray from the image.")

def update_black_threshold():
    global latest_rgb, black_threshold
    
    if latest_rgb is not None and check_picked_color(latest_rgb):
        black_threshold = latest_rgb[0]
        print(f"The black threshold is set to: {black_threshold}")
    else:
        print("The picked color is not a shade of gray or no color has been picked.")
        prompt_pick_color()

def colorpic(e):
    global latest_rgb, rgb_label, color_display
    x = int(e.x * scale_factor)
    y = int(e.y * scale_factor)
    
    # Get the pixel value from the numpy array
    pixs = img[y, x]
    
    if isinstance(pixs, np.ndarray):
        pixs = tuple(pixs)
    else:
        pixs = (pixs, pixs, pixs)  # Convert single value to RGB
    
    latest_rgb = pixs
    color = from_rgb(pixs)
    
    rgb_label.config(text=f"RGB: {pixs}")
    color_display.config(bg=color)
    
    update_black_threshold()

def open_image():
    global img, photo, scale_factor, c
    
   # file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif;*.tiff"), ("All files", "*.*")])
    file_path = '/Users/skylargu/Desktop/selected_images/BeaufortSea_2021_04_28.tif'
    if not file_path:
        return
    
    try:
        with rasterio.open(file_path) as src:
            img = src.read(1)  # Read the first band
        
        # Normalize the image data to 0-255 range
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # Convert to PIL Image for display
        pil_img = Image.fromarray(img)
        
        # Calculate scaling factor
        max_size = (800, 600)  # Maximum display size
        pil_img.thumbnail(max_size, Image.LANCZOS)
        scale_factor = max(pil_img.size[0] / max_size[0], pil_img.size[1] / max_size[1])
        
        photo = ImageTk.PhotoImage(pil_img)
        
        # Update or create canvas
        if 'c' in globals():
            c.delete("all")
            c.config(width=photo.width(), height=photo.height())
        else:
            c = tk.Canvas(root, width=photo.width(), height=photo.height())
            c.pack()
        
        c.create_image(0, 0, anchor=tk.NW, image=photo)
        c.bind("<Button-1>", colorpic)
        
        print("Image loaded successfully")
    except Exception as e:
        print(f"Error opening image: {e}")
        messagebox.showerror("Error", f"Could not open the image: {e}")

# Initialize variables
latest_rgb = None
black_threshold = 0
img = None

# Create UI elements
open_button = tk.Button(root, text="Open TIFF Image", command=open_image)
open_button.pack()

rgb_label = tk.Label(root, text="RGB: (-, -, -)")
rgb_label.pack()

color_display = tk.Label(root, text="", width=10, height=2)
color_display.pack()

root.mainloop()
