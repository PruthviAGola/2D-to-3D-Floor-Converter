# START OF FILE final code 2d to 3d mini pro.txt (REVISED WALL DETECTION)

import cv2
import numpy as np
import easyocr
import pyvista as pv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import re
import math
import os
import json
from PIL import Image, ImageTk
from PIL.Image import Resampling # For Image.Resampling.LANCZOS
from sklearn.cluster import DBSCAN

# Attempt to import ximgproc for thinning, will be handled if not available
try:
    from cv2 import ximgproc
    XIMGPROC_AVAILABLE = True
except ImportError:
    XIMGPROC_AVAILABLE = False
    print("Warning: cv2.ximgproc not available. Skeletonization will be skipped. Consider installing 'opencv-contrib-python'.")


class FloorPlanConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced 2D to 3D Floor Plan Converter")
        self.root.geometry("1000x700")
        
        self.trocr_processor = None
        self.trocr_model = None

        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=True)
            print("EasyOCR loaded with GPU support.")
        except Exception as e:
            print(f"Could not load EasyOCR with GPU, trying CPU: {e}")
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                print("EasyOCR loaded with CPU support.")
            except Exception as e_cpu:
                print(f"Could not load EasyOCR on CPU: {e_cpu}. Text extraction will fail.")
                self.easyocr_reader = None

        self.image_path = None
        self.original_image_pil = None 
        self.displayed_image_pil = None 
        self.photo = None 
        self.photo_with_detections = None 
        self.canvas_image_id = None

        self.room_dimensions = {}
        self.walls = [] 
        self.curved_walls = [] 
        self.scale_factor = 1.0 
        self.display_scale_factor = 1.0 
        self.default_height = 9.0 
        self.wall_thickness = 0.5 
        self.room_positions = {}
        
        self.door_width_default = 2.8
        self.door_height_default = 6.8
        self.window_width_default = 3.0
        self.window_height_default = 4.0
        self.window_sill_default = 3.0

        self.materials = {
            "wall": "#C19A6B", 
            "floor": "#D2B48C", 
            "ceiling": "#F5F5DC", 
            "door_frame": "#A0522D", "door_panel": "#8B4513", 
            "window_frame": "#A0522D", "window_glass": "#ADD8E6", 
            "furniture": {
                "bed": "#4682B4", "nightstand": "#8B4513", "counter": "#D3D3D3", "island": "#A9A9A9",
                "sofa": "#6B8E23", "table": "#8B4513", "chair": "#CD853F", "bathtub": "#B0E0E6",
                "toilet": "#F0F8FF", "sink": "#F5F5F5", "wardrobe": "#8B4513", "tv_stand": "#A9A9A9",
                "bookshelf": "#8B4513", "desk": "#D3D3D3", "oven": "#696969", "refrigerator": "#FFFFFF",
            }
        }
        
        self.label_font_size = 14
        self.show_labels_in_3d = False 
        self.selection_rect = None
        self.start_x_canvas = None
        self.start_y_canvas = None
        
        self.setup_ui()
    
    def setup_ui(self):
        self.left_frame = ttk.Frame(self.root, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.control_frame = ttk.LabelFrame(self.left_frame, text="Controls", padding=10)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        self.upload_button = ttk.Button(self.control_frame, text="Upload Floor Plan", command=self.upload_image)
        self.upload_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.process_button = ttk.Button(self.control_frame, text="Process Image", command=self.process_current_image, state=tk.DISABLED)
        self.process_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.generate_button = ttk.Button(self.control_frame, text="Generate 3D Model", command=self.generate_3d_model, state=tk.DISABLED)
        self.generate_button.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(self.control_frame, text="Room Height (ft):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.height_var = tk.StringVar(value=str(self.default_height))
        self.height_entry = ttk.Entry(self.control_frame, textvariable=self.height_var, width=10)
        self.height_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.control_frame, text="Wall Thickness (ft):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.thickness_var = tk.StringVar(value=str(self.wall_thickness))
        self.thickness_entry = ttk.Entry(self.control_frame, textvariable=self.thickness_var, width=10)
        self.thickness_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.control_frame, text="3D Label Font Size:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.font_size_var = tk.StringVar(value=str(self.label_font_size))
        self.font_size_entry = ttk.Entry(self.control_frame, textvariable=self.font_size_var, width=10)
        self.font_size_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.show_labels_var = tk.BooleanVar(value=self.show_labels_in_3d)
        self.show_labels_checkbox = ttk.Checkbutton(self.control_frame, text="Show Room Labels in 3D", 
                                                   variable=self.show_labels_var)
        self.show_labels_checkbox.grid(row=3, column=2, padx=5, pady=5, sticky=tk.W)
        
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset_app)
        self.reset_button.grid(row=4, column=0, padx=5, pady=5)
        
        self.save_button = ttk.Button(self.control_frame, text="Save Project", command=self.save_project)
        self.save_button.grid(row=4, column=1, padx=5, pady=5)
        
        self.load_button = ttk.Button(self.control_frame, text="Load Project", command=self.load_project)
        self.load_button.grid(row=4, column=2, padx=5, pady=5)
        
        self.image_frame = ttk.LabelFrame(self.left_frame, text="Floor Plan Image", padding=10)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.canvas = tk.Canvas(self.image_frame, bg="lightgrey", width=500, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.data_frame = ttk.LabelFrame(self.right_frame, text="Extracted Data", padding=10)
        self.data_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.tree = ttk.Treeview(self.data_frame, columns=("Room", "Dimensions", "Area"), show="headings")
        self.tree.heading("Room", text="Room")
        self.tree.heading("Dimensions", text="Dimensions")
        self.tree.heading("Area", text="Area (sq ft)")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(self.data_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.manual_frame = ttk.LabelFrame(self.right_frame, text="Manual Room Entry", padding=10)
        self.manual_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.manual_frame, text="Room Name:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.room_name_var = tk.StringVar()
        self.room_name_entry = ttk.Entry(self.manual_frame, textvariable=self.room_name_var, width=15)
        self.room_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.manual_frame, text="Width (ft):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.width_var = tk.StringVar()
        self.width_entry = ttk.Entry(self.manual_frame, textvariable=self.width_var, width=10)
        self.width_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.manual_frame, text="Length (ft):").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.length_var = tk.StringVar()
        self.length_entry = ttk.Entry(self.manual_frame, textvariable=self.length_var, width=10)
        self.length_entry.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)

        ttk.Label(self.manual_frame, text="X Pos (ft):").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.x_pos_var = tk.StringVar(value="0")
        self.x_pos_entry = ttk.Entry(self.manual_frame, textvariable=self.x_pos_var, width=10)
        self.x_pos_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        ttk.Label(self.manual_frame, text="Y Pos (ft):").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.y_pos_var = tk.StringVar(value="0")
        self.y_pos_entry = ttk.Entry(self.manual_frame, textvariable=self.y_pos_var, width=10)
        self.y_pos_entry.grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.manual_frame, text="Room Type:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.room_type_var = tk.StringVar(value="Bedroom")
        room_type_combo = ttk.Combobox(self.manual_frame, textvariable=self.room_type_var, 
                                     values=["Bedroom", "Kitchen", "Living Room", "Bathroom", "Dining Room", "Office", "Other"], width=15)
        room_type_combo.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.add_room_button = ttk.Button(self.manual_frame, text="Add Room", command=self.add_manual_room)
        self.add_room_button.grid(row=2, column=3, columnspan=2, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def reset_app(self):
        self.image_path = None
        self.original_image_pil = None
        self.displayed_image_pil = None
        self.photo = None
        self.photo_with_detections = None
        
        self.room_dimensions = {}
        self.walls = []
        self.curved_walls = []
        self.scale_factor = 1.0 
        self.display_scale_factor = 1.0
        self.room_positions = {}
        
        if self.canvas.winfo_exists():
            self.canvas.delete("all") 
        self.canvas_image_id = None 
        
        self.tree.delete(*self.tree.get_children())
        self.process_button.config(state=tk.DISABLED)
        self.generate_button.config(state=tk.DISABLED)
        self.status_var.set("Application reset. Upload an image to start.")

    def save_project(self):
        if not self.image_path and not self.room_dimensions and not self.walls and not self.curved_walls:
            messagebox.showinfo("Save Project", "Nothing to save.")
            return

        project_data = {
            "image_path": self.image_path,
            "room_dimensions": self.room_dimensions,
            "walls": self.walls,
            "curved_walls": self.curved_walls,
            "scale_factor": self.scale_factor,
            "default_height": float(self.height_var.get()) if self.height_var.get() else self.default_height,
            "wall_thickness": float(self.thickness_var.get()) if self.thickness_var.get() else self.wall_thickness,
            "room_positions": self.room_positions
        }
        
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(project_data, f, indent=4)
                self.status_var.set(f"Project saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save project: {e}")
                self.status_var.set("Error saving project")
    
    def load_project(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, "r") as f:
                    project_data = json.load(f)
                
                self.reset_app() 

                self.image_path = project_data.get("image_path")
                self.room_dimensions = project_data.get("room_dimensions", {})
                self.walls = project_data.get("walls", [])
                self.curved_walls = project_data.get("curved_walls", [])
                self.scale_factor = project_data.get("scale_factor", 1.0)
                self.default_height = project_data.get("default_height", self.default_height)
                self.wall_thickness = project_data.get("wall_thickness", self.wall_thickness)
                self.room_positions = project_data.get("room_positions", {})

                self.height_var.set(str(self.default_height))
                self.thickness_var.set(str(self.wall_thickness))
                
                if self.image_path and os.path.exists(self.image_path):
                    self.original_image_pil = Image.open(self.image_path).convert("RGB")
                    self.display_image(self.original_image_pil.copy()) 
                    self.process_button.config(state=tk.NORMAL)
                    if self.walls or self.curved_walls or \
                       any(data.get("ocr_bbox_center_pixels") or data.get("pixel_bounds") 
                           for data in self.room_dimensions.values()):
                        self.visualize_detections_on_canvas()
                else:
                    if self.canvas.winfo_exists():
                        self.canvas.delete("all")
                        canvas_w = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 250
                        canvas_h = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 200
                        if self.image_path: 
                            self.canvas.create_text(canvas_w // 2, canvas_h // 2,
                                                    text=f"Image not found: {os.path.basename(self.image_path)}", fill="red", width=canvas_w-20)
                        else: 
                             self.canvas.create_text(canvas_w // 2, canvas_h // 2,
                                                    text="No image specified in project.", fill="red")
                    self.image_path = None 
                    self.original_image_pil = None
                    self.displayed_image_pil = None
                    self.canvas_image_id = None 

                self.update_room_list() 
                self.status_var.set(f"Project loaded from {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Load Error", f"Could not load project: {e}")
                self.status_var.set("Error loading project")
                
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
            
        self.image_path = file_path 
        try:
            img_test = Image.open(file_path)
            img_test.verify() 
            img_test.close() 
            self.original_image_pil = Image.open(file_path).convert("RGB") 
        except Exception as e:
            messagebox.showerror("Image Error", f"Cannot open or invalid image: {e}\nPlease select a valid image file.")
            self.image_path = None 
            self.original_image_pil = None
            return

        self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
        self.process_button.config(state=tk.NORMAL)
        self.generate_button.config(state=tk.DISABLED) 
        self.room_dimensions = {} 
        self.walls = []
        self.curved_walls = []
        self.scale_factor = 1.0 
        self.update_room_list()
        self.display_image(self.original_image_pil.copy()) 

    def display_image(self, image_to_display_pil): 
        if not isinstance(image_to_display_pil, Image.Image):
            messagebox.showerror("Internal Error", "display_image expects a PIL Image object.")
            return

        if not (self.root.winfo_exists() and self.canvas.winfo_width() > 1 and self.canvas.winfo_height() > 1): 
            self.root.after(100, lambda p=image_to_display_pil: self.display_image(p))
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = image_to_display_pil.size
        
        current_display_scale_factor_for_this_image = min(canvas_width / img_width if img_width > 0 else 1, 
                                           canvas_height / img_height if img_height > 0 else 1, 1.0)
        
        if current_display_scale_factor_for_this_image <= 0: current_display_scale_factor_for_this_image = 1.0 
        
        new_width = int(img_width * current_display_scale_factor_for_this_image)
        new_height = int(img_height * current_display_scale_factor_for_this_image)
        
        if new_width <=0 or new_height <=0: 
            new_width = max(1, new_width); new_height = max(1, new_height)

        img_resized_pil = image_to_display_pil.resize((new_width, new_height), Resampling.LANCZOS)
        
        is_base_image_display = (self.original_image_pil is not None and 
                                 image_to_display_pil.tobytes() == self.original_image_pil.tobytes())


        if is_base_image_display and self.original_image_pil:
            self.displayed_image_pil = img_resized_pil 
            self.photo = ImageTk.PhotoImage(img_resized_pil, master=self.canvas) 
            photo_to_render_on_canvas = self.photo
            self.display_scale_factor = current_display_scale_factor_for_this_image
        else: 
            self.photo_with_detections = ImageTk.PhotoImage(img_resized_pil, master=self.canvas) 
            photo_to_render_on_canvas = self.photo_with_detections

        if self.canvas.winfo_exists():
            self.canvas.delete("all") 
        self.canvas_image_id = self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=photo_to_render_on_canvas)
        
        if is_base_image_display:
            self.canvas.unbind("<Button-1>") 
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.bind("<Button-1>", self.start_selection)
            self.canvas.bind("<B1-Motion>", self.update_selection)
            self.canvas.bind("<ButtonRelease-1>", self.end_selection)
        
        self.selection_rect = None
        self.start_x_canvas = None
        self.start_y_canvas = None
        
    def _canvas_to_original_coords(self, canvas_x, canvas_y):
        if not self.original_image_pil or self.display_scale_factor == 0: 
            return canvas_x, canvas_y 

        canvas_center_x = self.canvas.winfo_width() / 2
        canvas_center_y = self.canvas.winfo_height() / 2
        
        img_display_width_on_canvas = self.original_image_pil.width * self.display_scale_factor
        img_display_height_on_canvas = self.original_image_pil.height * self.display_scale_factor

        offset_x = canvas_center_x - img_display_width_on_canvas / 2
        offset_y = canvas_center_y - img_display_height_on_canvas / 2

        rel_x_on_displayed_img = canvas_x - offset_x
        rel_y_on_displayed_img = canvas_y - offset_y

        original_x = rel_x_on_displayed_img / self.display_scale_factor
        original_y = rel_y_on_displayed_img / self.display_scale_factor
        
        original_x = max(0, min(original_x, self.original_image_pil.width))
        original_y = max(0, min(original_y, self.original_image_pil.height))

        return original_x, original_y

    def start_selection(self, event):
        self.start_x_canvas = self.canvas.canvasx(event.x)
        self.start_y_canvas = self.canvas.canvasy(event.y)
        
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
            
        self.selection_rect = self.canvas.create_rectangle(
            self.start_x_canvas, self.start_y_canvas, self.start_x_canvas, self.start_y_canvas,
            outline="red", width=2, tags="selection_rect"
        )

    def update_selection(self, event):
        if not self.start_x_canvas or not self.selection_rect or not self.canvas.winfo_exists(): return 
        cur_x_canvas = self.canvas.canvasx(event.x)
        cur_y_canvas = self.canvas.canvasy(event.y)
        self.canvas.coords(self.selection_rect, self.start_x_canvas, self.start_y_canvas, cur_x_canvas, cur_y_canvas)

    def end_selection(self, event):
        if self.start_x_canvas is None or self.selection_rect is None or not self.canvas.winfo_exists():
            if self.selection_rect and self.canvas.winfo_exists(): self.canvas.delete(self.selection_rect)
            self.selection_rect = None
            return

        end_x_canvas = self.canvas.canvasx(event.x)
        end_y_canvas = self.canvas.canvasy(event.y)

        x1_c = min(self.start_x_canvas, end_x_canvas)
        y1_c = min(self.start_y_canvas, end_y_canvas)
        x2_c = max(self.start_x_canvas, end_x_canvas)
        y2_c = max(self.start_y_canvas, end_y_canvas)

        orig_x1, orig_y1 = self._canvas_to_original_coords(x1_c, y1_c)
        orig_x2, orig_y2 = self._canvas_to_original_coords(x2_c, y2_c)

        if abs(orig_x2 - orig_x1) < 5 or abs(orig_y2 - orig_y1) < 5: 
            if self.selection_rect and self.canvas.winfo_exists(): self.canvas.delete(self.selection_rect)
            self.selection_rect = None
            return

        room_name = simpledialog.askstring("Room Name", "Enter room name for selection:")
        if room_name:
            scale_input_str = "" 
            if self.scale_factor == 1.0:
                 try:
                    scale_input_str = simpledialog.askstring("Scale Factor", "Enter scale: PIXELS PER FOOT (e.g., 10 means 10px = 1ft).\nLeave empty if dimensions will be defined by OCR for this room.")
                    if scale_input_str: 
                        user_scale = float(scale_input_str)
                        if user_scale <=0: raise ValueError("Scale must be positive.")
                        self.scale_factor = user_scale
                 except ValueError as e:
                    messagebox.showerror("Error", f"Invalid scale factor: {e}. Using previous scale or 1.0 (pixels = feet for this room).")

            needs_ocr_for_dims = (not scale_input_str and self.scale_factor == 1.0)

            if needs_ocr_for_dims:
                self.room_dimensions[room_name] = {
                    "pixel_bounds": (orig_x1, orig_y1, orig_x2, orig_y2), "width": 0, "length": 0, 
                    "dim_str": "To be OCR'd", "area": 0, "type": self.determine_room_type(room_name),
                    "position_pixels": ((orig_x1 + orig_x2) / 2, (orig_y1 + orig_y2) / 2)
                }
            else: 
                current_scale_for_calc = self.scale_factor if self.scale_factor > 0 else 1.0
                width_px = abs(orig_x2 - orig_x1)
                length_px = abs(orig_y2 - orig_y1)
                width_ft = width_px / current_scale_for_calc
                length_ft = length_px / current_scale_for_calc
                dim_str = f"{width_ft:.1f}' x {length_ft:.1f}'"
                
                center_x_ft = (orig_x1 / current_scale_for_calc) + (width_ft / 2) 
                center_y_ft = (orig_y1 / current_scale_for_calc) + (length_ft / 2) 
                
                self.room_dimensions[room_name] = {
                    "width": width_ft, "length": length_ft, "dim_str": dim_str, "area": width_ft * length_ft, 
                    "type": self.determine_room_type(room_name), "position": (center_x_ft, center_y_ft),
                    "pixel_bounds": (orig_x1, orig_y1, orig_x2, orig_y2) 
                }
                self.room_positions[room_name] = { 
                    "center_x": center_x_ft, "center_y": center_y_ft,
                    "min_x": orig_x1 / current_scale_for_calc, "max_x": orig_x2 / current_scale_for_calc,
                    "min_y": orig_y1 / current_scale_for_calc, "max_y": orig_y2 / current_scale_for_calc
                }
            self.update_room_list()
            self.visualize_detections_on_canvas() 
        
        if self.selection_rect and self.canvas.winfo_exists():
            self.canvas.delete(self.selection_rect)
        self.selection_rect = None

    def determine_room_type(self, room_name):
        room_name_lower = room_name.lower()
        if "bed" in room_name_lower: return "Bedroom"
        if "kitch" in room_name_lower: return "Kitchen"
        if "living" in room_name_lower: return "Living Room"
        if "bath" in room_name_lower: return "Bathroom"
        if "dining" in room_name_lower: return "Dining Room"
        if "office" in room_name_lower: return "Office"
        if "hall" in room_name_lower: return "Other" 
        return "Other"

    def update_room_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for room_name, data in self.room_dimensions.items():
            pos_info = ""
            if "position" in data and isinstance(data["position"], tuple) and len(data["position"]) == 2:
                pos_x, pos_y = data["position"]
                if pos_x != 0 or pos_y != 0 : 
                    pos_info = f" @({pos_x:.1f}, {pos_y:.1f})"
            elif "position_pixels" in data and data.get("dim_str") == "To be OCR'd": 
                px_center, py_center = data["position_pixels"]
                pos_info = f" (sel. px: {int(px_center)},{int(py_center)})"

            dim_str = data.get("dim_str", "N/A")
            area = data.get("area", 0)
            if area == 0 and dim_str != "To be OCR'd" and dim_str != "N/A": 
                try:
                    w = data.get("width",0)
                    l = data.get("length",0)
                    area = w * l
                except: pass
                
            self.tree.insert("", tk.END, values=(
                room_name, 
                dim_str + pos_info, 
                f"{area:.2f}" if area > 0 else ("Pending OCR" if dim_str == "To be OCR'd" else "0.00")
            ))
        
        if self.room_dimensions or self.walls or self.curved_walls: 
            self.generate_button.config(state=tk.NORMAL)
        else:
            self.generate_button.config(state=tk.DISABLED)

    def add_manual_room(self):
        room_name = self.room_name_var.get().strip()
        width_str = self.width_var.get().strip()
        length_str = self.length_var.get().strip()
        room_type = self.room_type_var.get()
        x_pos_str = self.x_pos_var.get().strip() 
        y_pos_str = self.y_pos_var.get().strip() 
        
        if not all([room_name, width_str, length_str, x_pos_str, y_pos_str]):
            messagebox.showerror("Input Error", "Please enter all room information, including X and Y position.")
            return
        
        try:
            width_ft = float(width_str)
            length_ft = float(length_str)
            pos_x = float(x_pos_str) 
            pos_y = float(y_pos_str) 
            
            if width_ft <= 0 or length_ft <= 0:
                messagebox.showerror("Input Error", "Width and length must be positive numbers.")
                return

            dim_str = f"{width_ft:.1f}' x {length_ft:.1f}'"
            
            self.room_dimensions[room_name] = {
                "width": width_ft, "length": length_ft, "dim_str": dim_str,
                "area": width_ft * length_ft, "type": room_type,
                "position": (pos_x, pos_y) 
            }
            self.room_positions[room_name] = { 
                "center_x": pos_x, "center_y": pos_y,
                "min_x": pos_x - width_ft / 2, "max_x": pos_x + width_ft / 2,
                "min_y": pos_y - length_ft / 2, "max_y": pos_y + length_ft / 2
            }
            self.update_room_list()
            if self.original_image_pil: 
                self.visualize_detections_on_canvas()

            self.room_name_var.set("")
            self.width_var.set("")
            self.length_var.set("")
            self.x_pos_var.set("0")
            self.y_pos_var.set("0")
            self.status_var.set(f"Added room: {room_name}")
        except ValueError:
            messagebox.showerror("Input Error", "Width, length, and position must be numbers.")

    def process_current_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return
        if not self.original_image_pil: 
            messagebox.showerror("Error", "Original image data not loaded. Please re-upload.")
            return

        self.status_var.set("Processing image...")
        self.root.update_idletasks()
        
        try:
            cv_original_image = np.array(self.original_image_pil.convert('RGB'))
            cv_original_image = cv2.cvtColor(cv_original_image, cv2.COLOR_RGB2BGR)
            
            image_for_text_masking = cv_original_image.copy()
            image_for_wall_detection = cv_original_image.copy() # This will be modified by text masking

            if self.scale_factor == 1.0: 
                 try:
                    scale_input = simpledialog.askstring("Scale Factor Confirmation", 
                                                         f"Current scale: {self.scale_factor:.2f} px/ft. "
                                                         "Enter new scale (PIXELS PER FOOT, e.g., 10 for 10px=1ft) "
                                                         "or leave empty to use current. This is crucial for dimensions.")
                    if scale_input:
                        new_scale = float(scale_input)
                        if new_scale <= 0: raise ValueError("Scale must be positive")
                        self.scale_factor = new_scale
                 except ValueError:
                    messagebox.showwarning("Scale Warning", f"Invalid scale input. Using existing scale: {self.scale_factor:.2f} px/ft.")
            
            if self.easyocr_reader:
                self.status_var.set("Performing OCR to mask text for wall detection...")
                self.root.update_idletasks()
                try:
                    gray_for_mask_ocr = cv2.cvtColor(image_for_text_masking, cv2.COLOR_BGR2GRAY)
                    ocr_results_for_masking = self.easyocr_reader.readtext(gray_for_mask_ocr, detail=1, paragraph=False)
                    
                    for (bbox, text, prob) in ocr_results_for_masking:
                        if prob < 0.3: continue 
                        points = np.array(bbox, dtype=np.int32)
                        rect_x_min = np.min(points[:, 0]) - 5 
                        rect_y_min = np.min(points[:, 1]) - 5
                        rect_x_max = np.max(points[:, 0]) + 5
                        rect_y_max = np.max(points[:, 1]) + 5
                        cv2.rectangle(image_for_wall_detection, 
                                      (rect_x_min, rect_y_min), (rect_x_max, rect_y_max), 
                                      (255, 255, 255), -1) 
                except Exception as e:
                    print(f"Error during OCR for text masking: {e}. Wall detection might be affected.")
                    self.status_var.set("OCR for masking failed or had issues. Proceeding...")

            self.status_var.set("Detecting walls and curves...")
            self.root.update_idletasks()
            self.walls = self.detect_walls(image_for_wall_detection) 
            self.curved_walls = self.detect_curved_walls(image_for_wall_detection) 

            # Manual Injection of Openings for Demonstration 
            if len(self.walls) > 0 and self.scale_factor > 0:
                walls_with_openings_added = 0
                for i, wall_data in enumerate(self.walls):
                    if walls_with_openings_added >= max(3, len(self.walls) // 5) : break # Add to a few walls
                    
                    wall_length_ft = wall_data["length"] / self.scale_factor
                    if wall_length_ft > 4: # Min wall length to add an opening
                        if "openings" not in wall_data: wall_data["openings"] = []
                        
                        # Add a sample door if wall is long enough for it
                        if wall_length_ft > self.door_width_default + 1.0:
                            wall_data["openings"].append({
                                "position_on_wall": wall_data["length"] * 0.5, 
                                "width_px": self.door_width_default * self.scale_factor,
                                "height_px": self.door_height_default * self.scale_factor,
                                "sill_px": 0, 
                                "type": "door"
                            })
                        
                        # Add a sample window if wall is significantly longer
                        if wall_length_ft > self.door_width_default + self.window_width_default + 3: 
                            wall_data["openings"].append({
                                "position_on_wall": wall_data["length"] * 0.25, 
                                "width_px": self.window_width_default * self.scale_factor,
                                "height_px": self.window_height_default * self.scale_factor,
                                "sill_px": self.window_sill_default * self.scale_factor,
                                "type": "window"
                            })
                        if wall_data["openings"]: # Only count if we actually added something
                            print(f"Manually added sample openings to wall index {i} (length: {wall_length_ft:.1f} ft)")
                            walls_with_openings_added +=1


            if self.easyocr_reader:
                self.status_var.set("Extracting room descriptions from original image...")
                self.root.update_idletasks()
                self.extract_room_descriptions(cv_original_image) 
            else:
                self.status_var.set("EasyOCR not available. Skipping text extraction.")

            self.update_room_list()
            self.visualize_detections_on_canvas() 
            
            self.status_var.set(f"Processed: {len(self.room_dimensions)} rooms, {len(self.walls)} walls, {len(self.curved_walls)} curves. Scale: {self.scale_factor:.2f} px/ft")
            if self.room_dimensions or self.walls or self.curved_walls:
                self.generate_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error processing image: {str(e)}")
            self.status_var.set(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()

    def visualize_detections_on_canvas(self):
        if not self.original_image_pil:
            if self.canvas.winfo_exists() and self.canvas.winfo_width() > 1 : 
                 self.canvas.delete("all") 
                 canvas_w = self.canvas.winfo_width()
                 canvas_h = self.canvas.winfo_height()
                 self.canvas.create_text(canvas_w//2, canvas_h//2,
                                        text="Original image not available for drawing detections.", fill="orange")
            return

        pil_image_for_drawing = self.original_image_pil.copy()
        image_for_drawing_cv = np.array(pil_image_for_drawing.convert("RGB"))
        image_for_drawing_cv = cv2.cvtColor(image_for_drawing_cv, cv2.COLOR_RGB2BGR)

        for room_name, data in self.room_dimensions.items():
            text_to_display = room_name
            label_x_px, label_y_px = 0, 0 
            
            if "ocr_bbox_center_pixels" in data: 
                label_x_px, label_y_px = map(int, data["ocr_bbox_center_pixels"])
                if data.get("dim_str") and data["dim_str"] != "To be OCR'd":
                     text_to_display += f"\n{data['dim_str']}"
            elif "pixel_bounds" in data: 
                b_x1, b_y1, b_x2, b_y2 = data["pixel_bounds"]
                cv2.rectangle(image_for_drawing_cv, (int(b_x1), int(b_y1)), (int(b_x2), int(b_y2)), (255, 0, 0), 1) 
                label_x_px, label_y_px = int((b_x1 + b_x2) / 2), int(b_y1 - 10) 
                if data.get("dim_str") and data["dim_str"] != "To be OCR'd":
                     text_to_display += f"\n{data['dim_str']}"
                elif data.get("dim_str") == "To be OCR'd":
                     text_to_display += "\n(To be OCR'd)"
            elif "position" in data and self.scale_factor > 0 : 
                center_x_ft, center_y_ft = data["position"]
                label_x_px = int(center_x_ft * self.scale_factor)
                label_y_px = int(center_y_ft * self.scale_factor)
                text_to_display += f"\n{data.get('dim_str', '')}"

            if label_x_px > 0 and label_y_px > 0:
                y0, dy = label_y_px, 12 
                for i, line_text in enumerate(text_to_display.split('\n')):
                    y = y0 + i * dy
                    cv2.putText(image_for_drawing_cv, line_text, (max(0,label_x_px), max(10,y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 128), 1, cv2.LINE_AA) 

        for wall in self.walls:
            cv2.line(image_for_drawing_cv, wall["start"], wall["end"], (0, 128, 0), 3, cv2.LINE_AA) 
            if "openings" in wall and self.scale_factor > 0:
                p1 = np.array(wall["start"])
                p2 = np.array(wall["end"])
                wall_vec = p2 - p1
                wall_len_px_calc = np.linalg.norm(wall_vec)
                if wall_len_px_calc < 1e-6: continue
                wall_unit_vec = wall_vec / wall_len_px_calc
                
                for opening in wall["openings"]:
                    pos_on_wall_px = opening["position_on_wall"] 
                    w_px = opening["width_px"]
                    
                    op_start_on_wall_coord = pos_on_wall_px - w_px/2
                    op_end_on_wall_coord   = pos_on_wall_px + w_px/2

                    op_start_pt_on_canvas = p1 + wall_unit_vec * op_start_on_wall_coord
                    op_end_pt_on_canvas   = p1 + wall_unit_vec * op_end_on_wall_coord

                    cv2.line(image_for_drawing_cv, 
                             (int(op_start_pt_on_canvas[0]), int(op_start_pt_on_canvas[1])),
                             (int(op_end_pt_on_canvas[0]), int(op_end_pt_on_canvas[1])),
                             (0,0,255) if opening["type"] == "door" else (255,165,0), 4) 


        for curve in self.curved_walls:
            points_orig_px = [ (int(p[0]), int(p[1])) for p in curve["points"] ]
            if points_orig_px and len(points_orig_px) > 1: 
                cv2.polylines(image_for_drawing_cv, [np.array(points_orig_px, dtype=np.int32)],
                              isClosed=False, color=(200, 200, 0), thickness=3, lineType=cv2.LINE_AA) 
        
        image_rgb_with_detections = cv2.cvtColor(image_for_drawing_cv, cv2.COLOR_BGR2RGB)
        pil_image_with_detections = Image.fromarray(image_rgb_with_detections)
        
        self.display_image(pil_image_with_detections) 

    def _merge_lines(self, lines, angle_threshold_deg=5, dist_threshold_px=20):
        if lines is None or len(lines) == 0:
            return []

        # Represent lines in Hessian normal form (rho, theta)
        # rho = x*cos(theta) + y*sin(theta)
        line_params = []
        for line_seg in lines:
            x1, y1, x2, y2 = line_seg[0]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0: continue # Skip zero-length segments

            angle_rad = np.arctan2(dy, dx)
            # Ensure theta is in [0, pi) for Hough-like representation
            if angle_rad < 0: angle_rad += np.pi
            if angle_rad >= np.pi: angle_rad -= np.pi # Should not happen if atan2 is used correctly

            # Calculate rho. Using midpoint of segment for stability.
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            rho = mid_x * np.cos(angle_rad) + mid_y * np.sin(angle_rad)
            
            line_params.append({'rho': rho, 'theta': angle_rad, 'points': [(x1, y1), (x2, y2)], 'orig_line': line_seg})

        # Sort by theta then rho to group similar lines
        line_params.sort(key=lambda p: (p['theta'], p['rho']))

        merged_lines_data = []
        if not line_params: return []

        current_group = [line_params[0]]
        for i in range(1, len(line_params)):
            prev_line = current_group[-1]
            curr_line = line_params[i]

            delta_theta_deg = np.degrees(abs(curr_line['theta'] - prev_line['theta']))
            # Handle angle wrap-around (e.g., 1deg and 179deg are similar for lines)
            if delta_theta_deg > 90: delta_theta_deg = 180 - delta_theta_deg
            
            delta_rho = abs(curr_line['rho'] - prev_line['rho'])

            # Check proximity of endpoints as well to avoid merging distant parallel lines
            # This is a simplified proximity check; more robust would be point-to-line distance
            close_endpoints = False
            for p_curr in curr_line['points']:
                for p_prev_grp in current_group: # Check against all lines in current group
                    for p_prev in p_prev_grp['points']:
                        dist_sq = (p_curr[0] - p_prev[0])**2 + (p_curr[1] - p_prev[1])**2
                        if dist_sq < (dist_threshold_px * 2)**2 : # Increased check radius
                            close_endpoints = True
                            break
                    if close_endpoints: break
                if close_endpoints: break


            if delta_theta_deg < angle_threshold_deg and delta_rho < dist_threshold_px and close_endpoints:
                current_group.append(curr_line)
            else:
                merged_lines_data.append(current_group)
                current_group = [curr_line]
        merged_lines_data.append(current_group) # Add the last group

        final_merged_segments = []
        for group in merged_lines_data:
            if not group: continue
            
            all_points = []
            for line_data in group:
                all_points.extend(line_data['points'])
            
            if not all_points: continue

            # Create a single line segment spanning the extent of all points in the group
            # Find the two points that are furthest apart
            max_dist_sq = -1
            p_start, p_end = all_points[0], all_points[0]

            if len(all_points) < 2: # Should not happen if group is valid
                if len(all_points) == 1:
                    # This indicates an issue, maybe make a tiny segment or skip
                    # For now, let's try to use the original line if possible
                    if group[0]['orig_line'] is not None:
                        final_merged_segments.append(group[0]['orig_line'])
                    continue

            # Brute-force find max distance pair (for small number of points in a group)
            # For larger groups, convex hull and rotating calipers would be more efficient
            # but for typical Hough fragments, this should be acceptable.
            for j in range(len(all_points)):
                for k in range(j + 1, len(all_points)):
                    pt1 = all_points[j]
                    pt2 = all_points[k]
                    d_sq = (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2
                    if d_sq > max_dist_sq:
                        max_dist_sq = d_sq
                        p_start = pt1
                        p_end = pt2
            
            final_merged_segments.append(np.array([[p_start[0], p_start[1], p_end[0], p_end[1]]], dtype=np.int32))
            
        return final_merged_segments

    def detect_walls(self, image_cv):
        if image_cv is None:
            print("Error: Received None image in detect_walls")
            return []

        # 1. Preprocessing
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Gaussian Blur (kernel size odd, e.g., (3,3) or (5,5))
        # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # For some floor plans, median blur might still be better if there's salt-and-pepper noise
        blurred = cv2.medianBlur(gray, 3) 

        # Adaptive Thresholding (walls become white, background black)
        # blockSize must be odd and >1. C is a constant subtracted from mean/weighted sum.
        # Fine-tune blockSize and C based on line thickness and contrast.
        binarized = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 21, 7) # Tunable
        # cv2.imwrite("debug_walls_adaptive_thresh.png", binarized)

        # Morphological Operations
        # Kernel for closing: A bit larger to connect slightly broken wall lines
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1)) # Rectangular kernel, more horizontal
        closed_h = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5)) # Rectangular kernel, more vertical
        closed_v = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        closed_img = cv2.bitwise_or(closed_h, closed_v) # Combine horizontal and vertical closing

        # Kernel for opening: Smaller to remove noise without eroding walls too much
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel_open, iterations=1)
        # cv2.imwrite("debug_walls_morph.png", opened_img)
        
        # Canny Edge Detection
        # Lower thresholds make it more sensitive. Higher thresholds are stricter.
        low_canny = 50  # Tunable
        high_canny = 150 # Tunable
        edges = cv2.Canny(opened_img, low_canny, high_canny, apertureSize=3)
        # cv2.imwrite("debug_walls_canny_edges.png", edges)

        # HoughLinesP Transform
        # threshold: Min number of votes (intersections in Hough space)
        # minLineLength: Min length of a line in pixels.
        # maxLineGap: Max allowed gap between points on the same line to link them.
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=30,       # Tunable (e.g., 20-50)
            minLineLength=20,   # Tunable (e.g., 15-50 pixels)
            maxLineGap=10       # Tunable (e.g., 5-20 pixels)
        )

        if lines is None:
            print("No lines detected by HoughP.")
            return []
            
        # Merge fragmented lines from HoughP
        # Angle threshold in degrees, distance threshold in pixels for grouping
        merged_hough_lines = self._merge_lines(lines, angle_threshold_deg=7, dist_threshold_px=25) # Tunable

        detected_walls = []
        min_final_wall_length = 25 # Minimum length for a wall after merging
        angle_tolerance_deg_hv = 8 # Stricter tolerance for Horizontal/Vertical

        for line_segment in merged_hough_lines:
            x1, y1, x2, y2 = line_segment[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            if length < min_final_wall_length: continue

            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            
            is_horizontal = (abs(angle_deg) < angle_tolerance_deg_hv or 
                             abs(abs(angle_deg) - 180.0) < angle_tolerance_deg_hv)
            is_vertical = (abs(abs(angle_deg) - 90.0) < angle_tolerance_deg_hv)
                               
            if is_horizontal: wall_type = "horizontal"
            elif is_vertical: wall_type = "vertical"
            else: continue # Skip diagonal lines for now, or assign "diagonal"
            
            detected_walls.append({
                "start": (int(x1), int(y1)), "end": (int(x2), int(y2)), 
                "type": wall_type, "length": length, "openings": [] 
            })
        
        print(f"Detected {len(detected_walls)} wall candidates after HoughP and merging.")
        return detected_walls

    def detect_curved_walls(self, image_cv):
        if image_cv is None: return []
        
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3)
        binarized = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 3)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) 
        closed_img = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel_open, iterations=1) 

        image_for_contours = opened_img
        if XIMGPROC_AVAILABLE:
            try:
                thinned = ximgproc.thinning(opened_img)
                image_for_contours = thinned
            except Exception as e:
                print(f"Error during thinning for curves: {e}. Using morphed image.")

        contours, _ = cv2.findContours(image_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        curved_walls_detected = []
        
        min_contour_length_pixels = 30 
        min_points_for_curve_approx = 4 
        min_contour_area_heuristic = 40

        for contour in contours:
            length = cv2.arcLength(contour, False) 
            if length < min_contour_length_pixels or cv2.contourArea(contour) < min_contour_area_heuristic:
                continue
            
            epsilon_factor = 0.01 
            epsilon = epsilon_factor * length 
            approx = cv2.approxPolyDP(contour, epsilon, False) 
            
            if len(approx) >= min_points_for_curve_approx:
                curved_walls_detected.append({
                    "points": [tuple(p[0]) for p in approx], "length": length, "openings": [] })
        print(f"Detected {len(curved_walls_detected)} curved wall candidates.")
        return curved_walls_detected

    def extract_room_descriptions(self, image_input_cv): 
        if not self.easyocr_reader:
            print("EasyOCR reader not initialized. Skipping text extraction.")
            return
        try:
            easyocr_results = self.easyocr_reader.readtext(image_input_cv, detail=1, paragraph=False)
        except Exception as e:
            print(f"EasyOCR error in extract_room_descriptions: {e}")
            return

        all_text_detections = []
        for (bbox, text, prob) in easyocr_results:
            if prob < 0.4: continue 
            points = np.array(bbox, dtype=np.int32)
            all_text_detections.append({
                "text": text, 
                "center_x_px": np.mean(points[:, 0]), "center_y_px": np.mean(points[:, 1]),
                "min_x_px": np.min(points[:, 0]), "max_x_px": np.max(points[:, 0]),
                "min_y_px": np.min(points[:, 1]), "max_y_px": np.max(points[:, 1]),
                "bbox_pixels": points.tolist() 
            })

        processed_detection_indices = set() 

        temp_room_dimensions = self.room_dimensions.copy() 
        for room_name, room_data in temp_room_dimensions.items():
            if "pixel_bounds" in room_data and room_data.get("dim_str") == "To be OCR'd": 
                sel_min_x, sel_min_y, sel_max_x, sel_max_y = room_data["pixel_bounds"]
                
                best_match_ocr = None
                min_dist_to_sel_center = float('inf')
                sel_center_x = (sel_min_x + sel_max_x) / 2
                sel_center_y = (sel_min_y + sel_max_y) / 2
                best_match_idx = -1 

                for idx, detection in enumerate(all_text_detections):
                    if idx in processed_detection_indices: continue
                    
                    ocr_box_center_x, ocr_box_center_y = detection["center_x_px"], detection["center_y_px"]
                    if (sel_min_x <= ocr_box_center_x <= sel_max_x and
                        sel_min_y <= ocr_box_center_y <= sel_max_y):
                        
                        dist = math.sqrt((ocr_box_center_x - sel_center_x)**2 + (ocr_box_center_y - sel_center_y)**2)
                        if dist < min_dist_to_sel_center:
                            min_dist_to_sel_center = dist
                            best_match_ocr = detection
                            best_match_idx = idx 
                
                if best_match_ocr and best_match_idx != -1: 
                    parsed = self._parse_room_text(best_match_ocr["text"])
                    if parsed:
                        width_ft, length_ft, dim_str = parsed
                        current_scale = self.scale_factor if self.scale_factor > 0 else 1.0

                        self.room_dimensions[room_name].update({
                            "width": width_ft, "length": length_ft, "dim_str": dim_str,
                            "area": width_ft * length_ft,
                            "position": (best_match_ocr["center_x_px"] / current_scale, 
                                         best_match_ocr["center_y_px"] / current_scale),
                            "ocr_bbox_center_pixels": (best_match_ocr["center_x_px"], best_match_ocr["center_y_px"])
                        })
                        if room_name in self.room_positions:
                             self.room_positions[room_name].update({
                                "center_x": best_match_ocr["center_x_px"] / current_scale, 
                                "center_y": best_match_ocr["center_y_px"] / current_scale,
                                "min_x": (best_match_ocr["center_x_px"] / current_scale) - (width_ft / 2), 
                                "max_x": (best_match_ocr["center_x_px"] / current_scale) + (width_ft / 2), 
                                "min_y": (best_match_ocr["center_y_px"] / current_scale) - (length_ft / 2), 
                                "max_y": (best_match_ocr["center_y_px"] / current_scale) + (length_ft / 2)
                            })
                        processed_detection_indices.add(best_match_idx)
        
        unprocessed_text_detections = [
            det for i, det in enumerate(all_text_detections) if i not in processed_detection_indices
        ]

        if unprocessed_text_detections:
            positions = np.array([[d["center_x_px"], d["center_y_px"]] for d in unprocessed_text_detections])
            if len(positions) > 0:
                clustering = DBSCAN(eps=60, min_samples=1).fit(positions) 
                cluster_labels = clustering.labels_
                
                processed_clusters = set()
                for i, detection_from_unprocessed_list in enumerate(unprocessed_text_detections):
                    cluster_id = cluster_labels[i]
                    if cluster_id == -1 or cluster_id in processed_clusters: continue 

                    cluster_elements = [unprocessed_text_detections[j] for j, cid in enumerate(cluster_labels) if cid == cluster_id]
                    cluster_elements.sort(key=lambda e: (e["center_y_px"], e["center_x_px"]))
                    full_cluster_text = " ".join([elem["text"] for elem in cluster_elements])
                    
                    avg_cluster_center_x_px = np.mean([elem["center_x_px"] for elem in cluster_elements])
                    avg_cluster_center_y_px = np.mean([elem["center_y_px"] for elem in cluster_elements])
                    
                    parsed_dims = self._parse_room_text(full_cluster_text)
                    if parsed_dims:
                        width_ft, length_ft, dim_str = parsed_dims
                        room_name_from_text = "Room"; 
                        
                        name_match = re.search(r'\b(kitchen|bath(?:room)?|bed(?:room)?|living|dining|office|hallway|garage|closet|study|room|master|guest|play|nook|den|pantry|foyer|laundry)\b', full_cluster_text.lower(), re.IGNORECASE)
                        if name_match:
                            room_name_from_text = name_match.group(1).capitalize()
                        else: 
                            candidate_name = re.sub(r'[^a-zA-Z0-9\s]', '', full_cluster_text).strip()
                            candidate_name_parts = candidate_name.split()
                            if candidate_name_parts:
                                room_name_from_text = " ".join(candidate_name_parts[:2]).title() if len(candidate_name_parts) >1 else candidate_name_parts[0].title()
                            if not room_name_from_text : room_name_from_text = "Area"

                        counter = 1; final_room_name = room_name_from_text
                        while final_room_name in self.room_dimensions: 
                            counter += 1; final_room_name = f"{room_name_from_text} {counter}"
                        
                        room_type = self.determine_room_type(final_room_name)
                        current_scale = self.scale_factor if self.scale_factor > 0 else 1.0
                        pos_x_ft = avg_cluster_center_x_px / current_scale
                        pos_y_ft = avg_cluster_center_y_px / current_scale

                        self.room_dimensions[final_room_name] = {
                            "width": width_ft, "length": length_ft, "dim_str": dim_str, "area": width_ft * length_ft, 
                            "type": room_type, "position": (pos_x_ft, pos_y_ft), 
                            "ocr_bbox_center_pixels": (avg_cluster_center_x_px, avg_cluster_center_y_px) 
                        }
                        self.room_positions[final_room_name] = {
                            "center_x": pos_x_ft, "center_y": pos_y_ft,
                            "min_x": pos_x_ft - width_ft / 2, "max_x": pos_x_ft + width_ft / 2, 
                            "min_y": pos_y_ft - length_ft / 2, "max_y": pos_y_ft + length_ft / 2 
                        }
                    processed_clusters.add(cluster_id)

    def _parse_room_text(self, text): 
        dim_pattern = re.compile(
            r"(\d+)(?:['\‘\’`]\s*(?:(\d{1,2})\s*[\"”])?)?"  
            r"\s*[xX]\s*"                                   
            r"(\d+)(?:['\‘\’`]\s*(?:(\d{1,2})\s*[\"”])?)?"  
        )
        processed_text = text.replace("’", "'").replace("”", '"').replace("`","'")
        
        match = dim_pattern.search(processed_text)
        if match:
            try:
                w_ft_str, w_in_str, l_ft_str, l_in_str = match.groups()
                
                width_ft = float(w_ft_str)
                if w_in_str: width_ft += float(w_in_str) / 12.0
                
                length_ft = float(l_ft_str)
                if l_in_str: length_ft += float(l_in_str) / 12.0

                has_foot_marker_w_in_match = "'" in (w_ft_str or "") or (match.group(0) and "'" in match.group(0).split('x')[0].split('X')[0])
                has_foot_marker_l_in_match = "'" in (l_ft_str or "") or (match.group(0) and len(match.group(0).split('x')) > 1 and "'" in match.group(0).split('x')[1].split('X')[0]) \
                                        or (match.group(0) and len(match.group(0).split('X')) > 1 and "'" in match.group(0).split('X')[1].split('x')[0])


                if not (has_foot_marker_w_in_match or has_foot_marker_l_in_match):
                    if not w_in_str and width_ft > 40 and (width_ft % 1 == 0):
                         potential_w_from_inches = width_ft / 12.0
                         if 2 < potential_w_from_inches < 30: 
                             width_ft = potential_w_from_inches
                    if not l_in_str and length_ft > 40 and (length_ft % 1 == 0):
                         potential_l_from_inches = length_ft / 12.0
                         if 2 < potential_l_from_inches < 30:
                             length_ft = potential_l_from_inches
                                
                dim_str = f"{width_ft:.1f}' x {length_ft:.1f}'"
                return width_ft, length_ft, dim_str
            except ValueError: 
                return None
        return None

    def create_wall_segment_3d(self, plotter, p1_2d_ft, p2_2d_ft, z_start_ft, height_ft, thickness_ft, wall_color):
        dx = p2_2d_ft[0] - p1_2d_ft[0]
        dy = p2_2d_ft[1] - p1_2d_ft[1]
        length_sq = dx*dx + dy*dy
        if length_sq < 1e-6: return 
        length = math.sqrt(length_sq)

        dx_norm, dy_norm = dx/length, dy/length
        perp_x, perp_y = -dy_norm, dx_norm 

        half_thick = thickness_ft / 2.0
        
        v = [
            (p1_2d_ft[0] - perp_x*half_thick, p1_2d_ft[1] - perp_y*half_thick, z_start_ft),
            (p1_2d_ft[0] + perp_x*half_thick, p1_2d_ft[1] + perp_y*half_thick, z_start_ft),
            (p2_2d_ft[0] + perp_x*half_thick, p2_2d_ft[1] + perp_y*half_thick, z_start_ft),
            (p2_2d_ft[0] - perp_x*half_thick, p2_2d_ft[1] - perp_y*half_thick, z_start_ft),
            (p1_2d_ft[0] - perp_x*half_thick, p1_2d_ft[1] - perp_y*half_thick, z_start_ft + height_ft),
            (p1_2d_ft[0] + perp_x*half_thick, p1_2d_ft[1] + perp_y*half_thick, z_start_ft + height_ft),
            (p2_2d_ft[0] + perp_x*half_thick, p2_2d_ft[1] + perp_y*half_thick, z_start_ft + height_ft),
            (p2_2d_ft[0] - perp_x*half_thick, p2_2d_ft[1] - perp_y*half_thick, z_start_ft + height_ft)
        ]
        vertices = np.array(v)
        
        faces = np.array([
            4, 0, 1, 2, 3,  # Bottom
            4, 7, 6, 5, 4,  # Top
            4, 0, 4, 7, 3,  # Side
            4, 1, 5, 6, 2,  # Side
            4, 0, 1, 5, 4,  # End
            4, 3, 2, 6, 7   # End
        ]).ravel()


        segment_mesh = pv.PolyData(vertices, faces)
        plotter.add_mesh(segment_mesh, color=wall_color, smooth_shading=False)

    def create_wall_with_openings(self, plotter, wall_data_px, overall_height_ft, thickness_ft, scale_factor):
        start_px_orig = np.array(wall_data_px["start"])
        end_px_orig = np.array(wall_data_px["end"])
        wall_length_px_orig = wall_data_px["length"]
        
        if wall_length_px_orig < 1e-6 or scale_factor < 1e-6: return

        wall_unit_vec_px = (end_px_orig - start_px_orig) / wall_length_px_orig if wall_length_px_orig > 1e-9 else np.array([0,0])
        wall_color = self.materials["wall"]
        
        processed_openings = []
        for op_px_data in wall_data_px.get("openings", []):
            center_pos_on_wall_px = op_px_data["position_on_wall"]
            width_px = op_px_data["width_px"]
            
            op_start_dist_px = center_pos_on_wall_px - width_px / 2.0
            op_end_dist_px = center_pos_on_wall_px + width_px / 2.0
            
            op_start_dist_px = max(0, op_start_dist_px)
            op_end_dist_px = min(wall_length_px_orig, op_end_dist_px)
            if op_end_dist_px <= op_start_dist_px + 1e-3: continue 

            processed_openings.append({
                "start_dist_px": op_start_dist_px, 
                "end_dist_px": op_end_dist_px,
                "width_px": op_end_dist_px - op_start_dist_px, 
                "height_px": op_px_data["height_px"],
                "sill_px": op_px_data["sill_px"], 
                "type": op_px_data["type"]
            })
        processed_openings.sort(key=lambda o: o["start_dist_px"])

        current_wall_pos_px = 0.0 

        for op in processed_openings:
            op_s_px = op["start_dist_px"] 
            op_e_px = op["end_dist_px"]   
            
            op_width_ft = op["width_px"] / scale_factor
            op_height_ft = op["height_px"] / scale_factor
            op_sill_ft = op["sill_px"] / scale_factor

            if op_s_px > current_wall_pos_px + 1e-3: 
                seg_start_pt_px = start_px_orig + wall_unit_vec_px * current_wall_pos_px
                seg_end_pt_px = start_px_orig + wall_unit_vec_px * op_s_px
                self.create_wall_segment_3d(plotter, seg_start_pt_px / scale_factor, seg_end_pt_px / scale_factor, 
                                            0, overall_height_ft, thickness_ft, wall_color)

            op_seg_start_pt_px = start_px_orig + wall_unit_vec_px * op_s_px
            op_seg_end_pt_px   = start_px_orig + wall_unit_vec_px * op_e_px
            
            op_seg_start_ft = op_seg_start_pt_px / scale_factor
            op_seg_end_ft   = op_seg_end_pt_px / scale_factor

            if op_sill_ft > 1e-3: 
                self.create_wall_segment_3d(plotter, op_seg_start_ft, op_seg_end_ft,
                                            0, op_sill_ft, thickness_ft, wall_color)

            header_start_z_ft = op_sill_ft + op_height_ft
            if header_start_z_ft < overall_height_ft - 1e-3: 
                header_height_ft = overall_height_ft - header_start_z_ft
                self.create_wall_segment_3d(plotter, op_seg_start_ft, op_seg_end_ft,
                                            header_start_z_ft, header_height_ft, thickness_ft, wall_color)
            
            op_center_pt_px = start_px_orig + wall_unit_vec_px * (op_s_px + op["width_px"] / 2.0)
            op_center_pt_ft = op_center_pt_px / scale_factor
            
            wall_angle_rad = math.atan2(wall_unit_vec_px[1], wall_unit_vec_px[0])

            if op["type"] == "door":
                self.create_door_model(plotter, op_center_pt_ft, op_width_ft, op_height_ft, thickness_ft, wall_angle_rad)
            elif op["type"] == "window":
                self.create_window_model(plotter, op_center_pt_ft, op_width_ft, op_height_ft, op_sill_ft, thickness_ft, wall_angle_rad)

            current_wall_pos_px = op_e_px 

        if current_wall_pos_px < wall_length_px_orig - 1e-3 : 
            seg_start_pt_px = start_px_orig + wall_unit_vec_px * current_wall_pos_px
            self.create_wall_segment_3d(plotter, seg_start_pt_px / scale_factor, end_px_orig / scale_factor, 
                                        0, overall_height_ft, thickness_ft, wall_color)

    def create_door_model(self, plotter, center_pos_2d_ft, width_ft, height_ft, wall_thickness_ft, angle_rad_wall):
        panel_color = self.materials["door_panel"]
        frame_color = self.materials["door_frame"]
        door_panel_thickness = 0.15 
        frame_element_thickness = 0.2 

        panel_width = width_ft - 2 * frame_element_thickness
        panel_height = height_ft - frame_element_thickness 

        if panel_width <= 0 or panel_height <= 0: 
            door_box = pv.Cube(center=(0,0, height_ft/2),
                               x_length=width_ft, y_length=wall_thickness_ft*0.8, z_length=height_ft)
            door_box.rotate_z(math.degrees(angle_rad_wall), inplace=True)
            door_box.translate(list(center_pos_2d_ft) + [0], inplace=True)
            plotter.add_mesh(door_box, color=panel_color, smooth_shading=False)
            return

        door_panel = pv.Cube(center=(0, 0, panel_height / 2), 
                             x_length=panel_width, 
                             y_length=door_panel_thickness, 
                             z_length=panel_height)
        door_panel.rotate_z(math.degrees(angle_rad_wall), inplace=True)
        door_panel.translate(list(center_pos_2d_ft) + [0], inplace=True)
        plotter.add_mesh(door_panel, color=panel_color, smooth_shading=False)

        frame_depth = wall_thickness_ft * 0.8 
        top_frame = pv.Cube(center=(0, 0, height_ft - frame_element_thickness / 2),
                            x_length=width_ft,
                            y_length=frame_depth, 
                            z_length=frame_element_thickness)
        left_frame = pv.Cube(center=(-width_ft/2 + frame_element_thickness/2, 0, (height_ft - frame_element_thickness)/2),
                             x_length=frame_element_thickness,
                             y_length=frame_depth,
                             z_length=height_ft - frame_element_thickness) 
        right_frame = pv.Cube(center=(width_ft/2 - frame_element_thickness/2, 0, (height_ft - frame_element_thickness)/2),
                              x_length=frame_element_thickness,
                              y_length=frame_depth,
                              z_length=height_ft - frame_element_thickness)

        frame_parts = [top_frame, left_frame, right_frame]
        for part in frame_parts:
            part.rotate_z(math.degrees(angle_rad_wall), inplace=True)
            part.translate(list(center_pos_2d_ft) + [0], inplace=True)
            plotter.add_mesh(part, color=frame_color, smooth_shading=False)


    def create_window_model(self, plotter, center_pos_2d_ft, width_ft, height_ft, sill_ft, wall_thickness_ft, angle_rad_wall):
        glass_color = self.materials["window_glass"]
        frame_color = self.materials["window_frame"]
        glass_thickness = 0.05 
        frame_element_thickness = 0.15 

        glass_width = width_ft - 2 * frame_element_thickness
        glass_height = height_ft - 2 * frame_element_thickness

        if glass_width <=0 or glass_height <=0: 
            window_box = pv.Cube(center=(0,0, sill_ft + height_ft/2),
                                 x_length=width_ft, y_length=wall_thickness_ft*0.7, z_length=height_ft)
            window_box.rotate_z(math.degrees(angle_rad_wall), inplace=True)
            window_box.translate(list(center_pos_2d_ft) + [0], inplace=True)
            plotter.add_mesh(window_box, color=glass_color, opacity=0.5, smooth_shading=False)
            return

        glass_pane_center_z = sill_ft + frame_element_thickness + glass_height / 2
        glass_pane = pv.Cube(center=(0, 0, glass_pane_center_z),
                             x_length=glass_width, 
                             y_length=glass_thickness, 
                             z_length=glass_height)
        glass_pane.rotate_z(math.degrees(angle_rad_wall), inplace=True)
        glass_pane.translate(list(center_pos_2d_ft) + [0], inplace=True)
        plotter.add_mesh(glass_pane, color=glass_color, opacity=0.5, smooth_shading=False)
        
        frame_depth = wall_thickness_ft * 0.7 
        top_f = pv.Cube(center=(0,0, sill_ft + height_ft - frame_element_thickness/2),
                        x_length=width_ft, y_length=frame_depth, z_length=frame_element_thickness)
        bot_f = pv.Cube(center=(0,0, sill_ft + frame_element_thickness/2),
                        x_length=width_ft, y_length=frame_depth, z_length=frame_element_thickness)
        left_f_height = height_ft - 2 * frame_element_thickness
        left_f_center_z = sill_ft + frame_element_thickness + left_f_height/2
        left_f = pv.Cube(center=(-width_ft/2 + frame_element_thickness/2, 0, left_f_center_z),
                         x_length=frame_element_thickness, y_length=frame_depth, z_length=left_f_height)
        right_f = pv.Cube(center=(width_ft/2 - frame_element_thickness/2, 0, left_f_center_z), 
                          x_length=frame_element_thickness, y_length=frame_depth, z_length=left_f_height)

        frame_parts = [top_f, bot_f, left_f, right_f]
        for part in frame_parts:
            part.rotate_z(math.degrees(angle_rad_wall), inplace=True)
            part.translate(list(center_pos_2d_ft) + [0], inplace=True)
            plotter.add_mesh(part, color=frame_color, smooth_shading=False)

    def create_curved_wall(self, plotter, points_2d_ft, height_ft, thickness_ft):
        if len(points_2d_ft) < 2: return
        
        path_points = np.array([(p[0], p[1], 0.0) for p in points_2d_ft]) 
        num_path_points = len(path_points)
        half_thickness = thickness_ft / 2.0
        
        offset_vectors_3d = [] 
        for i in range(num_path_points):
            tangent_3d = np.zeros(3)
            if i == 0: 
                if num_path_points > 1: tangent_3d = path_points[i+1] - path_points[i]
            elif i == num_path_points - 1: 
                tangent_3d = path_points[i] - path_points[i-1]
            else: 
                tangent_prev = path_points[i] - path_points[i-1]
                tangent_next = path_points[i+1] - path_points[i]
                norm_prev = np.linalg.norm(tangent_prev); norm_next = np.linalg.norm(tangent_next)
                if norm_prev > 1e-9: tangent_prev /= norm_prev
                if norm_next > 1e-9: tangent_next /= norm_next
                tangent_3d = (tangent_prev + tangent_next) 
                norm_avg_tangent = np.linalg.norm(tangent_3d)
                if norm_avg_tangent > 1e-9 : tangent_3d /= norm_avg_tangent

            tangent_2d = tangent_3d[:2] 
            norm_tangent_2d = np.linalg.norm(tangent_2d)
            
            normal_vec_2d = np.array([0.0, 1.0]) 
            if norm_tangent_2d > 1e-9:
                normalized_tangent_2d = tangent_2d / norm_tangent_2d
                normal_vec_2d = np.array([-normalized_tangent_2d[1], normalized_tangent_2d[0]]) 
            elif offset_vectors_3d: 
                prev_offset_dir = offset_vectors_3d[-1][:2]
                prev_offset_norm = np.linalg.norm(prev_offset_dir)
                if prev_offset_norm > 1e-9:
                    normal_vec_2d = prev_offset_dir / prev_offset_norm
            elif num_path_points > 1 and i + 1 < num_path_points : 
                fallback_tangent = path_points[i+1] - path_points[i]
                fallback_tangent_2d = fallback_tangent[:2]
                norm_fallback_tangent_2d = np.linalg.norm(fallback_tangent_2d)
                if norm_fallback_tangent_2d > 1e-9:
                    normalized_fallback_tangent_2d = fallback_tangent_2d / norm_fallback_tangent_2d
                    normal_vec_2d = np.array([-normalized_fallback_tangent_2d[1], normalized_fallback_tangent_2d[0]])

            offset_vectors_3d.append(np.array([normal_vec_2d[0], normal_vec_2d[1], 0.0]) * half_thickness)

        points_side1_base = path_points - np.array(offset_vectors_3d)
        points_side2_base = path_points + np.array(offset_vectors_3d)
        
        all_vertices_list = []
        for p1, p2 in zip(points_side1_base, points_side2_base):
            all_vertices_list.extend([p1, p2])
        
        num_base_vertices_total_strip = len(all_vertices_list) 
        
        for i in range(num_base_vertices_total_strip):
            p_base = all_vertices_list[i]
            all_vertices_list.append((p_base[0], p_base[1], height_ft))
            
        vertices_np = np.array(all_vertices_list)
        
        faces_list = []
        n_segments = num_path_points - 1 
        if n_segments <= 0: return

        offset_to_top_vertices = num_path_points * 2 
        
        for i in range(n_segments):
            idx_b_s1_curr = i * 2
            idx_b_s2_curr = i * 2 + 1    
            idx_b_s1_next = (i + 1) * 2
            idx_b_s2_next = (i + 1) * 2 + 1
            
            idx_t_s1_curr = idx_b_s1_curr + offset_to_top_vertices
            idx_t_s2_curr = idx_b_s2_curr + offset_to_top_vertices
            idx_t_s1_next = idx_b_s1_next + offset_to_top_vertices
            idx_t_s2_next = idx_b_s2_next + offset_to_top_vertices

            faces_list.append([4, idx_b_s1_curr, idx_b_s1_next, idx_t_s1_next, idx_t_s1_curr]) 
            faces_list.append([4, idx_b_s2_next, idx_b_s2_curr, idx_t_s2_curr, idx_t_s2_next]) 
            faces_list.append([4, idx_t_s1_curr, idx_t_s2_curr, idx_t_s2_next, idx_t_s1_next]) 

        if num_path_points > 0: 
            idx_b_s1_start = 0; idx_b_s2_start = 1   
            idx_t_s1_start = idx_b_s1_start + offset_to_top_vertices
            idx_t_s2_start = idx_b_s2_start + offset_to_top_vertices
            faces_list.append([4, idx_b_s2_start, idx_b_s1_start, idx_t_s1_start, idx_t_s2_start]) 
            
            idx_b_s1_end = (num_path_points - 1) * 2
            idx_b_s2_end = idx_b_s1_end + 1
            idx_t_s1_end = idx_b_s1_end + offset_to_top_vertices
            idx_t_s2_end = idx_b_s2_end + offset_to_top_vertices
            faces_list.append([4, idx_b_s1_end, idx_b_s2_end, idx_t_s2_end, idx_t_s1_end]) 

        if not faces_list: return
        try:
            curved_wall_mesh = pv.PolyData(vertices_np, faces=np.hstack(faces_list))
            if curved_wall_mesh.n_points > 0 and curved_wall_mesh.n_cells > 0:
                 plotter.add_mesh(curved_wall_mesh, color=self.materials["wall"], smooth_shading=False)
            else:
                print("Warning: Generated curved wall mesh is empty or invalid.")
        except Exception as e:
            print(f"Error creating PolyData for curved wall: {e} with {len(vertices_np)} vertices and faces_list: {len(faces_list)} cells.")


    def create_furniture(self, plotter, room_type, room_bounds_ft, room_height):
        x_min, x_max, y_min, y_max = room_bounds_ft
        width = x_max - x_min; length = y_max - y_min
        center_x, center_y = x_min + width/2, y_min + length/2

        if width <= 1e-3 or length <= 1e-3: return 
        furniture_color = self.materials["furniture"]

        if room_type == "Bedroom":
            bed_w, bed_l, bed_h = min(width * 0.6, 6.0), min(length * 0.7, 7.0), 2.0 
            if bed_w > 1.0 and bed_l > 1.5 : 
                if width < length: 
                    bed_x_pos, bed_y_pos = center_x, y_min + bed_l/2 + 0.5 
                else: 
                    bed_w, bed_l = bed_l, bed_w 
                    bed_x_pos, bed_y_pos = x_min + bed_w/2 + 0.5, center_y
                
                bed_bounds = [bed_x_pos - bed_w/2, bed_x_pos + bed_w/2,
                              bed_y_pos - bed_l/2, bed_y_pos + bed_l/2,
                              0, bed_h]
                plotter.add_mesh(pv.Box(bounds=bed_bounds), color=furniture_color["bed"])

        elif room_type == "Kitchen":
            counter_h, counter_d = 2.9, 2.0 
            if length > counter_d + 0.5 : 
                plotter.add_mesh(pv.Box(bounds=[x_min, x_max, y_max - counter_d, y_max, 0, counter_h]), 
                                 color=furniture_color["counter"]) 
            if width > counter_d + 0.5 : 
                y_extent_for_side_counter = y_max - (counter_d if length > counter_d + 0.5 else 0)
                if y_extent_for_side_counter > y_min:
                    plotter.add_mesh(pv.Box(bounds=[x_max - counter_d, x_max, y_min, y_extent_for_side_counter, 0, counter_h]), 
                                    color=furniture_color["counter"]) 
            if width > 7 and length > 7: 
                island_w, island_l = min(width*0.3, 4), min(length*0.25, 3)
                if island_w > 1.5 and island_l > 1.5:
                    island_bounds = [center_x - island_w/2, center_x + island_w/2,
                                     center_y - island_l/2, center_y + island_l/2,
                                     0, counter_h]
                    plotter.add_mesh(pv.Box(bounds=island_bounds), color=furniture_color["island"])

        elif room_type == "Living Room":
            sofa_max_w, sofa_max_d, sofa_h = min(width * 0.7, 7), min(length*0.35, 3.0), 2.5             
            if sofa_max_w > 2.0 and sofa_max_d > 1.5:
                sofa_actual_w, sofa_actual_d = sofa_max_w, sofa_max_d
                sofa_x_pos, sofa_y_pos = center_x, y_min + sofa_actual_d/2 + 0.5 
                
                if width > length * 1.1: 
                    sofa_actual_w = sofa_max_w; sofa_actual_d = sofa_max_d 
                    sofa_x_pos = center_x; sofa_y_pos = y_min + sofa_actual_d/2 + 0.5 
                elif length > width * 1.1: 
                    sofa_actual_w = sofa_max_d; sofa_actual_d = sofa_max_w 
                    sofa_x_pos = x_min + sofa_actual_d/2 + 0.5; sofa_y_pos = center_y
                
                sofa_bounds = [sofa_x_pos - sofa_actual_w/2, sofa_x_pos + sofa_actual_w/2,
                               sofa_y_pos - sofa_actual_d/2, sofa_y_pos + sofa_actual_d/2,
                               0, sofa_h]
                plotter.add_mesh(pv.Box(bounds=sofa_bounds), color=furniture_color["sofa"])
    
    def generate_3d_model(self):
        if not self.room_dimensions and not self.walls and not self.curved_walls:
            messagebox.showerror("Error", "No data to generate a model. Process an image or add rooms/walls.")
            return
        try:
            current_height_ft = float(self.height_var.get()); 
            current_wall_thickness_ft = float(self.thickness_var.get())
            current_font_size = int(self.font_size_var.get()); 
            show_labels_flag = self.show_labels_var.get()
        except ValueError:
            messagebox.showerror("Input Error", "Room height, wall thickness, and font size must be valid numbers.")
            return
        if current_height_ft <=0 or current_wall_thickness_ft <=0:
            messagebox.showerror("Input Error", "Height and thickness must be positive.")
            return

        plotter = pv.Plotter(window_size=[1000,800], lighting='three lights') 
        plotter.background_color = "#F0F0F0" 
        plotter.enable_shadows()
        plotter.enable_ssao(radius=max(current_height_ft * 0.2, 0.5) , bias=0.01, kernel_size=256)


        all_points_ft = []
        if self.scale_factor > 0:
            for wall_data in self.walls:
                s_px, e_px = np.array(wall_data["start"]), np.array(wall_data["end"])
                all_points_ft.append(s_px / self.scale_factor)
                all_points_ft.append(e_px / self.scale_factor)
            for curve_data in self.curved_walls:
                for pt_px in curve_data["points"]:
                    all_points_ft.append(np.array(pt_px) / self.scale_factor)
        
        if not all_points_ft and self.room_dimensions:
             for room_name, data in self.room_dimensions.items():
                if "position" in data and "width" in data and "length" in data:
                    cx, cy = data["position"]
                    w, l = data["width"], data["length"]
                    all_points_ft.append((cx - w/2, cy - l/2))
                    all_points_ft.append((cx + w/2, cy + l/2))
        
        if not all_points_ft: 
             plotter.add_mesh(pv.Plane(center=(0,0,-0.1), direction=(0,0,1), i_size=50, j_size=50),
                              color=self.materials["floor"])
        else:
            all_points_ft_np = np.array(all_points_ft)
            min_coord_x = np.min(all_points_ft_np[:,0])
            max_coord_x = np.max(all_points_ft_np[:,0])
            min_coord_y = np.min(all_points_ft_np[:,1])
            max_coord_y = np.max(all_points_ft_np[:,1])

            floor_padding = max(5.0, current_height_ft * 0.5) 
            floor_center_x = (min_coord_x + max_coord_x) / 2
            floor_center_y = (min_coord_y + max_coord_y) / 2
            floor_i_size = (max_coord_x - min_coord_x) + 2 * floor_padding
            floor_j_size = (max_coord_y - min_coord_y) + 2 * floor_padding

            plotter.add_mesh(pv.Plane(center=(floor_center_x, floor_center_y, -0.05), direction=(0,0,1), 
                                      i_size=floor_i_size, j_size=floor_j_size),
                              color=self.materials["floor"])
            plotter.add_mesh(pv.Plane(center=(floor_center_x, floor_center_y, current_height_ft + 0.05), direction=(0,0,-1),
                                      i_size=floor_i_size, j_size=floor_j_size),
                              color=self.materials["ceiling"], opacity=0.7)


        for room_name, data in self.room_dimensions.items():
            if "position" in data and "width" in data and "length" in data:
                width_ft = data["width"]; length_ft = data["length"]
                center_x_ft, center_y_ft = data["position"]
                if width_ft <=0 or length_ft <=0: continue
                
                if show_labels_flag: 
                    plotter.add_point_labels([(center_x_ft, center_y_ft, current_height_ft / 2)], [room_name], 
                                            font_size=current_font_size, text_color="#000000", shape=None, show_points=False,
                                            always_visible=False, point_size=10) 
            
                r_min_x = center_x_ft - width_ft / 2.0; r_max_x = center_x_ft + width_ft / 2.0
                r_min_y = center_y_ft - length_ft / 2.0; r_max_y = center_y_ft + length_ft / 2.0
                room_bounds_ft_for_furniture = (r_min_x, r_max_x, r_min_y, r_max_y)
                if width_ft * length_ft > 10: 
                    self.create_furniture(plotter, data.get("type", "Other"), room_bounds_ft_for_furniture, current_height_ft)

        for wall_data_px in self.walls: 
            if self.scale_factor > 0: 
                self.create_wall_with_openings(plotter, wall_data_px, current_height_ft, 
                                            current_wall_thickness_ft, self.scale_factor)
        
        for curve_data_px in self.curved_walls:
            if self.scale_factor > 0:
                points_ft = [(p[0] / self.scale_factor, p[1] / self.scale_factor) for p in curve_data_px["points"]]
                self.create_curved_wall(plotter, points_ft, current_height_ft, current_wall_thickness_ft)
        
        plotter.show_axes_all()
        plotter.camera_position = 'iso' 
        plotter.camera.elevation = 35  
        plotter.camera.azimuth = -45   
        plotter.camera.zoom(1.2)       
        plotter.enable_parallel_projection() 
        plotter.show(title="3D Floor Plan Model", auto_close=False) 

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    pv.set_plot_theme("document") 
    root = tk.Tk()
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc") 
    except ImportError:
        print("ttkthemes not found, using default Tk theme.")
        pass 
    app = FloorPlanConverter(root)
    app.run()

