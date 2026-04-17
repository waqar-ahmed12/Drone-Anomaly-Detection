import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ContinuousGridSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Continuous Patch Sorter")
        self.root.geometry("1000x900")
        
        # --- Settings ---
        self.rows = 3
        self.cols = 4
        self.num_slots = self.rows * self.cols
        self.thumb_size = 200  # Size of images in the grid

        # --- Variables ---
        self.image_queue = []          # Holds all pending images
        self.current_display = [None] * self.num_slots  # Tracks what is in the 12 slots
        self.tk_images = [None] * self.num_slots        # Keeps images in memory
        self.grid_labels = []          # The Tkinter Label widgets

        # Mode Selection Variable
        self.is_non_field_mode = tk.BooleanVar()
        self.is_non_field_mode.set(False) # False = Field, True = Non-Field

        # --- UI Layout ---
        self.top_frame = tk.Frame(root, pady=10)
        self.top_frame.pack(fill=tk.X)

        self.info_label = tk.Label(self.top_frame, text="Select your patches folder to begin.", font=("Arial", 14))
        self.info_label.pack()

        self.btn_frame = tk.Frame(self.top_frame)
        self.btn_frame.pack(pady=10)

        self.load_btn = tk.Button(self.btn_frame, text="Load Folder", command=self.load_folder, font=("Arial", 12))
        self.load_btn.pack(side=tk.LEFT, padx=10)

        # The Mode Checkbox (Hidden until folder is loaded)
        self.mode_cb = tk.Checkbutton(
            self.btn_frame, 
            text="Assign single clicks to: FIELD", 
            variable=self.is_non_field_mode,
            command=self.update_mode_ui,
            font=("Arial", 14, "bold"),
            fg="green",
            indicatoron=True
        )

        # Bulk Action Buttons (Hidden until folder is loaded)
        self.btn_all_fields = tk.Button(self.btn_frame, text="All visible are Fields", command=self.mark_all_visible_fields, font=("Arial", 12), bg="lightgreen")
        self.btn_all_non_fields = tk.Button(self.btn_frame, text="All visible are Non-Fields", command=self.mark_all_visible_non_fields, font=("Arial", 12), bg="salmon")

        # Grid frame
        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # Initialize the fixed grid of labels
        self.setup_grid()

    def setup_grid(self):
        """Creates the 12 empty slots for images to drop into."""
        for i in range(self.num_slots):
            row = i // self.cols
            col = i % self.cols
            
            # Create a label with a background color (acts as a border)
            lbl = tk.Label(self.grid_frame, bd=5, relief="solid", bg="green", cursor="hand2")
            lbl.grid(row=row, column=col, padx=10, pady=10)
            
            # Bind the left mouse click to process the image in this specific slot
            lbl.bind("<Button-1>", lambda e, idx=i: self.process_single_image(idx))
            
            # Hide them initially
            lbl.grid_remove()
            self.grid_labels.append(lbl)

    def load_folder(self):
        self.input_dir = filedialog.askdirectory(title="Select Folder Containing Patches")
        if not self.input_dir:
            return

        # Setup target directories
        self.non_field_dir = os.path.join(self.input_dir, "non_field")
        self.field_dir = os.path.join(self.input_dir, "field")
        os.makedirs(self.non_field_dir, exist_ok=True)
        os.makedirs(self.field_dir, exist_ok=True)

        # Get all valid images
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.image_queue = [
            os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir)
            if f.lower().endswith(valid_extensions)
        ]

        if not self.image_queue:
            messagebox.showinfo("Empty", "No images found in that folder!")
            return

        # UI Updates
        self.load_btn.pack_forget()
        self.mode_cb.pack(side=tk.LEFT, padx=10)
        
        # Add some spacing between the checkbox and bulk buttons
        tk.Label(self.btn_frame, text=" | ").pack(side=tk.LEFT, padx=5)
        
        self.btn_all_fields.pack(side=tk.LEFT, padx=5)
        self.btn_all_non_fields.pack(side=tk.LEFT, padx=5)
        
        # Fill the initial 12 slots
        for i in range(self.num_slots):
            if self.image_queue:
                self.current_display[i] = self.image_queue.pop(0)
                self.update_slot_ui(i)
            else:
                self.current_display[i] = None

        self.update_info_label()
        self.update_mode_ui()

    def update_slot_ui(self, slot_index):
        """Loads the image into a specific grid slot and makes it visible."""
        path = self.current_display[slot_index]
        if path:
            img = Image.open(path)
            img.thumbnail((self.thumb_size, self.thumb_size))
            tk_img = ImageTk.PhotoImage(img)
            
            self.tk_images[slot_index] = tk_img
            self.grid_labels[slot_index].config(image=tk_img)
            self.grid_labels[slot_index].grid() # Show the label

    def process_single_image(self, slot_index):
        """Moves a single clicked image and replaces it with the next one."""
        img_path = self.current_display[slot_index]
        if not img_path:
            return

        # Determine destination based on Checkbox state
        filename = os.path.basename(img_path)
        if self.is_non_field_mode.get():
            destination = os.path.join(self.non_field_dir, filename)
        else:
            destination = os.path.join(self.field_dir, filename)

        self._move_file(img_path, destination)

        # Drop the next image from the queue into this slot
        if self.image_queue:
            new_path = self.image_queue.pop(0)
            self.current_display[slot_index] = new_path
            self.update_slot_ui(slot_index)
        else:
            # If the queue is empty, clear this slot and hide the label
            self.current_display[slot_index] = None
            self.grid_labels[slot_index].grid_remove()

        self.update_info_label()

    def mark_all_visible_fields(self):
        """Moves all currently visible images to the field folder and refills the grid."""
        self._process_bulk(self.field_dir)

    def mark_all_visible_non_fields(self):
        """Moves all currently visible images to the non_field folder and refills the grid."""
        self._process_bulk(self.non_field_dir)

    def _process_bulk(self, target_dir):
        """Helper method to handle moving all visible slots to a specific folder."""
        # 1. Move all currently visible images
        for i in range(self.num_slots):
            img_path = self.current_display[i]
            if img_path:
                filename = os.path.basename(img_path)
                destination = os.path.join(target_dir, filename)
                self._move_file(img_path, destination)
                self.current_display[i] = None # Clear slot

        # 2. Refill the entire grid from the queue
        for i in range(self.num_slots):
            if self.image_queue:
                new_path = self.image_queue.pop(0)
                self.current_display[i] = new_path
                self.update_slot_ui(i)
            else:
                self.grid_labels[i].grid_remove() # Hide if no more images

        self.update_info_label()

    def _move_file(self, src, dest):
        """Safely moves a file."""
        try:
            shutil.move(src, dest)
        except Exception as e:
            print(f"Error moving {os.path.basename(src)}: {e}")

    def update_mode_ui(self):
        """Updates text and border colors so you always know what a single click will do."""
        if self.is_non_field_mode.get():
            self.mode_cb.config(text="Assign single clicks to: NON-FIELD", fg="red")
            for lbl in self.grid_labels:
                lbl.config(bg="red")
        else:
            self.mode_cb.config(text="Assign single clicks to: FIELD", fg="green")
            for lbl in self.grid_labels:
                lbl.config(bg="green")

    def update_info_label(self):
        """Shows remaining images."""
        images_left = len(self.image_queue) + sum(1 for path in self.current_display if path is not None)
        
        if images_left == 0:
            self.info_label.config(text="All done! No more images to sort.")
            self.mode_cb.pack_forget()
            self.btn_all_fields.pack_forget()
            self.btn_all_non_fields.pack_forget()
        else:
            self.info_label.config(text=f"Images remaining: {images_left}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ContinuousGridSorterApp(root)
    root.focus_force()
    root.mainloop()