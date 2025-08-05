"""
Trajectory Browser GUI Application

A browser-like interface for viewing RoboDM trajectories with tabbed navigation.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

# Add the parent robodm package to path
current_dir = Path(__file__).parent
robodm_root = current_dir.parent
sys.path.insert(0, str(robodm_root))

try:
    import robodm
    from robodm import Trajectory
except ImportError:
    print("RoboDM not found. Please install robodm package.")
    sys.exit(1)


class TrajectoryTab:
    """Represents a single trajectory tab in the browser."""
    
    def __init__(self, parent, trajectory_path: str, trajectory_id: str):
        self.parent = parent
        self.trajectory_path = trajectory_path
        self.trajectory_id = trajectory_id
        self.trajectory = None
        self.data = None
        self.current_frame = 0
        self.total_frames = 0
        self.feature_names = []
        self.image_features = []
        self.numeric_features = []
        
        # Create the tab content
        self.frame = ttk.Frame(parent)
        self._create_widgets()
        self._load_trajectory()
    
    def _create_widgets(self):
        """Create the widgets for this tab."""
        # Main container with scrollbar
        main_container = ttk.Frame(self.frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top control panel
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Trajectory info
        info_frame = ttk.LabelFrame(control_frame, text="Trajectory Info", padding=5)
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.info_label = ttk.Label(info_frame, text=f"Loading {self.trajectory_id}...")
        self.info_label.pack(anchor=tk.W)
        
        # Playback controls
        playback_frame = ttk.LabelFrame(control_frame, text="Playback Controls", padding=5)
        playback_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Frame slider
        slider_frame = ttk.Frame(playback_frame)
        slider_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT)
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, 
                                    command=self._on_frame_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        self.frame_label = ttk.Label(slider_frame, text="0/0")
        self.frame_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Playback buttons
        button_frame = ttk.Frame(playback_frame)
        button_frame.pack(fill=tk.X)
        
        self.play_button = ttk.Button(button_frame, text="▶", width=3, command=self._toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=(0, 2))
        
        self.stop_button = ttk.Button(button_frame, text="⏹", width=3, command=self._stop_playback)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        
        self.prev_button = ttk.Button(button_frame, text="⏮", width=3, command=self._prev_frame)
        self.prev_button.pack(side=tk.LEFT, padx=2)
        
        self.next_button = ttk.Button(button_frame, text="⏭", width=3, command=self._next_frame)
        self.next_button.pack(side=tk.LEFT, padx=2)
        
        # Speed control
        ttk.Label(button_frame, text="Speed:").pack(side=tk.LEFT, padx=(10, 2))
        self.speed_var = tk.StringVar(value="1x")
        speed_combo = ttk.Combobox(button_frame, textvariable=self.speed_var, 
                                  values=["0.25x", "0.5x", "1x", "2x", "4x"], 
                                  width=5, state="readonly")
        speed_combo.pack(side=tk.LEFT)
        
        # Main content area with notebook
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Image viewer tab
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Images")
        
        # Create canvas for image display
        self.image_canvas = tk.Canvas(self.image_frame, bg="black")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Data viewer tab
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data")
        
        # Create treeview for data display
        self.data_tree = ttk.Treeview(self.data_frame, columns=("value", "shape", "dtype"), show="tree headings")
        self.data_tree.heading("#0", text="Feature")
        self.data_tree.heading("value", text="Value")
        self.data_tree.heading("shape", text="Shape")
        self.data_tree.heading("dtype", text="Type")
        
        # Add scrollbars
        data_scroll_y = ttk.Scrollbar(self.data_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        data_scroll_x = ttk.Scrollbar(self.data_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=data_scroll_y.set, xscrollcommand=data_scroll_x.set)
        
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        data_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Metadata tab
        self.metadata_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metadata_frame, text="Metadata")
        
        self.metadata_text = scrolledtext.ScrolledText(self.metadata_frame, wrap=tk.WORD)
        self.metadata_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize playback state
        self.is_playing = False
        self.playback_speed = 1.0
        self.animation = None
    
    def _load_trajectory(self):
        """Load the trajectory data."""
        try:
            self.trajectory = Trajectory(self.trajectory_path, mode="r")
            self.data = self.trajectory.load()
            
            if not isinstance(self.data, dict):
                raise ValueError("Trajectory data is not in expected format")
            
            self.feature_names = list(self.data.keys())
            self.total_frames = self._get_trajectory_length()
            
            # Categorize features
            for feature_name in self.feature_names:
                if self._is_image_feature(feature_name):
                    self.image_features.append(feature_name)
                else:
                    self.numeric_features.append(feature_name)
            
            # Update UI
            self.frame_slider.configure(to=max(1, self.total_frames))
            self.frame_label.configure(text=f"1/{self.total_frames}")
            self.info_label.configure(
                text=f"{self.trajectory_id} | {self.total_frames} frames | "
                     f"{len(self.image_features)} image features | "
                     f"{len(self.numeric_features)} data features"
            )
            
            # Load initial frame
            self._update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load trajectory: {e}")
            self.info_label.configure(text=f"Error loading {self.trajectory_id}")
    
    def _get_trajectory_length(self) -> int:
        """Get the length of the trajectory."""
        if not self.data:
            return 0
        
        # Find the length of the first feature
        first_feature = next(iter(self.data.values()))
        if hasattr(first_feature, '__len__'):
            return len(first_feature)
        return 0
    
    def _is_image_feature(self, feature_name: str) -> bool:
        """Check if a feature contains image data."""
        if feature_name not in self.data:
            return False
        
        data = self.data[feature_name]
        if not hasattr(data, '__len__') or len(data) == 0:
            return False
        
        # Check if the first element looks like an image
        first_item = data[0]
        if isinstance(first_item, np.ndarray):
            return len(first_item.shape) >= 2  # At least 2D array
        return False
    
    def _on_frame_change(self, value):
        """Handle frame slider change."""
        try:
            self.current_frame = int(float(value))
            self._update_display()
        except ValueError:
            pass
    
    def _update_display(self):
        """Update the display for the current frame."""
        if not self.data or self.current_frame >= self.total_frames:
            return
        
        # Update frame label
        self.frame_label.configure(text=f"{self.current_frame + 1}/{self.total_frames}")
        
        # Update image display
        self._update_image_display()
        
        # Update data display
        self._update_data_display()
        
        # Update metadata
        self._update_metadata_display()
    
    def _update_image_display(self):
        """Update the image display for the current frame."""
        if not self.image_features:
            return
        
        # Find the first image feature to display
        image_feature = self.image_features[0]
        if image_feature in self.data:
            image_data = self.data[image_feature][self.current_frame]
            
            if isinstance(image_data, np.ndarray):
                # Convert to PIL Image
                if image_data.dtype != np.uint8:
                    # Normalize to 0-255 range
                    if image_data.max() > 0:
                        image_data = ((image_data - image_data.min()) / 
                                    (image_data.max() - image_data.min()) * 255).astype(np.uint8)
                    else:
                        image_data = image_data.astype(np.uint8)
                
                # Handle different image formats
                if len(image_data.shape) == 2:
                    # Grayscale
                    pil_image = Image.fromarray(image_data, mode='L')
                elif len(image_data.shape) == 3:
                    if image_data.shape[2] == 3:
                        # RGB
                        pil_image = Image.fromarray(image_data, mode='RGB')
                    elif image_data.shape[2] == 4:
                        # RGBA
                        pil_image = Image.fromarray(image_data, mode='RGBA')
                    else:
                        # Other 3D - take first channel
                        pil_image = Image.fromarray(image_data[:, :, 0], mode='L')
                else:
                    # Higher dimensional - take first slice
                    pil_image = Image.fromarray(image_data[0], mode='L')
                
                # Resize to fit canvas
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    # Calculate aspect ratio preserving resize
                    img_width, img_height = pil_image.size
                    scale = min(canvas_width / img_width, canvas_height / img_height)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    
                    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage and display
                self.photo_image = ImageTk.PhotoImage(pil_image)
                self.image_canvas.delete("all")
                
                # Center the image
                x = (canvas_width - self.photo_image.width()) // 2
                y = (canvas_height - self.photo_image.height()) // 2
                self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.photo_image)
    
    def _update_data_display(self):
        """Update the data tree view for the current frame."""
        # Clear existing items
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if not self.data:
            return
        
        # Add data for current frame
        for feature_name in self.feature_names:
            if feature_name in self.data:
                data = self.data[feature_name]
                if hasattr(data, '__len__') and len(data) > self.current_frame:
                    frame_data = data[self.current_frame]
                    
                    # Format the value
                    if isinstance(frame_data, np.ndarray):
                        value_str = f"Array {frame_data.shape}"
                        shape_str = str(frame_data.shape)
                        dtype_str = str(frame_data.dtype)
                    else:
                        value_str = str(frame_data)
                        shape_str = "scalar"
                        dtype_str = type(frame_data).__name__
                    
                    self.data_tree.insert("", tk.END, text=feature_name, 
                                        values=(value_str, shape_str, dtype_str))
    
    def _update_metadata_display(self):
        """Update the metadata display."""
        self.metadata_text.delete(1.0, tk.END)
        
        metadata = {
            "Trajectory ID": self.trajectory_id,
            "File Path": self.trajectory_path,
            "Total Frames": self.total_frames,
            "Current Frame": self.current_frame + 1,
            "Image Features": self.image_features,
            "Numeric Features": self.numeric_features,
            "All Features": self.feature_names
        }
        
        for key, value in metadata.items():
            self.metadata_text.insert(tk.END, f"{key}: {value}\n")
    
    def _toggle_playback(self):
        """Toggle playback animation."""
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()
    
    def _start_playback(self):
        """Start playback animation."""
        if self.total_frames <= 1:
            return
        
        self.is_playing = True
        self.play_button.configure(text="⏸")
        
        # Get speed multiplier
        speed_text = self.speed_var.get()
        speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
        self.playback_speed = speed_map.get(speed_text, 1.0)
        
        # Calculate interval in milliseconds
        interval = int(1000 / (10 * self.playback_speed))  # 10 FPS base
        
        def animate():
            if self.is_playing:
                self._next_frame()
                if self.current_frame >= self.total_frames - 1:
                    self._stop_playback()
                else:
                    self.frame.after(interval, animate)
        
        animate()
    
    def _stop_playback(self):
        """Stop playback animation."""
        self.is_playing = False
        self.play_button.configure(text="▶")
    
    def _prev_frame(self):
        """Go to previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.set(self.current_frame)
            self._update_display()
    
    def _next_frame(self):
        """Go to next frame."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.set(self.current_frame)
            self._update_display()
    
    def close(self):
        """Close the trajectory and clean up resources."""
        if self.trajectory:
            try:
                self.trajectory.close()
            except:
                pass
        self._stop_playback()


class TrajectoryBrowser:
    """Main trajectory browser application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RoboDM Trajectory Browser")
        self.root.geometry("1200x800")
        
        # State variables
        self.trajectory_tabs = {}
        self.current_data_path = None
        
        # Create the GUI
        self._create_widgets()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_widgets(self):
        """Create the main GUI widgets."""
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Top toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # New tab button (like browser + button)
        self.new_tab_button = ttk.Button(toolbar, text="+", width=3, command=self._open_new_tab)
        self.new_tab_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Separator
        separator = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Data path selection (for reference)
        ttk.Label(toolbar, text="Last Directory:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(toolbar, textvariable=self.path_var, width=30, state="readonly")
        self.path_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(toolbar, text="Browse", command=self._browse_directory).pack(side=tk.LEFT, padx=(0, 5))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Click + to open a trajectory")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, sticky="ew", padx=5, pady=(0, 5))
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        
        # Bind tab close events
        self.notebook.bind("<Button-2>", self._on_tab_right_click)  # Right click
        self.notebook.bind("<Button-3>", self._on_tab_right_click)  # Right click (alternative)
        
        # Create initial empty tab
        self._create_welcome_tab()
    
    def _create_welcome_tab(self):
        """Create a welcome tab when no trajectories are loaded."""
        welcome_frame = ttk.Frame(self.notebook)
        self.notebook.add(welcome_frame, text="Welcome")
        
        # Welcome message
        welcome_text = """
        Welcome to RoboDM Trajectory Browser!
        
        To get started:
        1. Click the '+' button to open a trajectory file
        2. Or use 'Browse' to set a default directory
        
        Features:
        • View trajectory data in organized tabs
        • Playback image sequences with controls
        • Inspect trajectory metadata
        • Browse data features in tree view
        
        Right-click on tabs to close them.
        """
        
        welcome_label = ttk.Label(welcome_frame, text=welcome_text, justify=tk.LEFT, font=("Arial", 12))
        welcome_label.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)
    
    def _open_new_tab(self):
        """Open a new trajectory tab by selecting a file."""
        file_path = filedialog.askopenfilename(
            title="Select Trajectory File",
            filetypes=[("VLA files", "*.vla"), ("All files", "*.*")]
        )
        
        if file_path:
            # Update the path display
            directory = os.path.dirname(file_path)
            self.path_var.set(directory)
            self.current_data_path = directory
            
            # Create the trajectory tab
            trajectory_id = os.path.basename(file_path).replace('.vla', '')
            self._create_trajectory_tab(file_path, trajectory_id)
            
            # Update status
            self.status_var.set(f"Opened: {trajectory_id}")
    
    def _browse_directory(self):
        """Open directory browser dialog to set default directory."""
        directory = filedialog.askdirectory(title="Select Default Trajectory Directory")
        if directory:
            self.path_var.set(directory)
            self.current_data_path = directory
            self.status_var.set(f"Default directory set: {os.path.basename(directory)}")
    
    def _create_trajectory_tab(self, file_path: str, trajectory_id: str):
        """Create a new trajectory tab."""
        try:
            tab = TrajectoryTab(self.notebook, file_path, trajectory_id)
            self.trajectory_tabs[trajectory_id] = tab
            
            # Add tab to notebook
            self.notebook.add(tab.frame, text=trajectory_id)
            
            # Switch to the new tab
            self.notebook.select(tab.frame)
            
            # Remove welcome tab if it's the only other tab
            if len(self.notebook.tabs()) == 2:  # Welcome + new tab
                welcome_tab = None
                for tab_id in self.notebook.tabs():
                    if self.notebook.tab(tab_id, "text") == "Welcome":
                        welcome_tab = tab_id
                        break
                if welcome_tab:
                    self.notebook.forget(welcome_tab)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create tab for {trajectory_id}: {e}")
    
    def _close_all_tabs(self):
        """Close all trajectory tabs."""
        for tab in self.trajectory_tabs.values():
            tab.close()
        
        self.trajectory_tabs.clear()
        
        # Remove all tabs from notebook
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)
        
        # Recreate welcome tab
        self._create_welcome_tab()
    
    def _on_tab_right_click(self, event):
        """Handle right-click on tab to close it."""
        # Find which tab was clicked
        tab_id = self.notebook.identify(event.x, event.y)
        if tab_id:
            # Get the tab index
            tab_index = self.notebook.index(tab_id)
            if tab_index >= 0:
                # Get the tab text (trajectory ID)
                tab_text = self.notebook.tab(tab_index, "text")
                if tab_text in self.trajectory_tabs:
                    self._close_trajectory_tab(tab_text)
    
    def _close_trajectory_tab(self, trajectory_id: str):
        """Close a specific trajectory tab."""
        if trajectory_id in self.trajectory_tabs:
            tab = self.trajectory_tabs[trajectory_id]
            tab.close()
            del self.trajectory_tabs[trajectory_id]
            
            # Remove from notebook
            self.notebook.forget(tab.frame)
            
            # If no tabs left, show welcome tab
            if len(self.notebook.tabs()) == 0:
                self._create_welcome_tab()
    
    def _on_closing(self):
        """Handle application closing."""
        self._close_all_tabs()
        self.root.destroy()
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = TrajectoryBrowser()
    app.run()


if __name__ == "__main__":
    main()

