"""
Cataract Detection GUI using trained TensorFlow model
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

class CataractDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cataract Detection System")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Model settings
        self.model_path = "cataract_detector_final.h5"
        self.img_size = 224
        self.class_names = ['Cataract', 'Normal']
        self.model = None
        self.current_image_path = None
        
        # Load model
        self.load_model()
        
        # Setup GUI
        self.setup_gui()
    
    def load_model(self):
        """Load trained model"""
        try:
            if not os.path.exists(self.model_path):
                messagebox.showerror("Error", f"Model file '{self.model_path}' not found!\n\nPlease train the model first.")
                self.root.destroy()
                return
            
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.root.destroy()
    
    def setup_gui(self):
        """Setup GUI components"""
        
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🔬 Cataract Detection System",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(expand=True)
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#ecf0f1")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left side - Image display
        left_frame = tk.Frame(main_frame, bg="#ecf0f1")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Image canvas
        self.image_frame = tk.Frame(left_frame, bg="white", width=400, height=400, relief=tk.SUNKEN, bd=2)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame, text="No image loaded", bg="white", fg="gray")
        self.image_label.pack(expand=True)
        
        # Buttons
        button_frame = tk.Frame(left_frame, bg="#ecf0f1")
        button_frame.pack(pady=10)
        
        self.select_btn = tk.Button(
            button_frame,
            text="📁 Select Image",
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self.select_image
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.predict_btn = tk.Button(
            button_frame,
            text="🔍 Predict",
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self.predict_image,
            state=tk.DISABLED
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Right side - Results
        right_frame = tk.Frame(main_frame, bg="#ecf0f1", width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(20, 0))
        right_frame.pack_propagate(False)
        
        result_title = tk.Label(
            right_frame,
            text="Prediction Results",
            font=("Arial", 16, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        result_title.pack(pady=(0, 20))
        
        # Result display frame
        self.result_frame = tk.Frame(right_frame, bg="white", relief=tk.RAISED, bd=2)
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.result_label = tk.Label(
            self.result_frame,
            text="Please select an image\nand click Predict",
            font=("Arial", 12),
            bg="white",
            fg="gray",
            justify=tk.CENTER
        )
        self.result_label.pack(expand=True)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready | Model loaded successfully",
            font=("Arial", 9),
            bg="#34495e",
            fg="white",
            anchor=tk.W,
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_image(self):
        """Open file dialog to select image"""
        file_path = filedialog.askopenfilename(
            title="Select Eye Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Image loaded: {os.path.basename(file_path)}")
            
            # Clear previous results
            self.clear_results()
    
    def display_image(self, image_path):
        """Display selected image"""
        try:
            # Load and resize image for display
            img = Image.open(image_path)
            
            # Resize to fit frame while maintaining aspect ratio
            img.thumbnail((380, 380), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def clear_results(self):
        """Clear previous prediction results"""
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        self.result_label = tk.Label(
            self.result_frame,
            text="Ready to predict...",
            font=("Arial", 12),
            bg="white",
            fg="gray"
        )
        self.result_label.pack(expand=True)
    
    def predict_image(self):
        """Predict the selected image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        try:
            # Update status
            self.status_bar.config(text="Predicting...")
            self.root.update()
            
            # Load and preprocess image
            img = Image.open(self.current_image_path)
            
            # Convert to RGB if image has alpha channel (PNG) or is grayscale
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize((self.img_size, self.img_size))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # Predict
            prediction_prob = self.model.predict(img_array, verbose=0)[0][0]
            
            # Interpret result
            if prediction_prob > 0.5:
                prediction = self.class_names[1]  # Normal
                confidence = prediction_prob
                result_color = "#27ae60"  # Green
            else:
                prediction = self.class_names[0]  # Cataract
                confidence = 1 - prediction_prob
                result_color = "#e74c3c"  # Red
            
            # Display results
            self.display_results(prediction, confidence, prediction_prob, result_color)
            
            # Update status
            self.status_bar.config(text=f"Prediction complete: {prediction} ({confidence*100:.1f}%)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
            self.status_bar.config(text="Prediction failed")
    
    def display_results(self, prediction, confidence, prob, color):
        """Display prediction results"""
        # Clear previous results
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Prediction label
        pred_frame = tk.Frame(self.result_frame, bg="white")
        pred_frame.pack(pady=20)
        
        tk.Label(
            pred_frame,
            text="Diagnosis:",
            font=("Arial", 12),
            bg="white",
            fg="#7f8c8d"
        ).pack()
        
        tk.Label(
            pred_frame,
            text=prediction.upper(),
            font=("Arial", 28, "bold"),
            bg="white",
            fg=color
        ).pack(pady=5)
        
        # Confidence
        conf_frame = tk.Frame(self.result_frame, bg="white")
        conf_frame.pack(pady=10)
        
        tk.Label(
            conf_frame,
            text="Confidence:",
            font=("Arial", 11),
            bg="white",
            fg="#7f8c8d"
        ).pack()
        
        tk.Label(
            conf_frame,
            text=f"{confidence*100:.1f}%",
            font=("Arial", 24, "bold"),
            bg="white",
            fg="#2c3e50"
        ).pack()
        
        # Probability bars
        prob_frame = tk.Frame(self.result_frame, bg="white")
        prob_frame.pack(pady=20, padx=20, fill=tk.X)
        
        tk.Label(
            prob_frame,
            text="Probability Distribution:",
            font=("Arial", 10, "bold"),
            bg="white",
            fg="#2c3e50"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Cataract bar
        cataract_prob = 1 - prob
        print(f"DEBUG: Cataract probability = {cataract_prob:.4f} ({cataract_prob*100:.1f}%)")
        self.create_probability_bar(
            prob_frame, 
            "Cataract", 
            cataract_prob, 
            "#e74c3c"
        )
        
        # Normal bar
        print(f"DEBUG: Normal probability = {prob:.4f} ({prob*100:.1f}%)")
        self.create_probability_bar(
            prob_frame, 
            "Normal", 
            prob, 
            "#27ae60"
        )
        
        # Warning if needed
        if prediction == "Cataract":
            warning_frame = tk.Frame(self.result_frame, bg="#fff3cd", relief=tk.RAISED, bd=1)
            warning_frame.pack(pady=10, padx=20, fill=tk.X)
            
            tk.Label(
                warning_frame,
                text="⚠️ Cataract Detected",
                font=("Arial", 10, "bold"),
                bg="#fff3cd",
                fg="#856404"
            ).pack(pady=1)
            
            tk.Label(
                warning_frame,
                text="Please consult an ophthalmologist\nfor professional diagnosis",
                font=("Arial", 9),
                bg="#fff3cd",
                fg="#856404",
                justify=tk.CENTER,
                wraplength=400
            ).pack(pady=(0, 1), padx=10)
    
    def create_probability_bar(self, parent, label, value, color):
        """Create a probability bar visualization"""
        bar_frame = tk.Frame(parent, bg="white")
        bar_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Configure grid columns - kolom 2 akan expand
        bar_frame.grid_columnconfigure(1, weight=1)
        
        # Label (kolom 0)
        tk.Label(
            bar_frame,
            text=f"{label}:",
            font=("Arial", 10),
            bg="white",
            fg="#2c3e50",
            width=10,
            anchor=tk.W
        ).grid(row=0, column=0, sticky=tk.W)
        
        # Bar background (kolom 1)
        bar_bg = tk.Frame(bar_frame, bg="#ecf0f1", height=20, width=100)
        bar_bg.grid(row=0, column=1, padx=5, sticky=tk.W)
        bar_bg.grid_propagate(False)
        
        # Bar fill - gunakan round untuk akurasi lebih baik
        bar_width = round(100 * value)
        print(f"DEBUG: Bar '{label}' - value={value:.4f}, bar_width={bar_width}/100")
        if bar_width > 0:
            bar_fill = tk.Frame(bar_bg, bg=color, height=20, width=bar_width)
            bar_fill.place(x=0, y=0)
        
        # Percentage (kolom 2)
        tk.Label(
            bar_frame,
            text=f"{value*100:.1f}%",
            font=("Arial", 10, "bold"),
            bg="white",
            fg="#2c3e50"
        ).grid(row=0, column=2, padx=10, sticky=tk.W)

def main():
    root = tk.Tk()
    app = CataractDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
