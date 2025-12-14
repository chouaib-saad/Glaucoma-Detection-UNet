"""
GlaucomaFundus - Cup-to-Disc Ratio Assessment for Retinal Evaluation
A modern GUI application for automatic glaucoma detection using deep learning
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import sys
import os
import threading

# Add the Code directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from localization import OD_localization
from segmentation import optic_segmentation
from classification import inference_glaucoma
from supporting_function import ekstrakROI

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class GlaucomaDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("GlaucomaFundus - CDR Assessment for Retinal Evaluation")
        self.geometry("1400x850")
        self.minsize(1200, 750)
        
        # Initialize variables
        self.original_image = None
        self.current_image_path = None
        self.OD_mask = None
        self.OC_mask = None
        self.disc_center = None
        self.result_image = None
        
        # Models
        self.models_loaded = False
        self.ODFinder = None
        self.segModel = None
        self.glPredictor = None
        
        # Parameters
        self.ROI_SIZE = 550
        self.R_COEFF, self.G_COEFF, self.B_COEFF, self.BR_COEFF = 1, 0.2, 0, 0.8
        self.STD_SIZE = (2000, 2000)
        self.GLAUCOMA_CDR_THRESHOLD = 0.5
        
        self.create_ui()
        
    def create_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # SIDEBAR
        self.sidebar = ctk.CTkScrollableFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # Logo
        ctk.CTkLabel(self.sidebar, text="👁️ GlaucomaFundus", 
                     font=ctk.CTkFont(size=24, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 5))
        ctk.CTkLabel(self.sidebar, text="CDR Assessment Tool", font=ctk.CTkFont(size=12), 
                     text_color="gray").grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Buttons
        self.import_btn = ctk.CTkButton(self.sidebar, text="📁 Import Fundus Image", 
                                        command=self.import_image, height=45,
                                        font=ctk.CTkFont(size=14, weight="bold"))
        self.import_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.analyze_btn = ctk.CTkButton(self.sidebar, text="🔍 Analyze Image",
                                         command=self.analyze_image, height=45,
                                         font=ctk.CTkFont(size=14, weight="bold"),
                                         fg_color="#28a745", hover_color="#218838", state="disabled")
        self.analyze_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        # DIAGNOSIS SECTION
        self.diagnosis_frame = ctk.CTkFrame(self.sidebar, fg_color="gray20", corner_radius=15)
        self.diagnosis_frame.grid(row=4, column=0, padx=20, pady=20, sticky="ew")
        
        ctk.CTkLabel(self.diagnosis_frame, text="🏥 DIAGNOSIS RESULT",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=20, pady=(15, 10))
        
        self.diagnosis_label = ctk.CTkLabel(self.diagnosis_frame, text="AWAITING\nANALYSIS",
                                            font=ctk.CTkFont(size=24, weight="bold"),
                                            text_color="gray", justify="center")
        self.diagnosis_label.grid(row=1, column=0, padx=20, pady=(5, 10))
        
        self.diagnosis_indicator = ctk.CTkFrame(self.diagnosis_frame, height=8, 
                                                fg_color="gray40", corner_radius=4)
        self.diagnosis_indicator.grid(row=2, column=0, padx=30, pady=(5, 15), sticky="ew")
        
        # CDR Section
        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30").grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(self.sidebar, text="📊 Cup-to-Disc Ratios",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(row=6, column=0, padx=20, pady=(10, 10))
        
        self.results_frame = ctk.CTkFrame(self.sidebar, fg_color="gray20")
        self.results_frame.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        
        self.create_cdr_row(self.results_frame, "Vertical CDR:", "vcdr_value", "vcdr_bar", 0)
        self.create_cdr_row(self.results_frame, "Horizontal CDR:", "hcdr_value", "hcdr_bar", 1)
        self.create_cdr_row(self.results_frame, "Area CDR:", "acdr_value", "acdr_bar", 2)
        
        ctk.CTkLabel(self.sidebar, text=f"⚠️ CDR > {self.GLAUCOMA_CDR_THRESHOLD} = Glaucoma risk",
                     font=ctk.CTkFont(size=11), text_color="orange").grid(row=8, column=0, padx=20, pady=(5, 10))
        
        # Measurements Section
        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30").grid(row=9, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(self.sidebar, text="📏 Measurements",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(row=10, column=0, padx=20, pady=(10, 10))
        
        self.measurements_frame = ctk.CTkFrame(self.sidebar, fg_color="gray20")
        self.measurements_frame.grid(row=11, column=0, padx=20, pady=5, sticky="ew")
        
        self.create_result_row(self.measurements_frame, "Disc Diameter (V):", "disc_v_value", 0)
        self.create_result_row(self.measurements_frame, "Disc Diameter (H):", "disc_h_value", 1)
        self.create_result_row(self.measurements_frame, "Cup Diameter (V):", "cup_v_value", 2)
        self.create_result_row(self.measurements_frame, "Cup Diameter (H):", "cup_h_value", 3)
        self.create_result_row(self.measurements_frame, "Disc Area:", "disc_area_value", 4)
        self.create_result_row(self.measurements_frame, "Cup Area:", "cup_area_value", 5)
        self.create_result_row(self.measurements_frame, "Rim Area:", "rim_area_value", 6)
        
        # Progress & Status
        self.progress_bar = ctk.CTkProgressBar(self.sidebar, mode="indeterminate")
        self.progress_bar.grid(row=12, column=0, padx=20, pady=15, sticky="ew")
        self.progress_bar.grid_remove()
        
        self.status_label = ctk.CTkLabel(self.sidebar, text="Ready to analyze",
                                         font=ctk.CTkFont(size=11), text_color="gray")
        self.status_label.grid(row=13, column=0, padx=20, pady=5)
        
        # Theme selector
        theme_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        theme_frame.grid(row=14, column=0, padx=20, pady=(10, 20), sticky="ew")
        ctk.CTkLabel(theme_frame, text="Theme:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 10))
        ctk.CTkOptionMenu(theme_frame, values=["Dark", "Light", "System"],
                          command=lambda m: ctk.set_appearance_mode(m.lower()), width=100).pack(side="left")
        
        # MAIN CONTENT
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Header
        header = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        ctk.CTkLabel(header, text="Fundus Image Analysis",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")
        self.export_btn = ctk.CTkButton(header, text="💾 Export Results",
                                        command=self.export_results, width=120, state="disabled")
        self.export_btn.pack(side="right")
        
        # Image frames
        self.create_image_frame("Original Fundus Image", 0, "original")
        self.create_image_frame("Segmentation Result", 1, "segmented")
        
        # Info panel
        info_frame = ctk.CTkFrame(self.main_frame)
        info_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(15, 0))
        self.info_text = ctk.CTkTextbox(info_frame, height=80, font=ctk.CTkFont(size=12))
        self.info_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.info_text.insert("0.0", "Welcome to GlaucomaFundus - Automatic Glaucoma Detection\n\n"
                              "▸ Import fundus image → Analyze → Get GLAUCOMA / NO GLAUCOMA diagnosis\n"
                              "▸ CDR (Cup-to-Disc Ratio) > 0.5 indicates potential glaucoma risk")
        self.info_text.configure(state="disabled")

    def create_cdr_row(self, parent, label_text, value_name, bar_name, row):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, padx=10, pady=8, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(frame, text=label_text, font=ctk.CTkFont(size=12), anchor="w").grid(row=0, column=0, sticky="w")
        value_label = ctk.CTkLabel(frame, text="--", font=ctk.CTkFont(size=14, weight="bold"), text_color="#00b4d8")
        value_label.grid(row=0, column=1, sticky="e", padx=(10, 0))
        setattr(self, value_name, value_label)
        
        bar = ctk.CTkProgressBar(frame, height=6, progress_color="#00b4d8")
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        bar.set(0)
        setattr(self, bar_name, bar)
        
    def create_result_row(self, parent, label_text, value_name, row):
        ctk.CTkLabel(parent, text=label_text, font=ctk.CTkFont(size=11), anchor="w").grid(
            row=row, column=0, padx=(15, 5), pady=4, sticky="w")
        value_label = ctk.CTkLabel(parent, text="--", font=ctk.CTkFont(size=11, weight="bold"), text_color="#00b4d8")
        value_label.grid(row=row, column=1, padx=(5, 15), pady=4, sticky="e")
        setattr(self, value_name, value_label)
        
    def create_image_frame(self, title, column, frame_name):
        frame = ctk.CTkFrame(self.main_frame)
        frame.grid(row=1, column=column, sticky="nsew", padx=5, pady=5)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, pady=10)
        image_label = ctk.CTkLabel(frame, text="No image loaded", font=ctk.CTkFont(size=14), text_color="gray")
        image_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        setattr(self, f"{frame_name}_label", image_label)
        
    def import_image(self):
        file_path = filedialog.askopenfilename(title="Select Fundus Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"), ("All files", "*.*")])
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.original_image = cv2.imread(file_path)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.display_image(self.original_image, self.original_label)
                self.reset_results()
                self.analyze_btn.configure(state="normal")
                self.status_label.configure(text=f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def display_image(self, img_array, label_widget, max_size=480):
        img = Image.fromarray(img_array) if isinstance(img_array, np.ndarray) else img_array
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        label_widget.configure(image=ctk_image, text="")
        label_widget.image = ctk_image
        
    def reset_results(self):
        for attr in ['vcdr_value', 'hcdr_value', 'acdr_value', 'disc_v_value', 'disc_h_value', 
                     'cup_v_value', 'cup_h_value', 'disc_area_value', 'cup_area_value', 'rim_area_value']:
            getattr(self, attr).configure(text="--", text_color="#00b4d8")
        for bar in ['vcdr_bar', 'hcdr_bar', 'acdr_bar']:
            getattr(self, bar).set(0)
            getattr(self, bar).configure(progress_color="#00b4d8")
        self.diagnosis_label.configure(text="AWAITING\nANALYSIS", text_color="gray")
        self.diagnosis_indicator.configure(fg_color="gray40")
        self.segmented_label.configure(image=None, text="Segmentation result")
        self.export_btn.configure(state="disabled")
        
    def load_models(self):
        if not self.models_loaded:
            self.ODFinder = OD_localization()
            self.segModel = optic_segmentation()
            self.glPredictor = inference_glaucoma()
            self.models_loaded = True
            
    def calculate_adaptive_roi_size(self, img_shape):
        min_dim = min(img_shape[0], img_shape[1])
        return max(350, min(700, int(min_dim * 0.28)))
            
    def analyze_image(self):
        if self.original_image is None:
            return
        self.analyze_btn.configure(state="disabled")
        self.import_btn.configure(state="disabled")
        self.progress_bar.grid()
        self.progress_bar.start()
        threading.Thread(target=self.run_analysis).start()
        
    def run_analysis(self):
        try:
            self.after(0, lambda: self.status_label.configure(text="Loading models..."))
            self.load_models()
            
            self.after(0, lambda: self.status_label.configure(text="Preprocessing..."))
            ret_img = self.ODFinder.preprocessing(self.original_image.copy(), self.STD_SIZE)
            self.ROI_SIZE = self.calculate_adaptive_roi_size(ret_img.shape)
            
            self.after(0, lambda: self.status_label.configure(text="Localizing optic disc..."))
            self.disc_center = self.ODFinder.locate(ret_img, coeff_args=(self.R_COEFF, self.G_COEFF, self.B_COEFF, self.BR_COEFF))
            
            self.after(0, lambda: self.status_label.configure(text="Segmenting disc and cup..."))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl_img = clahe.apply(ret_img[:, :, 1])
            ROI, coordinate = ekstrakROI(self.disc_center, self.ROI_SIZE, cl_img)
            self.OD_mask, self.OC_mask = self.segModel.do_segmentation(ROI, coordinate, ret_img.shape[:2], 
                                                                        elips_fit=True, use_multi_scale=True)
            
            self.after(0, lambda: self.status_label.configure(text="Calculating CDR..."))
            measurements = self.calculate_measurements()
            VCDR, HCDR, ACDR = self.glPredictor.CDR_calc(self.OD_mask, self.OC_mask)
            is_glaucoma = self.determine_glaucoma(VCDR, HCDR, ACDR)
            
            self.after(0, lambda: self.update_results(measurements, VCDR, HCDR, ACDR, is_glaucoma, ret_img))
        except Exception as e:
            error_msg = str(e)  # Capture error message immediately
            self.after(0, lambda: self.show_error(error_msg))
            
    def determine_glaucoma(self, VCDR, HCDR, ACDR):
        if not np.isnan(VCDR) and VCDR > self.GLAUCOMA_CDR_THRESHOLD:
            return True
        if not np.isnan(ACDR) and ACDR > 0.55:
            return True
        if not np.isnan(VCDR) and not np.isnan(HCDR) and VCDR > 0.45 and HCDR > 0.45:
            return True
        return False
            
    def calculate_measurements(self):
        measurements = {'disc_h': 0, 'disc_v': 0, 'disc_area': 0, 'cup_h': 0, 'cup_v': 0, 'cup_area': 0, 'rim_area': 0}
        try:
            c_OD, _ = cv2.findContours(self.OD_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if c_OD:
                x, y, w, h = cv2.boundingRect(cv2.approxPolyDP(c_OD[0], 3, True))
                measurements['disc_h'], measurements['disc_v'] = w, h
                measurements['disc_area'] = np.sum(self.OD_mask == 255)
            c_OC, _ = cv2.findContours(self.OC_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if c_OC:
                x, y, w, h = cv2.boundingRect(cv2.approxPolyDP(c_OC[0], 3, True))
                measurements['cup_h'], measurements['cup_v'] = w, h
                measurements['cup_area'] = np.sum(self.OC_mask == 255)
            measurements['rim_area'] = measurements['disc_area'] - measurements['cup_area']
        except: pass
        return measurements
        
    def update_results(self, measurements, VCDR, HCDR, ACDR, is_glaucoma, ret_img):
        # Update CDR values with color coding
        for name, val, bar_name, thresh in [('vcdr_value', VCDR, 'vcdr_bar', 0.5), 
                                             ('hcdr_value', HCDR, 'hcdr_bar', 0.5),
                                             ('acdr_value', ACDR, 'acdr_bar', 0.55)]:
            if not np.isnan(val):
                getattr(self, name).configure(text=f"{val:.3f}")
                getattr(self, bar_name).set(min(val, 1.0))
                color = "#ff4444" if val > thresh else "#00ff00"
                getattr(self, bar_name).configure(progress_color=color)
                getattr(self, name).configure(text_color=color)
            else:
                getattr(self, name).configure(text="N/A")
        
        # Update measurements
        self.disc_v_value.configure(text=f"{measurements['disc_v']} px")
        self.disc_h_value.configure(text=f"{measurements['disc_h']} px")
        self.cup_v_value.configure(text=f"{measurements['cup_v']} px")
        self.cup_h_value.configure(text=f"{measurements['cup_h']} px")
        self.disc_area_value.configure(text=f"{measurements['disc_area']:,} px²")
        self.cup_area_value.configure(text=f"{measurements['cup_area']:,} px²")
        self.rim_area_value.configure(text=f"{measurements['rim_area']:,} px²")
        
        # Update diagnosis - CLEAR GLAUCOMA / NO GLAUCOMA
        if is_glaucoma:
            self.diagnosis_label.configure(text="⚠️ GLAUCOMA", text_color="#ff4444")
            self.diagnosis_indicator.configure(fg_color="#ff4444")
        else:
            self.diagnosis_label.configure(text="✅ NO GLAUCOMA", text_color="#00ff00")
            self.diagnosis_indicator.configure(fg_color="#00ff00")
            
        self.create_segmentation_image(ret_img, VCDR, is_glaucoma)
        self.export_btn.configure(state="normal")
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.analyze_btn.configure(state="normal")
        self.import_btn.configure(state="normal")
        self.status_label.configure(text="✓ Analysis complete!")
        
    def create_segmentation_image(self, ret_img, VCDR, is_glaucoma):
        vis_img = ret_img.copy()
        
        # Overlay masks
        for mask, color, alpha in [(self.OD_mask, [0, 255, 0], 0.3), (self.OC_mask, [255, 100, 0], 0.3)]:
            colored = np.zeros_like(vis_img)
            colored[mask == 255] = color
            vis_img = cv2.addWeighted(vis_img, 1, colored, alpha, 0)
        
        # Draw contours
        od_cnt, _ = cv2.findContours(self.OD_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        oc_cnt, _ = cv2.findContours(self.OC_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, od_cnt, -1, (0, 255, 0), 4)
        cv2.drawContours(vis_img, oc_cnt, -1, (255, 100, 0), 4)
        
        # Legend
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(vis_img, (10, 10), (280, 130), (0, 0, 0), -1)
        cv2.rectangle(vis_img, (10, 10), (280, 130), (255, 255, 255), 2)
        cv2.putText(vis_img, "SEGMENTATION", (20, 35), font, 0.6, (255, 255, 255), 2)
        cv2.circle(vis_img, (30, 60), 10, (0, 255, 0), -1)
        cv2.putText(vis_img, "Optic Disc", (50, 65), font, 0.5, (0, 255, 0), 1)
        cv2.circle(vis_img, (30, 90), 10, (255, 100, 0), -1)
        cv2.putText(vis_img, "Optic Cup", (50, 95), font, 0.5, (255, 100, 0), 1)
        vcdr_text = f"VCDR: {VCDR:.3f}" if not np.isnan(VCDR) else "VCDR: N/A"
        cv2.putText(vis_img, vcdr_text, (20, 120), font, 0.5, (255, 255, 255), 1)
        
        # Diagnosis box
        diag_text = "GLAUCOMA" if is_glaucoma else "NO GLAUCOMA"
        diag_color = (0, 0, 255) if is_glaucoma else (0, 255, 0)
        box_x = vis_img.shape[1] - 250
        cv2.rectangle(vis_img, (box_x, 10), (vis_img.shape[1] - 10, 55), (0, 0, 0), -1)
        cv2.rectangle(vis_img, (box_x, 10), (vis_img.shape[1] - 10, 55), diag_color, 3)
        cv2.putText(vis_img, diag_text, (box_x + 15, 42), font, 0.9, diag_color, 2)
        
        self.display_image(vis_img, self.segmented_label)
        self.result_image = vis_img
        
    def export_results(self):
        if self.result_image is None:
            return
        file_path = filedialog.asksaveasfilename(title="Save Results", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if file_path:
            cv2.imwrite(file_path, cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Success", f"Saved to: {file_path}")
        
    def show_error(self, msg):
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.analyze_btn.configure(state="normal")
        self.import_btn.configure(state="normal")
        self.status_label.configure(text="Error occurred")
        messagebox.showerror("Error", msg)


def main():
    app = GlaucomaDetectorApp()
    
    # Splash screen
    splash = ctk.CTkToplevel(app)
    splash.geometry("450x300")
    splash.overrideredirect(True)
    splash.configure(fg_color="#1a1a2e")
    x, y = (splash.winfo_screenwidth() - 450) // 2, (splash.winfo_screenheight() - 300) // 2
    splash.geometry(f"450x300+{x}+{y}")
    
    ctk.CTkLabel(splash, text="👁️", font=ctk.CTkFont(size=50)).pack(pady=(40, 10))
    ctk.CTkLabel(splash, text="GlaucomaFundus", font=ctk.CTkFont(size=32, weight="bold"), 
                 text_color="#00b4d8").pack(pady=5)
    ctk.CTkLabel(splash, text="CDR Assessment Tool v2.0", font=ctk.CTkFont(size=12), 
                 text_color="gray").pack(pady=5)
    progress = ctk.CTkProgressBar(splash, width=250, mode="indeterminate")
    progress.pack(pady=25)
    progress.start()
    ctk.CTkLabel(splash, text="Loading...", font=ctk.CTkFont(size=11), text_color="gray").pack()
    
    app.withdraw()
    app.after(2000, lambda: (splash.destroy(), app.deiconify()))
    app.mainloop()


if __name__ == "__main__":
    main()
