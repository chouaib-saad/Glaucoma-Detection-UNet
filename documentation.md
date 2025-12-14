# 📘 GlaucomaFundus - Documentation

## 🎯 Project Overview

**GlaucomaFundus** is an automated glaucoma detection system that uses deep learning and image processing techniques to analyze retinal fundus images. The system calculates the **Cup-to-Disc Ratio (CDR)** to determine the presence of glaucoma - a leading cause of irreversible blindness worldwide.

---

## 🔬 How It Works

The glaucoma detection pipeline consists of **5 main stages**:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. Preprocessing │ → │  2. Localization │ → │  3. Segmentation │ → │  4. CDR Calc    │ → │  5. Prediction  │
│                   │    │  (Optic Disc)    │    │  (Disc & Cup)    │    │  (Features)     │    │  (Glaucoma?)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Stage 1: Preprocessing
- Resizes the input fundus image to a standard size (2000x2000 pixels)
- Normalizes the image for consistent processing

### Stage 2: Optic Disc Localization
- Uses **template matching** with Normalized Cross-Correlation (NCC)
- Extracts **brightness maps** using SLIC superpixel segmentation
- Combines multiple feature maps (Red, Green, Blue channels + Brightness)
- Locates the center of the optic disc

### Stage 3: Optic Disc & Cup Segmentation
- Extracts a **Region of Interest (ROI)** around the optic disc
- Uses **U-Net deep learning models** for semantic segmentation
- Segments both the **Optic Disc (OD)** and **Optic Cup (OC)**
- Applies **ellipse fitting** for smoother boundaries

### Stage 4: Feature Extraction (CDR Calculation)
- **Vertical CDR (VCDR)**: Cup height / Disc height
- **Horizontal CDR (HCDR)**: Cup width / Disc width  
- **Area CDR (ACDR)**: Cup area / Disc area

### Stage 5: Glaucoma Prediction
- Uses a **Logistic Regression classifier**
- CDR > 0.5 indicates potential glaucoma risk
- Returns **"Glaucoma"** or **"No Glaucoma"** diagnosis

---

## 🛠️ Technologies & Tools Used

### Programming Language
| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.12+ | Main programming language |

### Deep Learning & Machine Learning
| Library | Version | Purpose |
|---------|---------|---------|
| **TensorFlow** | ≥2.10.0 | Deep learning framework for U-Net models |
| **Keras** | (included in TF) | High-level neural network API |
| **scikit-learn** | ≥1.1.0 | Logistic Regression classifier |

### Image Processing
| Library | Version | Purpose |
|---------|---------|---------|
| **OpenCV (cv2)** | ≥4.6.0 | Image manipulation, contour detection, template matching |
| **scikit-image** | ≥0.19.0 | SLIC superpixel segmentation, image resizing |
| **Pillow (PIL)** | ≥9.0.0 | Image loading and display |

### GUI Framework
| Library | Version | Purpose |
|---------|---------|---------|
| **CustomTkinter** | Latest | Modern dark-themed GUI interface |
| **Tkinter** | Built-in | File dialogs, message boxes |

### Data Processing & Visualization
| Library | Version | Purpose |
|---------|---------|---------|
| **NumPy** | ≥1.23.0 | Numerical computations, array operations |
| **Matplotlib** | ≥3.5.0 | Visualization and plotting |
| **Pandas** | ≥1.4.0 | Data manipulation |

### Web Framework (Optional)
| Library | Version | Purpose |
|---------|---------|---------|
| **Flask** | ≥2.2.0 | Web-based GUI (alternative interface) |

---

## 📁 Project Structure

```
Glaucoma-Detection-UNet-main/
│
├── 📄 README.md                    # Project introduction
├── 📄 requirements.txt             # Python dependencies
├── 📄 documentation.md             # This documentation file
│
├── 📂 Code/                        # Main source code
│   ├── 🐍 main.py                  # CLI entry point
│   ├── 🐍 glaucoma_app.py          # GUI application (CustomTkinter)
│   ├── 🐍 localization.py          # Optic Disc localization module
│   ├── 🐍 segmentation.py          # U-Net segmentation module
│   ├── 🐍 classification.py        # CDR calculation & prediction
│   ├── 🐍 supporting_function.py   # Helper functions
│   │
│   ├── 📂 Models/                  # Pre-trained models
│   │   ├── 📂 model OD semantic/   # Optic Disc U-Net model
│   │   ├── 📂 model OC semantic/   # Optic Cup U-Net model
│   │   └── 📂 Inference model/     # Logistic Regression classifier
│   │
│   ├── 📂 Templates/               # Template images for localization
│   │   ├── 🖼️ ROItemplateRed.png
│   │   ├── 🖼️ ROItemplateGreen.png
│   │   └── 🖼️ ROItemplateBlue.png
│   │
│   ├── 📂 sample/                  # Sample fundus images
│   └── 📂 web gui/                 # Flask web interface
│
├── 📂 Notebooks/                   # Jupyter notebooks for training
│   ├── 📓 Main_notebook.ipynb
│   ├── 📓 Localization_Notebook.ipynb
│   ├── 📓 Segmentation_notebook.ipynb
│   └── 📓 Inferencing_notebook.ipynb
│
└── 📂 readme_img/                  # Documentation images
```

---

## 🔧 Principal Functions & Features

### 1. Localization Module (`localization.py`)

| Function | Description |
|----------|-------------|
| `OD_localization()` | Main class for optic disc localization |
| `preprocessing(src, std_size)` | Resizes image to standard dimensions |
| `extract_BR_map(src, mask)` | Creates brightness map using SLIC superpixels |
| `image_template_matching(img, template, mask)` | Performs NCC template matching |
| `locate(src, coeff_args)` | Returns the (x, y) center of the optic disc |

### 2. Segmentation Module (`segmentation.py`)

| Function | Description |
|----------|-------------|
| `optic_segmentation()` | Main class for disc/cup segmentation |
| `do_segmentation(ROI, coordinate, shape)` | Performs U-Net segmentation |
| `ellipsTransform(mask)` | Fits ellipse to contour for smooth boundaries |
| `multi_scale_segmentation(ROI)` | Robust segmentation at multiple scales |
| `validate_cup_within_disc(OD_mask, OC_mask)` | Ensures cup is inside disc |
| `enhance_cup_detection(ROI, OD_mask, OC_pred)` | Improves cup detection accuracy |

### 3. Classification Module (`classification.py`)

| Function | Description |
|----------|-------------|
| `inference_glaucoma()` | Main class for glaucoma prediction |
| `CDR_calc(OD_mask, OC_mask)` | Calculates VCDR, HCDR, ACDR |
| `predict(feature)` | Returns "Glaucoma" or "Normal" |

### 4. Supporting Functions (`supporting_function.py`)

| Function | Description |
|----------|-------------|
| `ekstrakROI(centroid, s, img)` | Extracts Region of Interest around disc |
| `rectfromcenter(center, s, h, w)` | Calculates ROI boundaries |
| `fscore(y_true, y_pred)` | Calculates F-score metric |
| `precision_m()` / `recall_m()` | Precision and recall metrics |

### 5. GUI Application (`glaucoma_app.py`)

| Feature | Description |
|---------|-------------|
| 📁 **Import Image** | Load fundus images (JPG, PNG, BMP, TIFF) |
| 🔍 **Analyze Image** | Run full detection pipeline |
| 📊 **CDR Display** | Shows VCDR, HCDR, ACDR with progress bars |
| 🏥 **Diagnosis** | Clear GLAUCOMA / NO GLAUCOMA result |
| 📏 **Measurements** | Disc/Cup dimensions and areas |
| 💾 **Export Results** | Save segmentation image |
| 🎨 **Theme Selector** | Dark / Light / System themes |

---

## 📊 Key Metrics & Thresholds

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| **VCDR** | > 0.5 | Glaucoma risk |
| **ACDR** | > 0.55 | Glaucoma risk |
| **Combined** | VCDR > 0.45 AND HCDR > 0.45 | Glaucoma risk |

---

## 🚀 How to Run

### Option 1: Command Line Interface
```bash
cd Code
python main.py path/to/fundus_image.jpg
```

### Option 2: GUI Application
```bash
py -3.12 Code/glaucoma_app.py
```

### Option 3: Web Interface (Flask)
```bash
cd Code/web\ gui/glaucoma\ gui
python main.py
```

---

## 📚 Datasets Used for Training

| Dataset | Description | Link |
|---------|-------------|------|
| **Drishti-GS1** | Indian glaucoma dataset | [IIIT Hyderabad](http://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) |
| **REFUGE** | Retinal Fundus Glaucoma Challenge | [Grand Challenge](https://refuge.grand-challenge.org/) |

---

## 🧠 Deep Learning Models

### U-Net Architecture
- **Input**: 256×256×1 grayscale ROI image
- **Output**: 256×256×1 binary segmentation mask
- **Models**: 
  - `model OD semantic` - Optic Disc segmentation
  - `model OC semantic` - Optic Cup segmentation

### Classifier
- **Algorithm**: Logistic Regression
- **Features**: VCDR, ACDR
- **Output**: Binary classification (Glaucoma / Normal)

---

## 📖 Reference

> A. N. Almustofa, A. Handayani, and T. L. R. Mengko, "Optic Disc and Optic Cup Segmentation on Retinal Image Based on Multimap Localization and U-Net Convolutional Neural Network," *Journal of Image and Graphics*, Vol. 10, No. 3, pp. 109-115, September 2022.

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Original | CLI-based detection |
| 2.0 | Dec 2025 | Added CustomTkinter GUI with modern interface |

---

## ⚠️ Disclaimer

This tool is intended for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult a qualified ophthalmologist for proper glaucoma screening and diagnosis.

---

*Documentation generated on December 11, 2025*
