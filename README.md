# Glaucoma Detection Using U-Net and Deep Learning

An automated glaucoma detection system that leverages deep learning and advanced image processing techniques to analyze retinal fundus images. This application calculates the Cup-to-Disc Ratio (CDR) to determine the presence of glaucoma - a leading cause of irreversible blindness worldwide.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Implementation](#technical-implementation)
- [Datasets](#datasets)
- [Results and Metrics](#results-and-metrics)
- [GUI Application](#gui-application)
- [Academic Context](#academic-context)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

---

## Project Overview

Glaucoma is the second leading cause of blindness globally. Early detection is crucial for preventing vision loss, yet traditional diagnosis methods are labor-intensive and subjective. This project presents an automated solution that provides:

- Accurate optic disc and cup segmentation using U-Net architecture
- Automated Cup-to-Disc Ratio (CDR) calculation
- Real-time glaucoma risk assessment
- User-friendly GUI for medical professionals

The system processes retinal fundus images through a sophisticated pipeline that combines template matching, deep learning segmentation, and machine learning classification to deliver reliable diagnostic results.

---

## Key Features

### Core Functionality
- **Automated Optic Disc Localization**: Multi-map localization using template matching and brightness analysis
- **U-Net Semantic Segmentation**: Precise segmentation of optic disc and optic cup structures
- **CDR Calculation**: Automatic computation of Vertical, Horizontal, and Area Cup-to-Disc Ratios
- **Glaucoma Classification**: Machine learning-based prediction using Logistic Regression

### Advanced Technical Features
- **Multi-Scale Segmentation**: Robust detection across different image zoom levels
- **Ellipse Fitting**: Smooth boundary generation for accurate measurements
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for improved image quality
- **SLIC Superpixel Analysis**: Advanced brightness map extraction for disc localization

### User Interface
- **Modern GUI Application**: Built with CustomTkinter for a professional dark-themed interface
- **Real-Time Processing**: Live progress updates during analysis
- **Export Functionality**: Save segmentation results and diagnostic reports
- **Theme Customization**: Dark, Light, and System theme options
- **Web Interface**: Flask-based alternative for browser access

---

## Technologies Used

### Programming Language
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.12+ | Primary development language |

### Deep Learning and Machine Learning
| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.10.0+ | Deep learning framework for U-Net models |
| Keras | Integrated | High-level neural network API |
| scikit-learn | 1.1.0+ | Logistic Regression classifier |

### Image Processing
| Library | Version | Purpose |
|---------|---------|---------|
| OpenCV | 4.6.0+ | Image manipulation, contour detection, template matching |
| scikit-image | 0.19.0+ | SLIC superpixel segmentation, image transformations |
| Pillow | 9.0.0+ | Image loading and format handling |

### GUI and Visualization
| Library | Version | Purpose |
|---------|---------|---------|
| CustomTkinter | Latest | Modern themed desktop GUI |
| Matplotlib | 3.5.0+ | Visualization and plotting |
| Flask | 2.2.0+ | Web-based interface |

### Data Processing
| Library | Version | Purpose |
|---------|---------|---------|
| NumPy | 1.23.0+ | Numerical computations |
| Pandas | 1.4.0+ | Data manipulation and analysis |
| SciPy | 1.9.0+ | Scientific computing |

---

## System Architecture

The glaucoma detection pipeline consists of five sequential stages:

```
Input Image --> Preprocessing --> Localization --> Segmentation --> CDR Calculation --> Prediction
     |              |                  |                |                 |                |
  Fundus       Resize to          Find Optic       U-Net Models      Calculate        Classify
  Image        Standard Size      Disc Center      (OD & OC)         VCDR/HCDR/ACDR   Result
```

### Stage 1: Preprocessing
- Image resizing to standard dimensions (2000x2000 pixels)
- Color space normalization
- Noise reduction

### Stage 2: Optic Disc Localization
- Template matching using Normalized Cross-Correlation (NCC)
- Brightness map extraction via SLIC superpixel segmentation
- Multi-channel feature fusion (R, G, B + Brightness)

### Stage 3: Semantic Segmentation
- Region of Interest (ROI) extraction
- U-Net model inference for Optic Disc segmentation
- U-Net model inference for Optic Cup segmentation
- Ellipse fitting for boundary smoothing

### Stage 4: Feature Extraction
- Vertical CDR (VCDR) = Cup Height / Disc Height
- Horizontal CDR (HCDR) = Cup Width / Disc Width
- Area CDR (ACDR) = Cup Area / Disc Area

### Stage 5: Classification
- Feature vector construction
- Logistic Regression prediction
- Risk assessment output

---

## Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager
- Git

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/chouaib-saad/Glaucoma-Detection-UNet.git
   cd Glaucoma-Detection-UNet
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Additional GUI Package**
   ```bash
   pip install customtkinter
   ```

5. **Verify Installation**
   ```bash
   python -c "import tensorflow; import cv2; print('Installation successful')"
   ```

---

## Usage

### Option 1: Command Line Interface

```bash
cd Code
python main.py path/to/fundus_image.jpg
```

Example:
```bash
python main.py sample/V0400.jpg
```

### Option 2: GUI Application

```bash
python Code/glaucoma_app.py
```

The GUI provides:
- Import fundus images (JPG, PNG, BMP, TIFF)
- One-click analysis
- Visual segmentation results
- CDR measurements with progress bars
- Export functionality

### Option 3: Web Interface

```bash
cd Code/web\ gui/glaucoma\ gui
python main.py
```

Access the application at `http://localhost:5000`

---

## Project Structure

```
Glaucoma-Detection-UNet/
|
|-- README.md                      # Project documentation
|-- requirements.txt               # Python dependencies
|-- documentation.md               # Detailed technical documentation
|
|-- Code/
|   |-- main.py                    # CLI entry point
|   |-- glaucoma_app.py            # GUI application
|   |-- localization.py            # Optic disc localization module
|   |-- segmentation.py            # U-Net segmentation module
|   |-- classification.py          # CDR calculation and prediction
|   |-- supporting_function.py     # Helper utilities
|   |
|   |-- Models/
|   |   |-- model OD semantic/     # Optic Disc U-Net model
|   |   |-- model OC semantic/     # Optic Cup U-Net model
|   |   |-- Inference model/       # Classification model
|   |
|   |-- Templates/                 # Template images for localization
|   |-- sample/                    # Sample fundus images
|   |-- web gui/                   # Flask web application
|
|-- Notebooks/
|   |-- Main_notebook.ipynb        # Complete pipeline notebook
|   |-- Localization_Notebook.ipynb
|   |-- Segmentation_notebook.ipynb
|   |-- Inferencing_notebook.ipynb
|
|-- readme_img/                    # Documentation images
```

---

## Technical Implementation

### U-Net Architecture

The segmentation models utilize the U-Net architecture, specifically designed for biomedical image segmentation:

- **Input Layer**: 256 x 256 x 1 (grayscale ROI)
- **Encoder**: Convolutional blocks with max pooling
- **Bottleneck**: Deep feature extraction
- **Decoder**: Transposed convolutions with skip connections
- **Output Layer**: 256 x 256 x 1 (binary segmentation mask)

### Localization Algorithm

The optic disc localization combines multiple techniques:

1. **CLAHE Enhancement**: Improves contrast in each color channel
2. **Template Matching**: NCC-based correlation with pre-defined templates
3. **Brightness Mapping**: SLIC superpixel-based brightness analysis
4. **Feature Fusion**: Weighted combination of all feature maps

Coefficients (optimized via grid search):
- Red Channel: 1.0
- Green Channel: 0.2
- Blue Channel: 0.0
- Brightness Map: 0.8

### Classification Thresholds

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| VCDR | > 0.5 | Glaucoma risk indicator |
| ACDR | > 0.55 | Glaucoma risk indicator |
| Combined | VCDR > 0.45 AND HCDR > 0.45 | High risk assessment |

---

## Datasets

The models were trained on publicly available retinal fundus image datasets:

| Dataset | Description | Source |
|---------|-------------|--------|
| Drishti-GS1 | Indian glaucoma screening dataset | IIIT Hyderabad |
| REFUGE | Retinal Fundus Glaucoma Challenge | MICCAI 2018/2020 |

---

## Results and Metrics

### Segmentation Performance
- Optic Disc Segmentation: High accuracy boundary detection
- Optic Cup Segmentation: Precise cup delineation within disc region

### Classification Performance
- The Logistic Regression classifier provides reliable glaucoma/normal classification
- Features used: VCDR and ACDR (optimal feature combination)

---

## GUI Application

The CustomTkinter-based GUI offers a professional interface with:

### Main Features
- **Image Import**: Support for multiple image formats
- **Real-Time Analysis**: Progress tracking during processing
- **Visual Results**: Side-by-side original and segmented images
- **Measurements Panel**: Detailed disc and cup measurements
- **CDR Display**: Color-coded ratio indicators
- **Diagnosis Output**: Clear GLAUCOMA / NO GLAUCOMA result
- **Export Options**: Save results as PNG/JPEG

### Interface Components
- Sidebar with controls and results
- Main canvas for image display
- Status bar with processing information
- Theme selector (Dark/Light/System)

---

## Academic Context

This project was developed as an academic research initiative focused on applying deep learning techniques to medical image analysis. The implementation follows methodologies presented in peer-reviewed literature and aims to contribute to the field of automated ophthalmic diagnosis.

### Reference Publication

A. N. Almustofa, A. Handayani, and T. L. R. Mengko, "Optic Disc and Optic Cup Segmentation on Retinal Image Based on Multimap Localization and U-Net Convolutional Neural Network," Journal of Image and Graphics, Vol. 10, No. 3, pp. 109-115, September 2022.

### Training and Development

All models were trained independently using the aforementioned datasets. The training process, hyperparameter tuning, and evaluation are documented in the Jupyter notebooks provided in the repository.

---

## Contributing

This project welcomes contributions from researchers, developers, and medical professionals interested in advancing automated glaucoma detection.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Areas for Contribution
- Model architecture improvements
- Additional dataset integration
- Performance optimization
- Documentation enhancement
- Bug fixes and testing

Collaborators interested in joining this project for further development and research are encouraged to reach out.

---

## Contact

**Developer**: Chouaib Saad

- **Email**: choiyebsaad2000@gmail.com
- **Portfolio**: [https://chouaib-saad.vercel.app/](https://chouaib-saad.vercel.app/)
- **GitHub**: [https://github.com/chouaib-saad](https://github.com/chouaib-saad)

For questions, suggestions, or collaboration inquiries, please feel free to reach out via email or open an issue in this repository.

---

## License

Copyright (c) 2025 Chouaib Saad. All Rights Reserved.

This project is provided for educational and research purposes. Unauthorized commercial use, reproduction, or distribution is prohibited without explicit permission from the author.

---

## Disclaimer

This tool is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult a qualified ophthalmologist for proper glaucoma screening and diagnosis.

---

**Last Updated**: December 2025
