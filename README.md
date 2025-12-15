# Shadow Impact Analysis on Tile2Net Road Extraction Accuracy

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive analysis pipeline for quantifying the impact of shadow occlusion on the accuracy of road extraction from aerial imagery using the Tile2Net deep learning model.

## Overview

This system provides an automated, reproducible framework to evaluate how environmental factors like shadows affect computer vision models for geospatial analysis. By processing aerial imagery of New York City and comparing Tile2Net predictions with official ground truth data, we quantify performance degradation in shadowed regions with statistical rigor.

**Key Finding**: Shadows reduce Tile2Net's road extraction F1-score by approximately 20% in the New York test area.

## Start

### Prerequisites
- **Python**: 3.9 or higher
- **GPU**: NVIDIA GPU with CUDA support (RTX 4060 or equivalent recommended)
- **RAM**: Minimum 8GB
- **Storage**: 30GB free space for data processing
- **OS**: Windows 11

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/shadow-impact-analysis.git
   cd shadow-impact-analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

```bash
# Run complete analysis pipeline for New York downtown
python scripts/run_full_pipeline.py --config configs/new_york.yaml

# Run shadow-only analysis on existing predictions
python scripts/run_shadow_analysis.py --input data/intermediate/predictions

# Generate visualizations and reports
python scripts/visualize_results.py --output reports/nyc_analysis
```

## 📁 Project Structure

```
Shadow_Analysis/
├── data/                        
│   ├── raw/                       # Raw input data
│   └── output/                    
│       ├── tiles/                 
│       ├── stitch/          
│       └── segmentation/                 
├── scripts/                    
│   ├── accuracy_with_shadow.py      
│   ├── calculate_total_accuracy.py    
│   ├── create_ground_truth.py     
│   ├── generate_tiles.py     
│   ├── reduce_sidewalk.py   
│   ├── shadow_detector.py     
│   ├── total_tpfn_json.py    
│   ├── total_tpfn_picture.py    
│   └── tpfn_json_with_shadow.py       
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Core Modules

### 1. Tile2Net Wrapper (`scripts/generate_tiles.py.py`)
Handles interaction with the Tile2Net API, downloading aerial imagery tiles, and executing model inference.

```python
from tile2net import Raster
import geopandas as gpd

location_ny = '40.4978740560533,-74.2550385911449,40.9151395390951,-73.6996327792974'
location_ny_small = '40.714,-73.980,40.744,-74.010'#you could change the range you want from this line

raster = Raster(
    location=location_ny_small,
    zoom = 19,
    name='new york',
    output_dir=r"data/output",# choose your own dir
    dump_percent=100,

)
raster

raster.generate(8)

raster.inference()
```

### 2. Semantic Normalizer (`scripts/reduce_sidewalk.py`)
Converts Tile2Net's four-class output (roads, crosswalks, sidewalks, background) to a three-class system by merging crosswalks into road category.

```python
input_folder = r"\seg_results"  # tile2net seg result
output_folder = r"\output"  # output dir
```

### 3. Shadow Detector (`scripts/shadow_detector.py`)
Implements adaptive shadow detection using LAB color space analysis with morphological optimization.

```python
#fulfill your own path of your stitched tiles in input and choose your own output path in line 120
input_folder = r""
output_folder = r""
```

### 4. Create Ground Truth (`scripts/create_ground_truth.py`)


```python
#input in line 61
x1 = 154359 #the x coord from first picture in tiles
x2 = 154406 #the x coord from last picture in tiles
y1 = 197062 #the y coord from first picture in tiles
y2 = 197125 #the y coord from last picture in tiles
#input in line 83 change your own dir and copy it to tile2net function below
output_png = f"D:/data/outputdir/new york/segmentation/new york/256_19_8/19/{x}_{y}.png"  
```

### 5. Total Accuracy Evaluator (`scripts/calculate_total_accuracy.py`)
Compares predictions against ground truth and computes pixel-level accuracy metrics.

```python
#input in line 119
folder1_path = r"" #folder of ground truth
folder2_path = r"" #folder of predicted photo after processed in reduce_sidewalk.py
output_folder = r"" #output folder you want
```

### 6. Metrics Calculator (`accuracy_with_shadow.py`)
Compares predictions against ground truth and computes pixel-level accuracy metrics from both shadow region and non-shadow region.

```python
#input in line 247
img1_folder = r"" #folder of ground truth
img2_folder = r"" #folder of segmentation photo after processed in reduce_sidewalk.py
shadow_mask_folder = r"" #folder of shadow mask processed in shadow_detector.py
output_folder = r"" #output folder you want
```

### 7. TP/FN Calculator (`scripts/total_tpfn_json.py`)

```python
#input in line 115
folder1_path = r"\ground_truth" #folder of ground truth
folder2_path = r"" #folder of predicted photo after processed in reduce_sidewalk.py
output_folder = r"" # it will output json file of total tp/fn
```

### 8. TP/FN Calculator with shadow (`scripts/tpfn_json_with_shadow.py`)
```python
#input in line 253
folder1_path = r"\ground_truth"#folder of ground truth
folder2_path = r""#folder of predicted photo after processed in reduce_sidewalk.py
shadow_folder_path = r""#folder of shadow mask processed in shadow_detector.py
output_folder = r""# it will output json file of total tp/fn
```

## 📚 Dataset Information

### Data Sources
1. **Aerial Imagery**
   - **Source**: Tile2Net API / OpenStreetMap tiles
   - **Resolution**: ~0.5 meters/pixel (Zoom level 19)
   - **Format**: RGB PNG
   - **Coverage**: New York City test area (40.714° to 40.744° N, -74.010° to -73.980° W)

2. **Ground Truth Road Data**
   - **Source**: NYC Planimetrics 2022 Roadbed Dataset
   - **Provider**: City of New York, Department of Information Technology and Telecommunications
   - **Format**: GeoJSON (converted from Shapefile)
   - **Reference**: [NYC Planimetrics Capture Rules](https://github.com/CityOfNewYork/nyc-planimetrics/blob/main/Capture_Rules.md)
   - **License**: NYC OpenData Terms of Use
   - **Attributes**: Roadbed geometry, feature identifiers, road classification

3. **Study Area**
   - **Location**: Downtown Manhattan, New York City, USA
   - **Area**: Approximately 2.56 km²
   - **Tile Count**: 192 tiles (8×8 organization)
   - **Coordinate System**: WGS 84 (EPSG:4326)

### Data Processing Pipeline
1. **Tile Organization**: Raw 256×256 tiles organized into 8×8 super-tiles (2048×2048)
2. **Spatial Alignment**: All raster layers aligned to identical coordinate reference and resolution
3. **Semantic Normalization**: Tile2Net output converted from 4-class to 3-class system
4. **Shadow Detection**: Adaptive LAB color space analysis with morphological optimization
5. **Accuracy Assessment**: Pixel-level comparison with confusion matrix calculations
6. **Statistical Analysis**: Confidence intervals and significance testing

## 🔬 Methodology

### Shadow Detection Algorithm
The system employs an adaptive thresholding approach in LAB color space:

1. **Color Space Conversion**: RGB → LAB (separates luminance from chromaticity)
2. **Statistical Analysis**: Compute mean and standard deviation for each channel
3. **Adaptive Thresholding**:
   - If mean(A) + mean(B) ≤ threshold: Use luminance-only threshold
   - Otherwise: Combine luminance and B-channel thresholds
4. **Morphological Optimization**: Closing and opening operations to clean mask
5. **Mask Inversion**: Convert to standard format (0=shadow, 255=non-shadow)

### Accuracy Assessment Metrics
- **Precision**: TP / (TP + FP) - How many predicted roads are actually roads
- **Recall**: TP / (TP + FN) - How many actual roads are detected
- **F1-Score**: Harmonic mean of Precision and Recall
- **IoU (Jaccard Index)**: TP / (TP + FP + FN) - Area overlap measure
- **Confidence Intervals**: 95% confidence intervals using normal approximation

## 🔮 Future Work

Potential extensions and improvements:

1. **Algorithm Enhancements**
   - Integration of deep learning-based shadow detection
   - Multi-temporal shadow analysis across different times of day
   - Shadow intensity classification (light/medium/heavy shadows)

2. **Extended Analysis**
   - Cross-city comparison (different urban forms and climates)
   - Seasonal variation analysis (summer vs. winter shadows)
   - Other environmental factors (cloud cover, vegetation, water reflections)

3. **System Improvements**
   - Web-based interface for interactive analysis
   - Real-time processing capabilities
   - Cloud-native deployment with scalable computing

4. **Model Enhancement**
   - Shadow-aware training data augmentation
   - Domain adaptation for shadow-robust models
   - Uncertainty quantification in shadow zones

---

*Last Updated: May 2024 | Version: 1.0.0 | [View Changelog](CHANGELOG.md)*
