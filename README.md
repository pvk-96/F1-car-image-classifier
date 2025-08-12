# F1 Car Image Classification: End-to-End Pipeline

## Overview
This project implements a deep learning pipeline to classify images of Formula 1 cars into multiple classes using convolutional neural networks and transfer learning.
It covers data cleaning, augmentation, model training (CNN, MobileNetV2), evaluation (confusion matrix, metrics), and model optimization (pruning).

## Dataset
**Source:** Kaggle F1 Cars Image Dataset (downloaded)  
**Drive Access:** Google Drive Download Folder  

**Description:**  
Contains ~800–850 MB of curated F1 car images, organized in folders by class (car/model/team).  
Data cleaning was performed to remove corrupted/non-image files.

**Usage:**  
Download the dataset from the Google Drive link above and place it in your workspace as `/F1CarsDataset/Formula One Cars`.

## Files Provided
You will find in this repository:

**Notebook(s):**
- `f1-car-classification.ipynb` – End-to-end training, evaluation, model optimization.

**Report:**
- `report.pdf` – A summary of the workflow, results, and key findings.

**Sample Outputs:**
- Confusion matrix, accuracy/f1 plots, setup logs, training graphs.

**Scripts:** (optional, if exported from Colab)
- `requirements.txt` – List of Python packages required.
- `utils.py` – Any helper functions (if used).

**Model Weights:**
- Link to Google Drive for pruned/trained model files (since large files can't go on GitHub).
- Example: `/F1CarsDataset/final_pruned_model` in Drive.

## Usage Instructions

### Dataset Setup
1. Download images from the Google Drive folder.
2. Place the folder path as required in the notebook.

### Requirements
- Python 3.7+
- TensorFlow 2.9+, tensorflow-model-optimization, numpy, matplotlib, scikit-learn

For Colab: just run cells; for local, use `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Run Workflow
1. Open notebook in Colab or Jupyter.
2. Follow stepwise cells: data loading, visualization, training, evaluation, and pruning.
3. Save generated models to your Drive.

## Results
- **Best Accuracy:** ~78% on validation data using MobileNetV2 and data augmentation.
- **Optimization:** Model pruning applied for reduced model size and improved inference speed.
- **Visualization:** Training curves, confusion matrix, and per-class metrics included.

## Project Structure
```
├── f1-car-classification.ipynb
├── report.pdf
├── requirements.txt
├── utils.py          # (optional)
├── /images_results/  # Figures, confusion matrices, sample outputs
└── README.md
```
Dataset and model weights stored externally due to size limits.

## What Else Can I Upload To GitHub?
- All code (scripts, notebooks, helper files)
- Documentation (`README.md`, `report.pdf`)
- Small figures, sample output images
- Sample data (a few images, if they are under the file size limits)
- Requirements/environment files
- Instructions for how to download/use the larger dataset and models

For large data and models, always use links to external storage (Google Drive, Kaggle, HuggingFace, etc.).

## References
- Kaggle Dataset Homepage (add your specific dataset page if you wish)
- TensorFlow Model Optimization Toolkit documentation

## License
This code is released under the MIT License.  
Dataset is used as per Kaggle’s and original creator’s terms.
