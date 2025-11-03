# Leukemia Cell Classification Using Deep Learning

An end-to-end computer vision pipeline for automated classification of leukemia cell subtypes from bone marrow smear images, achieving **100% accuracy** with lightweight YOLO models.

## Important Note

This project is based on the methodological approach described in Hosseini et al. (2023) [1], particularly the segmentation pipeline and preprocessing techniques. However, all code implementation was developed independently from scratch based on the paper's methodology description. Additionally, this project extends the original work by experimenting with different model architectures and input configurations, including the YOLO family models that were not part of the original study.

## Project Overview

This project implements an advanced image processing and deep learning workflow to differentiate between:
- **Benign hematogones** (healthy blood cells)
- **B-ALL malignant subtypes**:
  - Pre-B lymphoblasts
  - Pro-B lymphoblasts

The solution combines sophisticated cell segmentation techniques with modern CNN and YOLO architectures to achieve clinical-grade classification performance.

## Dataset

**Source**: [Kaggle Blood Cell Cancer ALL 4-Class Dataset](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class/data)

### Dataset Statistics
- **Total Images**: 2,263 bone marrow smear images
- **Resolution**: Variable (standardized to 224×224)
- **Classes**: 3 cell types

| Class | Original Count | After Augmentation |
|-------|----------------|-------------------|
| Benign | 512 | 811 |
| Pre-B | 955 | 811 |
| Pro-B | 796 | 811 |

### Sample Raw Dataset
![Sample Dataset](img/sample-raw-dataset.png)

## Methodology

The segmentation and preprocessing pipeline follows the approach described in Hosseini et al. (2023) [1], which uses LAB color space conversion and K-means clustering for effective cell isolation from bone marrow images.

### Project Pipeline Overview

![Full Pipeline](diagram-flow/[Diagram-flow]%20Leukemia.drawio.png)

### Data Processing Pipeline

![Processing Pipeline](diagram-flow/[Processing%20Diagram-flow]%20Leukemia%20Classification.png)

### Key Processing Stages

#### 1. **Advanced Segmentation Block**
Our custom segmentation pipeline isolates individual blast cells from complex bone marrow backgrounds, below is the sample visual of segmentation block to show the Original image into the Segmented image:

![Benign Segmentation](img/sample-data-processing-segmentation-benign.png)
![Pre-B Segmentation](img/sample-data-processing-segmentation-malignant-pre-b.png)
![Pro-B Segmentation](img/sample-data-processing-segmentation-malignant-pro-b.png)

**Technical Details**:
- LAB color space conversion for enhanced color separation
- K-means clustering (K=2) on A-channel
- Binary thresholding with morphological operations
- Size filtering (minimum 500px) to remove artifacts

#### 2. **Data Pairing for Dual-Channel Models**
For dual-input architectures, we create paired original-mask combinations:

| Benign | Pre-B Malignant | Pro-B Malignant |
|--------|-----------------|-----------------|
| ![Benign Pair](img/sample-data-ori-mask-pairing-dual-channel-model-benign.png) | ![Pre-B Pair](img/sample-data-ori-mask-pairing-dual-channel-model-malignant-pre-b.png) | ![Pro-B Pair](img/sample-data-ori-mask-pairing-dual-channel-model-malignant-pro-b.png) |

#### 3. **Data Augmentation & Balancing**
- **Techniques**: Horizontal and vertical flips
- **Result**: Perfect class balance (811 samples per class)
- **Applied to**: Training set only (prevents data leakage)

## Model Architectures

We benchmarked **7 state-of-the-art architectures** across three categories:

### Architecture Comparison

| Category | Model | Parameters | Trainable | Input Type |
|----------|-------|------------|-----------|------------|
| **CNN (Transfer Learning)** | EfficientNet-B0 | 4.38M | 0.33M | Mask only |
| | MobileNetV2 | 2.59M | 0.33M | Mask only |
| | NASNet-Mobile | ImageNet preset | Custom | Mask only |
| **Dual-Channel CNN** | Dual-MobileNetV2 | 3.70M | 1.44M | Original + Mask |
| **YOLO Family** | YOLOv8-n | 1.82M | All | Mask only |
| | **YOLOv11-n** | **1.63M** | All | Mask only |
| | YOLOv12-n | 1.82M | All | Mask only |

## Performance Results

### Overall Model Performance Ranking

| Rank | Model | Test Accuracy | Parameters | Key Strength |
|------|-------|--------------|------------|--------------|
| **1** | **YOLOv11-n** | **100%** | **1.63M** | Best efficiency |
| 1 | YOLOv8-n | 100% | 1.82M | Perfect accuracy |
| 1 | YOLOv12-n | 100% | 1.82M | Latest architecture |
| 4 | MobileNetV2 | 99.1% | 2.59M | Good accuracy |
| 4 | Dual-MobileNetV2 | 99.1% | 3.70M | Dual input |
| 6 | NASNet-Mobile | ~99% | Preset | NAS optimized |
| 7 | EfficientNet-B0 | 35.1% | 4.38M | Underperformed |

### Visual Performance Comparisons

#### CNN Models Performance
| EfficientNet-B0 | MobileNetV2 | NASNet-Mobile |
|-----------------|-------------|---------------|
| ![EfficientNet](img/model-performance-efficientnet-b0.png) | ![MobileNet](img/model-performance-efficientnet-mobilenet-v2.png) | ![NASNet](img/model-performance-efficientnet-nasnet-mobile.png) |

#### Dual-Channel Model
![Dual Channel](img/model-performance-custom-dual-channel-model.png)

#### YOLO Models Performance
| YOLOv8-n | YOLOv11-n | YOLOv12-n |
|----------|-----------|-----------|
| ![YOLOv8](img/model-performance-yolo-v8.png) | ![YOLOv11](img/model-performance-yolo-v11.png) | ![YOLOv12](img/model-performance-yolo-v12.png) |

### Test Set Performance Summary
![Test Performance](img/testing-performance-all-model.png)

## Key Findings

1. **Perfect Classification Achieved**: All YOLO models achieved 100% accuracy on test data
2. **Segmentation is Critical**: Mask-only input captures all necessary diagnostic features
3. **YOLOv11-n is Optimal**: Best accuracy-to-parameter ratio (1.63M params, 100% accuracy)
4. **Dual-Channel No Benefit**: Adding original images to masks provides no performance gain
5. **Clinical Viability**: >0.91 confidence scores suggest readiness for clinical validation

## Differences from Reference Paper

This implementation differs from Hosseini et al. (2023) [1] in several key aspects:

### Reference Paper (Hosseini et al., 2023):
- **Model Architectures**: EfficientNet-B0, MobileNet-V2, and NASNet-Mobile only
- **Input Configuration**: Dual-channel input combining original + segmented (mask-applied) images
- **Best Performance**: Achieved with dual-channel approach

### This Implementation:
- **Extended Model Set**:
  - Same CNN architectures (EfficientNet-B0, MobileNet-V2, NASNet-Mobile) but using **mask-only input**
  - Custom Dual-Channel MobileNet-V2 for fair comparison with the paper's approach
  - Additional YOLO family models (YOLOv8, YOLOv11, YOLOv12) not present in original study
- **Input Configurations Tested**:
  - Mask-only input for all models
  - Dual-channel (original + mask) specifically for MobileNet-V2
- **Key Finding**: Mask-only input performs equal or better than dual-channel, contradicting the original paper's approach
- **Best Performance**: YOLOv11-n with mask-only input (100% accuracy, 1.63M parameters)

## Technical Implementation

### Training Configuration
```python
# Global parameters
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
LEARNING_RATE = 0.001

# Data splits
TRAIN_SPLIT = 0.85  # 85%
VAL_SPLIT = 0.10    # 10%
TEST_SPLIT = 0.05   # 5%

# Model-specific
CNN_EPOCHS = 30
YOLO_EPOCHS = 23  # Early stopped
```

### Requirements
- Python 3.8+
- TensorFlow 2.x
- Ultralytics YOLOv8
- OpenCV
- scikit-learn
- NumPy, Pandas, Matplotlib

## Project Structure

```
leukemia-classification/
│
├── datasets/                      # Dataset files (excluded from git)
│   ├── train/                    # Training data
│   ├── valid/                    # Validation data
│   └── test/                     # Test data
│
├── docs/                          # Technical documentation
│   ├── [Technical Documentation] Leukemia Classification.docx
│   ├── [PROCESSING PIPELINE] Leukemia Classification.docx
│   └── [Project Presentation] Leukemia Classification.pdf
│
├── diagram-flow/                  # Process flow diagrams
│   ├── [Diagram-flow] Leukemia.drawio
│   ├── [Diagram-flow] Leukemia.drawio.png
│   ├── [Processing Diagram-flow] Leukemia Classification.drawio
│   └── [Processing Diagram-flow] Leukemia Classification.png
│
├── img/                          # Visualization images
│   ├── sample-*.png             # Data samples
│   └── model-performance-*.png  # Performance plots
│
├── leukemia-classification.ipynb # Main implementation notebook
├── [Summary Table] Leukemia Classification.xlsx  # Results summary
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Clinical Implications

This project demonstrates the feasibility of automated leukemia cell classification for:
- **Rapid screening** in resource-limited settings
- **Second-opinion systems** for pathologists
- **Training tools** for medical professionals
- **Point-of-care diagnostics** using edge devices

## Future Improvements

1. **External validation** on datasets from different medical centers
2. **Explainable AI** techniques for clinical interpretability
3. **Real-time inference** optimization for microscopy integration
4. **Multi-class expansion** to include more leukemia subtypes
5. **Uncertainty quantification** for borderline cases

## Summary Table

Detailed results and model comparisons are available in:
`[Summary Table] Leukemia Classification.xlsx`

## Research Applications

This work contributes to:
- Medical image analysis benchmarking
- Transfer learning in healthcare
- Edge AI for diagnostics
- Computer-aided diagnosis (CAD) systems

## Citations

If you use this work in your research, please cite both this implementation and the reference paper:

### This Implementation:
```bibtex
@misc{leukemia-classification-2024,
  author = {Zaky Ashari},
  title = {Leukemia Cell Classification Using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/zashari/leukemia-classification}
}
```

### Reference Paper:
```bibtex
@article{HOSSEINI2023101244,
  title = {A mobile application based on efficient lightweight CNN model for classification of B-ALL cancer from non-cancerous cells: A design and implementation study},
  author = {Azamossadat Hosseini and Mohammad Amir Eshraghi and Tania Taami and Hamidreza Sadeghsalehi and Zahra Hoseinzadeh and Mustafa Ghaderzadeh and Mohammad Rafiee},
  journal = {Informatics in Medicine Unlocked},
  volume = {39},
  pages = {101244},
  year = {2023},
  issn = {2352-9148},
  doi = {https://doi.org/10.1016/j.imu.2023.101244},
  url = {https://www.sciencedirect.com/science/article/pii/S2352914823000862}
}
```

## References

[1] Hosseini, A., Eshraghi, M. A., Taami, T., Sadeghsalehi, H., Hoseinzadeh, Z., Ghaderzadeh, M., & Rafiee, M. (2023). A mobile application based on efficient lightweight CNN model for classification of B-ALL cancer from non-cancerous cells: A design and implementation study. *Informatics in Medicine Unlocked*, 39, 101244. https://doi.org/10.1016/j.imu.2023.101244

## Acknowledgments

- **Dataset**: The dataset used in this project was provided by Hosseini et al. (2023) and is available on [Kaggle](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class/data)
- **Methodology**: The segmentation and preprocessing pipeline is based on the approach described in Hosseini et al. (2023) [1]
- **Implementation**: All code was independently developed based on the paper's methodology description
- Medical imaging community
- Open-source deep learning frameworks (TensorFlow, Ultralytics YOLO)

## License

This project is for educational and research purposes. Please ensure proper ethical approval for any clinical applications.

---
*Developed with focus on advancing AI-assisted medical diagnostics*
