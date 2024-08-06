# Nanoscale Single-vesicle Analysis
## Abstract
The analysis of membrane vesicles at the nanoscale level is crucial for advancing the understanding of intercellular communication and its implications for health and disease. Despite their significance, the nanoscale analysis of vesicles faces challenges owing to their small size and the complexity of biological fluids. This study introduces a novel approach that employs artificial intelligence (AI) to enhance the capabilities of super-resolution fluorescence microscopy for high-throughput single-vesicle analysis. By comparing classical clustering methods (K-means, DBSCAN, and SR-Tesseler) with deep-learning-based approaches (YOLO, DETR, Deformable DETR, and Faster R-CNN) for the analysis of super-resolution fluorescence images of exosomes, we identified the Deformable DETR algorithm as the most effective, with superior accuracy and a reduced processing time. Our findings demonstrate that AI-enhanced methods significantly outperform traditional clustering techniques in identifying individual vesicles and resolving the challenges related to misidentification and computational demands. Moreover, the application of the combined Deformable DETR and ConvNeXt-S algorithms to differently labeled exosomes revealed its capability to differentiate between them, indicating its potential to dissect the heterogeneity of vesicle populations. This breakthrough in vesicle analysis suggests a paradigm shift towards the integration of AI into super-resolution imaging, which is promising for unlocking new frontiers in vesicle biology, disease diagnostics, and the development of vesicle-based therapeutics.

## Usage

[YOLOv6](https://github.com/larpp/Nanoscale_Single-vesicle_Analysis/tree/main/YOLO)

[ConvNeXt](https://github.com/larpp/Nanoscale_Single-vesicle_Analysis/tree/main/ConvNeXt)

[MMDetection](https://github.com/larpp/Nanoscale_Single-vesicle_Analysis/tree/main/mmdetection) and [MMDetection inference](https://github.com/larpp/MMDetection_Inference)

## Results

### Detection

![스크린샷 2024-08-06 122854](https://github.com/user-attachments/assets/47b2dba9-068e-4417-826f-3f6803e1730e)

### Classification

![스크린샷 2024-08-06 123028](https://github.com/user-attachments/assets/21715ac9-47de-4ad8-9a95-e861436bcbb0)
