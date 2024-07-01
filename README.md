# Nanoscale Single-vesicle Analysis
## Abstract
The analysis of membrane vesicles at the nanoscale level is crucial for advancing the understanding of intercellular communication and its implications for health and disease. Despite their significance, the nanoscale analysis of vesicles faces challenges owing to their small size and the complexity of biological fluids. This study introduces a novel approach that employs artificial intelligence (AI) to enhance the capabilities of super-resolution fluorescence microscopy for high-throughput single-vesicle analysis. By comparing classical clustering methods (K-means, DBSCAN, and SR-Tesseler) with deep-learning-based approaches (YOLO, DETR, Deformable DETR, and Faster R-CNN) for the analysis of super-resolution fluorescence images of exosomes, we identified the Deformable DETR algorithm as the most effective, with superior accuracy and a reduced processing time. Our findings demonstrate that AI-enhanced methods significantly outperform traditional clustering techniques in identifying individual vesicles and resolving the challenges related to misidentification and computational demands. Moreover, the application of the combined Deformable DETR and ConvNeXt-S algorithms to differently labeled exosomes revealed its capability to differentiate between them, indicating its potential to dissect the heterogeneity of vesicle populations. This breakthrough in vesicle analysis suggests a paradigm shift towards the integration of AI into super-resolution imaging, which is promising for unlocking new frontiers in vesicle biology, disease diagnostics, and the development of vesicle-based therapeutics.

## Usage

[YOLOv6](https://github.com/larpp/Nanoscale_Single-vesicle_Analysis/tree/main/YOLO)

## Results

### Detection

![image](https://github.com/larpp/Nanoscale_Single-vesicle_Analysis/assets/87048326/af333f46-8963-4934-aac8-3662780df2e7)

### Classification

![image](https://github.com/larpp/Nanoscale_Single-vesicle_Analysis/assets/87048326/bfddb91c-caef-4e29-89c2-80ce32f4a7a4)

|   |Accuracy   |Precision   |Recall   |F1-score   |
|:---:|:---:|:---:|:---:|:---:|
|K-means   |27%   |27%   |98%   |43%   |
|DBSCAN   |82%   |85%   |96%   |90%   |
|SR-Tesseler   |50%   |97%   |50%   |66%   |
|YOLO   |74%   |81%   |89%   |85%   |
|Faster R-CNN   |84%   |90%   |93%   |91%   |
|DETR   |73%   |76%   |96%   |85%   |
|Deformable DETR   |85%   |96%   |89%   |92%   |

