Processing Pipeline & System Architecture
The core objective of this pipeline is to ingest high-resolution aerial imagery, preprocess multi-spectral bands, and utilize an unsupervised deep learning approach to isolate crop anomalies (such as disease stress or phenotypic variations) without requiring massive labeled datasets.

Dataset Specifications
Source: National Agricultural Research Centre (NARC)

Target Crop: Mungbean

Temporal Depth: 13 days of continuous multi-spectral flight data

Sensor Modalities: Co-registered RGB (Visible) and NIR (Near-Infrared) imagery

The End-to-End Workflow
[ Raw Drone Imagery ] 
         │
         ▼
 1. Image Stitching (Microsoft ICE)
         │
         ▼
 2. Manual Spatial Alignment (QGIS)
         │
         ▼
 3. Grid-Based Patch Chunking (Python Script)
         │
         ▼
 4. Feature Engineering: NDVI Extraction ───► $NDVI = \frac{NIR - Red}{NIR + Red}$
         │
         ▼
 5. Dual-Channel Convolutional Autoencoder
         │
         ▼
[ Anomaly Detection via Reconstruction Error Evaluation ]
Phase 1: Pipeline Preprocessing
Image Stitching: Raw aerial captures from drone flights were compiled and stitched together using Microsoft ICE (Image Composite Editor) to generate cohesive, high-resolution orthomosaics for each flight day.

Multi-Spectral Band Alignment: To correct lens distortions and spatial offsets between separate camera sensors, the RGB and NIR orthomosaics were manually aligned and co-registered within QGIS, ensuring perfect pixel-to-pixel correspondence across bands.

Phase 2: Core Engineering & Feature Extraction
Grid-Based Patch Chunking: High-resolution orthomosaics are computationally too massive for deep networks to process directly. Custom Python scripts were written to slice the massive imagery into manageable, uniform patches (chunks) optimized for GPU memory boundaries.

Vegetation Index Calculation: Leveraging the aligned NIR and Red bands, the Normalized Difference Vegetation Index (NDVI) was engineered to isolate plant vigor and chlorophyll activity using the mathematical formula:

NDVI = \frac{NIR+Red}{NIR−Red}
​
 
Phase 3: Unsupervised Deep Learning Pipeline
Convolutional Autoencoder Training: The chunked RGB and engineered NDVI data layers were fed simultaneously into a deep Convolutional Autoencoder. The network was trained to compress these inputs into a low-dimensional bottleneck layer and then reconstruct them back to their original states.

Reconstruction-Based Anomaly Detection: * The Logic: The autoencoder learns to perfectly reconstruct normal, healthy crop patterns.

The Catch: When the model encounters a patch containing anomalies (e.g., localized crop disease, weeds, or structural damage), it fails to reconstruct it accurately.

The Result: By calculating the mathematical variance (Mean Squared Error) between the original patch and the regenerated patch, high reconstruction errors automatically flag and isolate spatial anomalies across the field.

Working Video Demonstration
To see this multi-spectral alignment, patch extraction, and localized anomaly detection system executing in real-time, watch the pipeline walkthrough below:
