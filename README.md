\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{amsmath}

\begin{document}

\section*{Processing Pipeline \& System Architecture}

The core objective of this pipeline is to ingest high-resolution aerial imagery, preprocess multi-spectral bands, and utilize an unsupervised deep learning approach to isolate crop anomalies (such as disease stress or phenotypic variations) without requiring massive labeled datasets.

\subsection*{Dataset Specifications}
\begin{itemize}
    \item \textbf{Source:} National Agricultural Research Centre (NARC)
    \item \textbf{Target Crop:} Mungbean
    \item \textbf{Temporal Depth:} 13 days of continuous multi-spectral flight data
    \item \textbf{Sensor Modalities:} Co-registered \textbf{RGB} (Visible) and \textbf{NIR} (Near-Infrared) imagery
\end{itemize}

\subsection*{The End-to-End Workflow}

\subsubsection*{Phase 1: Pipeline Preprocessing}
\begin{enumerate}
    \item \textbf{Image Stitching:} Raw aerial captures from drone flights were compiled and stitched together using \textbf{Microsoft ICE (Image Composite Editor)} to generate cohesive, high-resolution orthomosaics for each flight day.
    \item \textbf{Multi-Spectral Band Alignment:} To correct lens distortions and spatial offsets between separate camera sensors, the RGB and NIR orthomosaics were manually aligned and co-registered within \textbf{QGIS}, ensuring perfect pixel-to-pixel correspondence across bands.
\end{enumerate}

\subsubsection*{Phase 2: Core Engineering \& Feature Extraction}
\begin{enumerate}
    \setcounter{enumi}{2}
    \item \textbf{Grid-Based Patch Chunking:} High-resolution orthomosaics are computationally too massive for deep networks to process directly. Custom Python scripts were written to slice the massive imagery into manageable, uniform patches (chunks) optimized for GPU memory boundaries.
    \item \textbf{Vegetation Index Calculation:} Leveraging the aligned NIR and Red bands, the \textbf{Normalized Difference Vegetation Index (NDVI)} was engineered to isolate plant vigor and chlorophyll activity using the following mathematical formula:
    \begin{equation}
        NDVI = \frac{NIR - Red}{NIR + Red}
    \end{equation}
\end{enumerate}

\subsubsection*{Phase 3: Unsupervised Deep Learning Pipeline}
\begin{enumerate}
    \setcounter{enumi}{4}
    \item \textbf{Convolutional Autoencoder Training:} The chunked RGB and engineered NDVI data layers were fed simultaneously into a deep \textbf{Convolutional Autoencoder}. The network was trained to compress these inputs into a low-dimensional bottleneck layer and then reconstruct them back to their original states.
    \item \textbf{Reconstruction-Based Anomaly Detection:}
    \begin{itemize}
        \item \textbf{The Logic:} The autoencoder learns to perfectly reconstruct normal, healthy crop patterns.
        \item \textbf{The Catch:} When the model encounters a patch containing anomalies (e.g., localized crop disease, weeds, or structural damage), it fails to reconstruct it accurately.
        \item \textbf{The Result:} By calculating the mathematical variance (Mean Squared Error) between the \textbf{original patch} and the \textbf{regenerated patch}, high reconstruction errors automatically flag and isolate spatial anomalies across the field.
    \end{itemize}
\end{enumerate}

\vspace{0.5cm}
\noindent\textbf{Working Video Demonstration}\\
To see this multi-spectral alignment, patch extraction, and localized anomaly detection system executing in real-time, watch the pipeline walkthrough below:

\vspace{0.2cm}
\noindent\href{

https://github.com/user-attachments/assets/aab19eee-5045-4545-b09d-a071f5677de7

}{\textbf{\textit{Click Here to Watch the Project Video Demonstration}}}

\end{document}
