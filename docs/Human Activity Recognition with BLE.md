---
noteId: "3e6be3d0d5cf11f080694155c34f83e4"
tags: []

---

# **Advanced Architectures and Fusion Paradigms for Multimodal Human Activity Recognition: Integrating BLE RSSI and Inertial Sensors**

## **1\. Introduction**

The domain of Human Activity Recognition (HAR) has progressively shifted from simple, heuristic-based detection of ambulatory states to complex, deep learning-driven systems capable of interpreting intricate behavioral patterns. This evolution is driven by the ubiquity of wearable sensors and the increasing sophistication of ambient sensing infrastructures. The research scenario presented—utilizing a dense network of 25 Bluetooth Low Energy (BLE) beacons for Received Signal Strength Indicator (RSSI) data, augmented by future accelerometer integration, over a 5-second sliding window yielding approximately 11,500 samples—represents a high-dimensional, heterogeneous time-series classification problem. This specific configuration places the system at the intersection of *kinematic* sensing (accelerometry) and *contextual* sensing (RF-based localization and occupancy), creating a rich but noisy feature space that necessitates advanced architectural interventions.

The sheer volume of samples (11,500 per window) implies a high-frequency data acquisition pipeline, likely dominated by the inertial measurement unit (IMU) or an aggregation of high-rate beacon advertisements. This density poses significant challenges for traditional architectures like Recurrent Neural Networks (RNNs) due to vanishing gradients over long sequences and high computational costs for Transformers due to quadratic attention complexity. Consequently, the selection of state-of-the-art (SOTA) Deep Learning (DL) architectures must balance the capability to model long-range temporal dependencies with the efficiency required for inference.

This report provides an exhaustive analysis of the theoretical underpinnings, optimal configurations, and practical implementations of SOTA architectures including 1D-CNNs, InceptionTime, ROCKET, and Time Series Transformers (TST). It further explores the physics of RSSI feature engineering, focusing on robust metrics like Level Crossing Rate (LCR) and spectral entropy, and delineates fusion strategies that effectively integrate the asynchronous, multi-rate nature of BLE and accelerometer data.

## **2\. State-of-the-Art Deep Learning Architectures for Time Series Classification**

The landscape of Time Series Classification (TSC) has diversified significantly beyond simple Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. Recent benchmarks on the UCR/UEA archives demonstrate that ensemble methods and kernel-based transforms often outperform pure recurrent architectures in both accuracy and training efficiency.

### **2.1. 1D Convolutional Neural Networks (1D-CNN) and ResNets**

While often considered a baseline, 1D-CNNs remain a critical component of HAR systems, particularly for edge deployment where computational resources are constrained. Unlike their 2D counterparts in computer vision, 1D-CNNs perform convolutions strictly along the temporal dimension, extracting translation-invariant features such as peaks, slopes, and periodic patterns indicative of gait or repetitive motion.

#### **2.1.1. Architectural Mechanics**

A standard 1D-CNN for HAR typically consists of three to four convolutional layers. Each layer applies a set of learnable filters (kernels) to the input window. For an input sequence $X \\in \\mathbb{R}^{T \\times C}$ (where $T$ is time steps and $C$ is channels, e.g., 25 beacons), the convolution operation at layer $l$ can be expressed as:

$$h\_{i}^{l} \= \\sigma \\left( \\sum\_{k=1}^{K} w\_{k}^{l} \* h\_{(i+k)}^{l-1} \+ b^{l} \\right)$$  
where $\\sigma$ is a non-linear activation function (typically ReLU), $w$ represents the kernel weights, and $b$ is the bias. The critical hyperparameter here is the kernel size. Small kernels (e.g., 3-5) capture high-frequency noise and rapid changes typical of accelerometer impacts, while large kernels (e.g., 20-50) smooth the data, making them more suitable for the slowly varying RSSI signals affected by multipath fading.

#### **2.1.2. Residual Networks (ResNet) for Time Series**

To enable deeper networks without the degradation of training accuracy (vanishing gradients), ResNets utilize skip connections. The SOTA ResNet architecture for TSC, as defined in recent benchmarks, comprises three residual blocks.1 Each block contains three convolutional layers with varying filter counts (e.g., 64, 128, 128\) and kernel sizes (e.g., 8, 5, 3), followed by Batch Normalization and ReLU activation. The residual connection adds the input of the block directly to its output:

$$y \= F(x, \\{W\_i\\}) \+ x$$  
This addition operation allows the gradient to flow directly through the network during backpropagation, facilitating the learning of identity mappings if deeper layers are unnecessary. For the user's high-dimensional input (11,500 samples), ResNets offer a distinct advantage: they can maintain a large receptive field through depth, allowing the model to correlate an RSSI drop at $t=0$ with an accelerometer spike at $t=4s$.2

### **2.2. InceptionTime: The Ensemble Standard**

InceptionTime is widely recognized as the current state-of-the-art for deep learning-based TSC, effectively serving as the "AlexNet" of the domain—a robust, high-performance baseline that is difficult to beat without significant computational overhead.1

#### **2.2.1. The Inception Module**

The core innovation of InceptionTime addresses the difficulty of selecting the optimal kernel size. In a heterogeneous sensor environment, the optimal kernel size for an accelerometer (capturing sharp impacts) differs from that of a BLE beacon (capturing gradual signal decay). An Inception module applies multiple filters of varying lengths (e.g., 10, 20, 40\) simultaneously to the same input and concatenates their outputs.5

$$\\text{Inception}(X) \= \\text{Concat}\\left( \\text{Conv}\_{1 \\times 10}(X), \\text{Conv}\_{1 \\times 20}(X), \\text{Conv}\_{1 \\times 40}(X), \\text{MaxPooling}(X) \\right)$$  
This multi-scale processing allows the network to learn both short-term and long-term dependencies within a single layer.

#### **2.2.2. Bottleneck Layers and Ensembling**

To manage the high dimensionality of the user's 28-channel input (25 BLE \+ 3 Accel), InceptionTime employs "bottleneck" layers—1x1 convolutions that reduce the number of input channels before the expensive large-kernel convolutions. This significantly reduces the number of parameters and computational cost. Furthermore, InceptionTime is strictly defined as an ensemble of five distinct Inception networks, trained with different random initializations. The final prediction is an average of the softmax outputs, which drastically reduces variance and improves stability on noisy RSSI datasets.5

### **2.3. ROCKET and MiniRocket**

ROCKET (RandOm Convolutional KErnel Transform) represents a paradigm shift from traditional deep learning. Instead of learning kernel weights via backpropagation, ROCKET generates a massive number (typically 10,000) of random kernels and uses them to transform the time series into a high-dimensional feature vector, which is then classified using a simple linear classifier (e.g., Ridge Regression).7

#### **2.3.1. Mechanism of Transformation**

For each random kernel, ROCKET computes two aggregate features across the time series:

1. **Maximum Value (Max):** Captures the presence of the most significant pattern match (e.g., the strongest signal peak).  
2. **Proportion of Positive Values (PPV):** Captures the frequency or duration of the pattern match (e.g., how long the user remained in a specific zone).

The transformation maps the input time series $X \\in \\mathbb{R}^{T \\times C}$ to a feature vector $f \\in \\mathbb{R}^{20,000}$. This method is exceptionally fast and has been shown to match or exceed the accuracy of HIVE-COTE and InceptionTime on the UCR archive while being orders of magnitude faster to train.7

#### **2.3.2. MiniRocket: Deterministic Efficiency**

MiniRocket refines the original concept by making the kernels *deterministic* and restricting their weights to a small set of values (e.g., \-1, 2). This optimization allows for highly optimized computational routines, making MiniRocket significantly faster than the original ROCKET. For the user's setup, which involves potentially retraining models as new data arrives or adapting to new environments, MiniRocket offers the most efficient path to high accuracy. It naturally handles multivariate data by applying kernels to random subsets of channels, effectively capturing cross-channel correlations (e.g., simultaneous signal drop in Beacon A and B).9

### **2.4. Time Series Transformers (TST)**

Transformers, originally designed for Natural Language Processing (NLP), utilize self-attention mechanisms to weigh the importance of different time steps relative to each other, theoretically capturing global dependencies across the entire sequence.

#### **2.4.1. The Self-Attention Mechanism**

The core of the Transformer is the Scaled Dot-Product Attention:

$$\\text{Attention}(Q, K, V) \= \\text{softmax}\\left( \\frac{QK^T}{\\sqrt{d\_k}} \\right)V$$  
For HAR, this mechanism allows the model to "attend" to relevant time steps regardless of their distance in the sequence. For example, the model can learn that a specific sequence of RSSI fluctuations at $t=0$ is highly relevant to an activity classification at $t=5$, ignoring the noise in between.11

#### **2.4.2. Challenges with Long Sequences**

The user's 11,500-sample window poses a critical challenge: standard self-attention has a time and memory complexity of $O(L^2)$, where $L$ is the sequence length. Calculating an $11,500 \\times 11,500$ attention matrix is computationally prohibitive for most GPUs.

* **Patching (PatchTST):** A solution proposed in recent literature (2023-2024) is "Patching." The time series is segmented into sub-sequences (patches) of length $P$ (e.g., 16 or 64). These patches are projected into embeddings and treated as tokens. This reduces the effective sequence length from $L$ to $L/P$, drastically reducing computational cost while preserving local semantic information.13  
* **Zerveas et al. Framework:** This framework emphasizes unsupervised pre-training. A Transformer encoder is trained on a "denoising" objective (predicting masked values) using massive amounts of unlabeled sensor data. This is highly relevant if the user has logged hours of unlabeled RSSI data. The pre-trained encoder is then fine-tuned with a classification head on the labeled HAR dataset.14

### **2.5. Comparative Analysis of Architectures**

The following table summarizes the suitability of these architectures for the user's specific heterogeneous environment.

| Architecture | Mechanism | Pros for BLE/Accel HAR | Cons for 11.5k Samples |
| :---- | :---- | :---- | :---- |
| **1D-CNN** | Local Convolution | Fast inference; good for detecting local features (steps). | Poor at capturing global context (5s duration) without extreme depth. |
| **InceptionTime** | Multi-scale Ensemble | Captures both short (Accel) and long (RSSI) patterns; SOTA accuracy. | Computationally heavy due to ensemble nature; moderate training time. |
| **MiniRocket** | Kernel Transform | **Fastest training**; handles multivariate naturally; high accuracy. | Feature vector can be large (20k-50k); linear classifier might limit complexity. |
| **Transformer (PatchTST)** | Self-Attention | Captures global dependencies; robust to noise via attention. | **$O(L^2)$ complexity** requires patching; difficult to train on small datasets. |
| **LSTM/BiLSTM** | Recurrence | Models sequential nature explicitly. | **Too slow** for 11.5k sequence; suffers from vanishing gradients. |

## **3\. Optimal Sliding Window Strategies**

The sliding window approach segments the continuous sensor stream into discrete units for classification. The choice of window size ($W$) and overlap ($O$) is a hyperparameter that dictates the system's latency, computational load, and recognition capability.

### **3.1. The 5-Second Window Analysis**

The user's choice of a 5-second window is relatively long compared to the typical 1-3 seconds used in accelerometer-based HAR. However, this duration is justifiable and potentially optimal for **BLE-based** recognition.

* **RSSI Stabilization:** BLE signals are stochastic. A single scan is unreliable due to fast fading. A 5-second window allows for the aggregation of multiple advertisement packets (e.g., 50-100 packets at 10-20Hz), facilitating the extraction of stable statistical moments (mean, median) that accurately reflect proximity.16  
* **Complex Activities:** Research indicates that while simple repetitive motions (walking, running) can be recognized in 2-3 seconds, complex activities involving environmental interaction (e.g., "taking the elevator," "entering a room," "sitting down at a desk") require longer temporal context to differentiate from transient pauses. A window of 5-6 seconds has been shown to maximize F1-scores for these stationary-transition activities.16

### **3.2. Overlapping Strategies**

To mitigate the latency inherent in a 5-second window (where the system waits 5s to make a decision), a high overlap strategy is essential.

* **Recommendation:** Implement a **75% to 80% overlap**.  
  * With a 5s window and 80% overlap, the "step size" is 1 second. This means the system outputs a prediction every 1 second, based on the previous 5 seconds of data.  
  * **Benefit:** This high overlap ensures that short-duration transitions (e.g., a 2-second "stand-to-sit" action) are centered in at least one or two windows, preventing them from being split across window boundaries and misclassified.17

### **3.3. Multi-Resolution Windowing (The "Zoom-In" Strategy)**

Given the heterogeneity of the sensors, a dual-window strategy can be employed within the fusion architecture:

1. **Macro-Window (5s):** Used for BLE RSSI data. This maximizes the stability of the location features.  
2. **Micro-Window (2s):** Used for Accelerometer data. A 2-second sub-window (centered or trailing) is extracted from the 5-second buffer. This focuses the kinematic classification on the immediate action while maintaining the broader locational context. This approach minimizes the "Null Class" problem where a 5s window might contain multiple distinct physical actions.16

## **4\. Advanced Feature Engineering for RSSI**

While Deep Learning models can theoretically learn features from raw data, feeding raw RSSI values (integers from \-100 to \-30 dBm) is often suboptimal due to the signal's logarithmic nature and high noise floor. Explicit feature engineering acts as an inductive bias, guiding the model toward robust physical properties of the signal.

### **4.1. Statistical Features (Distributional)**

These features describe the probability distribution of the signal strength over the 5-second window.

* **Central Tendency:** *Mean* and *Median* (more robust to outliers/packet collisions).  
* **Dispersion:** *Standard Deviation* and *Variance* (highly indicative of motion; static devices have low RSSI variance, moving devices have high variance due to multipath traversal).  
* **Shape:** *Skewness* and *Kurtosis* describe the asymmetry and tailedness of the signal distribution, which can characterize specific multipath environments (e.g., a narrow corridor vs. an open hall).19  
* **Quantiles:** *Interquartile Range (IQR)* and *75th Percentile* provide dispersion metrics robust to extreme outliers caused by signal shadowing.

### **4.2. Temporal and Differential Features**

These features capture the dynamics of the user's movement through the RF field.

* **Level Crossing Rate (LCR):** This is a critical feature derived from fading channel theory. It is defined as the number of times the RSSI signal crosses a specific threshold (usually the local mean) in the positive-going direction.  
  * *Significance:* LCR is directly proportional to the user's velocity and the carrier frequency. In HAR, a high LCR indicates rapid movement or a highly cluttered environment, while a low LCR indicates stationarity.21  
* **Differential RSSI:** The first-order difference sequence $\\Delta RSSI\_t \= RSSI\_t \- RSSI\_{t-1}$.  
  * *Significance:* This acts as a high-pass filter, removing the static path loss component (distance) and isolating the Doppler-induced or movement-induced variations. The variance of the differential RSSI is a powerful discriminator for "activity intensity".23

### **4.3. Entropy-Based Features**

Entropy quantifies the complexity and unpredictability of the signal.

* **Shannon Entropy:** $H(X) \= \-\\sum p(x) \\log p(x)$. A static user in a stable environment will have low RSSI entropy (peaked distribution). A moving user or a user in a changing environment (e.g., people walking by) will have higher entropy (flatter distribution).  
* **Spectral Entropy:** Calculated from the Power Spectral Density (PSD) of the signal. It measures the flatness of the spectrum. High spectral entropy implies a noise-like signal (active motion/NLoS), while low entropy implies dominant periodic components (potential gait artifacts in RSSI).25

### **4.4. Frequency Domain (FFT)**

Although BLE sampling is often irregular, interpolating the data to a fixed grid allows for Fast Fourier Transform (FFT) analysis.

* **Technique:** Apply a linear interpolation to generate a uniform 10Hz or 20Hz signal. Compute the FFT.  
* **Features:** *Peak Frequencies* (identifying gait cadence if shadowing is rhythmic) and *Spectral Energy* (total power in the AC component). This is particularly useful for distinguishing "walking" (periodic shadowing) from "random movement".27

## **5\. Multi-Modal Fusion Strategies**

The core challenge is fusing the **25-channel sparse/slow RSSI data** with the **3-channel dense/fast Accelerometer data**.

### **5.1. Early Fusion (Feature Concatenation)**

This is the simplest approach but requires careful synchronization.

* **Strategy:** Features (statistical, temporal, FFT) are extracted from both modalities over the same 5s window. These feature vectors are concatenated into a single vector $V\_{fused} \=$.  
* **Preprocessing:** Raw data fusion requires upsampling the BLE data (via Zero-Order Hold or Interpolation) to match the Accelerometer's sampling rate.  
* **Pros:** Allows the model to learn low-level correlations (e.g., "High Z-axis variance" \+ "Strong Beacon 5" \= "Climbing Stairs").  
* **Cons:** Increases dimensionality significantly; prone to "curse of dimensionality" if data is limited.28

### **5.2. Late Fusion (Ensemble Decision)**

* **Strategy:** Train two independent models:  
  1. **Context Model:** A ROCKET or MLP classifier trained *only* on RSSI data.  
  2. **Motion Model:** A 1D-CNN or InceptionTime model trained *only* on Accelerometer data.  
* **Aggregation:** Fuse the softmax probability outputs using a meta-learner (e.g., Logistic Regression) or a weighted average ($P\_{final} \= \\alpha P\_{RSSI} \+ \\beta P\_{Accel}$).  
* **Pros:** Highly modular. If the BLE system goes offline, the Motion Model continues to function. It inherently handles different sampling rates as fusion occurs at the decision level.30

### **5.3. Intermediate Fusion with Cross-Attention (Recommended SOTA)**

This architecture represents the cutting edge of multimodal HAR, leveraging the strengths of Transformers to dynamically weight modalities.

* **Architecture:**  
  1. **Stream 1 (Accel):** A CNN (e.g., ResNet) processes the accelerometer data to extract a sequence of motion embeddings $E\_{accel} \\in \\mathbb{R}^{T' \\times D}$.  
  2. **Stream 2 (RSSI):** A Transformer Encoder processes the RSSI sequence (potentially patched) to extract context embeddings $E\_{rssi} \\in \\mathbb{R}^{T' \\times D}$.  
  3. Cross-Attention Module: The motion embeddings $E\_{accel}$ serve as the Queries ($Q$), while the context embeddings $E\_{rssi}$ serve as Keys ($K$) and Values ($V$).

     $$\\text{Fusion}(Q, K, V) \= \\text{softmax}\\left( \\frac{E\_{accel} E\_{rssi}^T}{\\sqrt{D}} \\right) E\_{rssi}$$

     This allows the network to dynamically "query" the spatial context based on the current motion state. For instance, if the accelerometer detects "walking," the attention mechanism focuses on RSSI features that discriminate "walking in hall" from "walking in room," suppressing irrelevant beacon noise.32

### **5.4. Multi-Stream Multimodal Factorized Transformer (MSMFT)**

This recent architecture (2024) addresses "modality entanglement," where one modality dominates the learning process. It introduces a third "fusion stream" that interacts with the independent modality streams via factorized attention, ensuring that unique features of both BLE (absolute position) and Accel (relative motion) are preserved while enabling deep interaction.35

## **6\. Recent Papers (2022-2024)**

The following papers represent significant advancements relevant to the user's specific problem:

1. "Advancing Activity Recognition with Multimodal Fusion and Transformer Techniques" (2025/2024) 32: This study explicitly proposes a Transformer-based attention mechanism for fusing heterogeneous sensor data. It demonstrates that cross-attention fusion outperforms simple concatenation on benchmarks like Extrasensory, directly validating the Intermediate Fusion strategy recommended above.  
2. "CoSS: Co-optimizing Sensor and Sampling Rate" (2024) 36: This paper introduces a framework for dynamically selecting sensor modalities and sampling rates. It is highly relevant for the user's "future" accelerometer integration, suggesting that the system could learn to downsample the accelerometer when the user is static (detected via RSSI), saving energy.  
3. "FFTNet: Frequency–Time Hybrid Architecture" (2024) 27: This work proposes a hybrid architecture that processes time-domain and frequency-domain features in parallel branches. This supports the recommendation to include FFT-based features for RSSI to capture periodic shadowing effects.  
4. "Sensor-Adaptive Multimodal Fusion (SAMFusion)" (2024) 37: Although focused on 3D object detection, the core contribution—a learned weighting scheme that adapts to sensor degradation (e.g., using Radar when Lidar fails)—is transferable to HAR. It suggests a mechanism where the model can learn to ignore BLE features when RSSI variance is too high (unstable signal).

## **7\. PyTorch Implementation Recommendations**

Implementing this multimodal system requires handling data with potentially different lengths and sampling rates.

### **7.1. Custom Dataset and collate\_fn**

Standard PyTorch DataLoaders expect uniform tensor sizes. For multimodal data, a custom collate\_fn is necessary to pad sequences or return dictionary objects.

Python

import torch  
from torch.utils.data import Dataset, DataLoader  
from torch.nn.utils.rnn import pad\_sequence

class MultimodalHARDataset(Dataset):  
    def \_\_init\_\_(self, rssi\_list, accel\_list, labels):  
        \# rssi\_list: List of tensors  
        \# accel\_list: List of tensors  
        self.rssi \= rssi\_list  
        self.accel \= accel\_list  
        self.labels \= labels

    def \_\_len\_\_(self):  
        return len(self.labels)

    def \_\_getitem\_\_(self, idx):  
        return {  
            "rssi": self.rssi\[idx\],  
            "accel": self.accel\[idx\],  
            "label": self.labels\[idx\]  
        }

def multimodal\_collate(batch):  
    \# Separate modalities  
    rssi\_seqs \= \[item\['rssi'\].transpose(0, 1) for item in batch\] \# Transpose to for padding  
    accel\_seqs \= \[item\['accel'\].transpose(0, 1) for item in batch\]  
    labels \= torch.tensor(\[item\['label'\] for item in batch\])

    \# Pad sequences to max length in batch (if variable)  
    \# batch\_first=True \-\>  
    rssi\_padded \= pad\_sequence(rssi\_seqs, batch\_first=True, padding\_value=0.0)  
    accel\_padded \= pad\_sequence(accel\_seqs, batch\_first=True, padding\_value=0.0)

    \# Transpose back to if using 1D-CNN  
    return {  
        "rssi": rssi\_padded.transpose(1, 2),   
        "accel": accel\_padded.transpose(1, 2)  
    }, labels

\# Usage  
\# train\_loader \= DataLoader(dataset, batch\_size=32, collate\_fn=multimodal\_collate)

### **7.2. Fusion Model Skeleton**

The following code sketches an Intermediate Fusion model combining a CNN for Accel and a Transformer for RSSI.

Python

import torch  
import torch.nn as nn

class FusionHAR(nn.Module):  
    def \_\_init\_\_(self, num\_classes, rssi\_dim=25, accel\_dim=3):  
        super().\_\_init\_\_()  
          
        \# \--- Stream 1: Accelerometer (CNN) \---  
        self.accel\_net \= nn.Sequential(  
            nn.Conv1d(accel\_dim, 64, kernel\_size=7, padding=3),  
            nn.BatchNorm1d(64),  
            nn.ReLU(),  
            nn.MaxPool1d(2),  
            nn.Conv1d(64, 128, kernel\_size=5, padding=2),  
            nn.BatchNorm1d(128),  
            nn.ReLU(),  
            nn.AdaptiveAvgPool1d(1) \# Output: \-\>  
        )  
          
        \# \--- Stream 2: RSSI (Transformer) \---  
        \# Input: \-\> Transpose \-\>  
        self.rssi\_embedding \= nn.Linear(rssi\_dim, 64)   
        self.pos\_encoder \= nn.Parameter(torch.randn(1, 500, 64)) \# Learnable pos enc  
          
        encoder\_layer \= nn.TransformerEncoderLayer(d\_model=64, nhead=4, batch\_first=True)  
        self.rssi\_transformer \= nn.TransformerEncoder(encoder\_layer, num\_layers=2)  
          
        \# \--- Fusion Head \---  
        self.fusion\_layer \= nn.Sequential(  
            nn.Linear(128 \+ 64, 128), \# Concat Accel(128) \+ RSSI(64)  
            nn.ReLU(),  
            nn.Dropout(0.3),  
            nn.Linear(128, num\_classes)  
        )

    def forward(self, x\_rssi, x\_accel):  
        \# x\_accel:  
        \# x\_rssi:   
          
        \# 1\. Process Accel  
        accel\_feat \= self.accel\_net(x\_accel).squeeze(-1) \#  
          
        \# 2\. Process RSSI  
        x\_rssi \= x\_rssi.transpose(1, 2) \#  
        rssi\_emb \= self.rssi\_embedding(x\_rssi) \#  
        \# Add positional encoding (slice to current sequence length)  
        rssi\_emb \= rssi\_emb \+ self.pos\_encoder\[:, :rssi\_emb.size(1), :\]  
          
        rssi\_out \= self.rssi\_transformer(rssi\_emb)  
        \# Pooling: Take mean over time dimension  
        rssi\_feat \= rssi\_out.mean(dim=1) \#  
          
        \# 3\. Fusion  
        combined \= torch.cat((accel\_feat, rssi\_feat), dim=1)  
        return self.fusion\_layer(combined)

### **7.3. Using tsai for ROCKET**

For the ROCKET baseline, utilizing the tsai library is highly recommended over writing scratch implementations due to optimization.

Python

from tsai.all import \*

\# Prepare data as X:  
\# For multivariate ROCKET, simply stack RSSI and Accel (requires same length)  
X, y, splits \= get\_classification\_data(..., split\_data=False)

\# MiniRocket Classifier  
tfms \=  
batch\_tfms \= TSStandardize(by\_sample=True)  
clf \= TSClassifier(X, y, splits=splits, arch=MiniRocketClassifier,   
                   tfms=tfms, batch\_tfms=batch\_tfms, metrics=accuracy)  
clf.fit\_one\_cycle(10, 3e-4)

## **8\. Conclusion**

To address the requirements of recognizing human activity using 25-beacon BLE RSSI and accelerometer data over a 5-second window, the analysis points to a tiered strategy. For immediate, robust baselines, **MiniRocket** offers superior efficiency and SOTA accuracy, capable of handling the high dimensionality of the input without extensive hyperparameter tuning. For a deployable deep learning solution, an **Intermediate Fusion architecture** utilizing a **1D-CNN (InceptionTime)** for the inertial stream and a **Transformer Encoder** for the RSSI stream is recommended. This hybrid approach leverages the strengths of convolutions for local kinematic features and self-attention for global spatial context. The 5-second window should be maintained for RSSI stability but implemented with a **75% overlap** to ensure system responsiveness. Finally, feature engineering must move beyond raw RSSI to include **Differential RSSI** and **Level Crossing Rates** to mitigate the stochastic nature of the wireless channel.

#### **Nguồn trích dẫn**

1. A 1-D CNN inference engine for constrained platforms This work is supported by the German Research Foundation (Deutsche Forschungsgemeinschaft, DFG) as part of SPP 2378 (Resilient Worlds \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/html/2501.17269v1](https://arxiv.org/html/2501.17269v1)  
2. (PDF) State of the Art of Deep Neural Networks Models \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/370756673\_State\_of\_the\_Art\_of\_Deep\_Neural\_Networks\_Models](https://www.researchgate.net/publication/370756673_State_of_the_Art_of_Deep_Neural_Networks_Models)  
3. Performance Analysis of State-of-the-Art CNN Architectures for LUNA16 \- PMC \- NIH, truy cập vào tháng 12 10, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9227226/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9227226/)  
4. PyTorch implementation of InceptionTime model for multivariate time series classification. \- GitHub, truy cập vào tháng 12 10, 2025, [https://github.com/flaviagiammarino/inception-time-pytorch](https://github.com/flaviagiammarino/inception-time-pytorch)  
5. \[1909.04939\] InceptionTime: Finding AlexNet for Time Series Classification \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/abs/1909.04939](https://arxiv.org/abs/1909.04939)  
6. InceptionTime – tsai \- GitHub Pages, truy cập vào tháng 12 10, 2025, [https://timeseriesai.github.io/tsai//models.inceptiontime.html](https://timeseriesai.github.io/tsai//models.inceptiontime.html)  
7. ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels \- François Petitjean, truy cập vào tháng 12 10, 2025, [https://francois-petitjean.com/Research/Dempster2020-Rocket.pdf](https://francois-petitjean.com/Research/Dempster2020-Rocket.pdf)  
8. RocketClassifier — sktime documentation, truy cập vào tháng 12 10, 2025, [https://www.sktime.net/en/stable/api\_reference/auto\_generated/sktime.classification.kernel\_based.RocketClassifier.html](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.classification.kernel_based.RocketClassifier.html)  
9. 10\_Time\_Series\_Classification\_, truy cập vào tháng 12 10, 2025, [https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial\_nbs/10\_Time\_Series\_Classification\_and\_Regression\_with\_MiniRocket.ipynb](https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb)  
10. MiniRocketMultivariateVariable — sktime documentation, truy cập vào tháng 12 10, 2025, [https://www.sktime.net/en/latest/api\_reference/auto\_generated/sktime.transformations.panel.rocket.MiniRocketMultivariateVariable.html](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.transformations.panel.rocket.MiniRocketMultivariateVariable.html)  
11. Comparative Analysis of CNN, RNN, LSTM, and Transformer Architectures in Deep Learning | Educational Administration: Theory and Practice, truy cập vào tháng 12 10, 2025, [https://kuey.net/index.php/kuey/article/view/10364](https://kuey.net/index.php/kuey/article/view/10364)  
12. How to make a PyTorch Transformer for time series forecasting \- Towards Data Science, truy cập vào tháng 12 10, 2025, [https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e/](https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e/)  
13. Scalable Transformer for High Dimensional Multivariate Time Series Forecasting \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/html/2408.04245v1](https://arxiv.org/html/2408.04245v1)  
14. A Transformer-based Framework for Multivariate Time Series Representation Learning, truy cập vào tháng 12 10, 2025, [https://openreview.net/forum?id=lE1AB4stmX](https://openreview.net/forum?id=lE1AB4stmX)  
15. A Transformer-based Framework for Multivariate Time Series Representation Learning, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/353907780\_A\_Transformer-based\_Framework\_for\_Multivariate\_Time\_Series\_Representation\_Learning](https://www.researchgate.net/publication/353907780_A_Transformer-based_Framework_for_Multivariate_Time_Series_Representation_Learning)  
16. Impact of Sliding Window Length in Indoor Human Motion Modes ..., truy cập vào tháng 12 10, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6021910/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6021910/)  
17. Effects of sliding window variation in the performance of acceleration-based human activity recognition using deep learning models \- Instituto Superior Técnico \- Universidade de Lisboa, truy cập vào tháng 12 10, 2025, [http://web.tecnico.ulisboa.pt/daniel.s.lopes/papers/sliding-window-PeerJ-computer-science-2022.pdf](http://web.tecnico.ulisboa.pt/daniel.s.lopes/papers/sliding-window-PeerJ-computer-science-2022.pdf)  
18. Smartphone-Based Unconstrained Step Detection Fusing a Variable Sliding Window and an Adaptive Threshold \- MDPI, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/2072-4292/14/12/2926](https://www.mdpi.com/2072-4292/14/12/2926)  
19. Feature Engineering for Time-Series Data: Methods and ..., truy cập vào tháng 12 10, 2025, [https://www.geeksforgeeks.org/data-analysis/feature-engineering-for-time-series-data-methods-and-applications/](https://www.geeksforgeeks.org/data-analysis/feature-engineering-for-time-series-data-methods-and-applications/)  
20. Outdoor Localization Using BLE RSSI and Accessible Pedestrian Signals for the Visually Impaired at Intersections \- MDPI, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/1424-8220/22/1/371](https://www.mdpi.com/1424-8220/22/1/371)  
21. Mobile Broadband, truy cập vào tháng 12 10, 2025, [https://maxwell.sze.hu/\~ungert/Radiorendszerek\_satlab/Segedanyagok/Ajanlott\_irodalom/Springer.Mobile.Broadband.Including.WiMAX.And.LTE.Feb.2009.eBook-ELOHiM.pdf](https://maxwell.sze.hu/~ungert/Radiorendszerek_satlab/Segedanyagok/Ajanlott_irodalom/Springer.Mobile.Broadband.Including.WiMAX.And.LTE.Feb.2009.eBook-ELOHiM.pdf)  
22. KEH-Gait: Towards a Mobile Healthcare User Authentication System by Kinetic Energy Harvesting | Request PDF \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/310124228\_KEH-Gait\_Towards\_a\_Mobile\_Healthcare\_User\_Authentication\_System\_by\_Kinetic\_Energy\_Harvesting](https://www.researchgate.net/publication/310124228_KEH-Gait_Towards_a_Mobile_Healthcare_User_Authentication_System_by_Kinetic_Energy_Harvesting)  
23. An Entropy Source based on the Bluetooth Received Signal Strength Indicator \- SOL-SBC, truy cập vào tháng 12 10, 2025, [https://sol.sbc.org.br/index.php/sbseg/article/download/19231/19060/](https://sol.sbc.org.br/index.php/sbseg/article/download/19231/19060/)  
24. Programme for the 14th European Conference on Antennas and, truy cập vào tháng 12 10, 2025, [https://vbn.aau.dk/ws/portalfiles/portal/336211192/EuCAP\_2020\_Technical\_Programme\_ver\_2020.03.31.pdf](https://vbn.aau.dk/ws/portalfiles/portal/336211192/EuCAP_2020_Technical_Programme_ver_2020.03.31.pdf)  
25. Evaluation of entropy features and classifier performance in person authentication using resting-state EEG \- NIH, truy cập vào tháng 12 10, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12623379/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12623379/)  
26. \[PDF\] Spectral entropy based feature for robust ASR \- Semantic Scholar, truy cập vào tháng 12 10, 2025, [https://www.semanticscholar.org/paper/Spectral-entropy-based-feature-for-robust-ASR-Misra-Ikbal/52d231d5cfe38453f5760e0eb1c0dcb50e004a6d](https://www.semanticscholar.org/paper/Spectral-entropy-based-feature-for-robust-ASR-Misra-Ikbal/52d231d5cfe38453f5760e0eb1c0dcb50e004a6d)  
27. FFTNet: Fusing Frequency and Temporal Awareness in Long-Term Time Series Forecasting, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/2079-9292/14/7/1303](https://www.mdpi.com/2079-9292/14/7/1303)  
28. (PDF) Combining RSSI and Accelerometer Features for Room-Level ..., truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/350848586\_Combining\_RSSI\_and\_Accelerometer\_Features\_for\_Room-Level\_Localization](https://www.researchgate.net/publication/350848586_Combining_RSSI_and_Accelerometer_Features_for_Room-Level_Localization)  
29. Early Fusion vs. Late Fusion in Multimodal Data Processing \- GeeksforGeeks, truy cập vào tháng 12 10, 2025, [https://www.geeksforgeeks.org/deep-learning/early-fusion-vs-late-fusion-in-multimodal-data-processing/](https://www.geeksforgeeks.org/deep-learning/early-fusion-vs-late-fusion-in-multimodal-data-processing/)  
30. Sensor Data Acquisition and Multimodal Sensor Fusion for Human Activity Recognition Using Deep Learning \- NIH, truy cập vào tháng 12 10, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6479605/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6479605/)  
31. Early and Late Fusion for Multimodal Aggression Prediction in Dementia Patients: A Comparative Analysis \- MDPI, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/2076-3417/15/11/5823](https://www.mdpi.com/2076-3417/15/11/5823)  
32. Advancing Activity Recognition With Multimodal Fusion and Transformer Techniques, truy cập vào tháng 12 10, 2025, [https://ieeexplore.ieee.org/document/10955127/](https://ieeexplore.ieee.org/document/10955127/)  
33. Advancing Activity Recognition With Multimodal Fusion and Transformer Techniques | Request PDF \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/390578822\_Advancing\_Activity\_Recognition\_with\_Multimodal\_Fusion\_and\_Transformer\_Techniques](https://www.researchgate.net/publication/390578822_Advancing_Activity_Recognition_with_Multimodal_Fusion_and_Transformer_Techniques)  
34. Fusion-driven multimodal learning for biomedical time series in surgical care \- Frontiers, truy cập vào tháng 12 10, 2025, [https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1605406/full](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1605406/full)  
35. MSMFT: Multi-Stream Multimodal Factorized Transformer for Human Activity Recognition, truy cập vào tháng 12 10, 2025, [https://ieeexplore.ieee.org/document/10850630/](https://ieeexplore.ieee.org/document/10850630/)  
36. CoSS: Co-optimizing Sensor and Sampling Rate for Data-Efficient AI in Human Activity Recognition \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/html/2401.05426v1](https://arxiv.org/html/2401.05426v1)  
37. SAMFusion: Sensor-Adaptive Multimodal Fusion for 3D Object Detection in Adverse Weather \- Princeton Computational Imaging Lab, truy cập vào tháng 12 10, 2025, [https://light.princeton.edu/publication/samfusion/](https://light.princeton.edu/publication/samfusion/)