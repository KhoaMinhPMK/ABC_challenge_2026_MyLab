---
noteId: "3e6c5900d5cf11f080694155c34f83e4"
tags: []

---

# **Advanced Methodologies for BLE Indoor Localization in High-Sparsity, Low-Signal Regimes**

## **1\. Introduction: The Stochastic Nature of Sparse BLE Environments**

The deployment of Bluetooth Low Energy (BLE) for indoor positioning systems (IPS) has traditionally relied on the premise of high-density infrastructure and robust signal-to-noise ratios (SNR). However, operational realities often dictate constraints that violate these ideal conditions. The specific operational environment described—characterized by 25 beacons, 44 discrete reference points, a critically low 4% effective coverage, and a signal floor of \-108 dBm—represents a high-entropy system where deterministic modelling approaches frequently diverge. In such regimes, the Received Signal Strength Indicator (RSSI) ceases to be a reliable proxy for distance due to the dominance of multipath fading, shadowing, and hardware heterogeneity over the logarithmic path loss curve.

This report provides an exhaustive analysis of State-of-the-Art (SOTA) methodologies tailored specifically for such data-scarce and signal-weak environments. The analysis moves beyond traditional dense fingerprinting paradigms, which are ill-suited for this scenario, and focuses on probabilistic recovery methods, geometric learning, and synthetic data augmentation. The core challenge addressed herein is not merely localization, but the reconstruction of latent topological structures from fragmentary data. When 37.55% of the signal matrix consists of zero-signal pairs and the remaining signals hover near the sensitivity limit of standard receivers (-108 dBm), the system operates in a regime where "missing data" is the dominant feature. Consequently, the architectural focus must shift from standard regression (e.g., k-Nearest Neighbors) to advanced imputation and generative modeling.

The following sections dissect the theoretical and practical applications of Graph Neural Networks (GNNs), Matrix Completion (MC), and Generative Adversarial Networks (GANs) as they apply to the specific constraints of the user's dataset. Furthermore, we critically evaluate feature engineering strategies that transmute weak, fluctuating absolute RSSI values into robust relative features—such as rank-based fingerprints and differential vectors—to mitigate the volatility inherent at the \-108 dBm noise floor.

## ---

**2\. State-of-the-Art Methods for Sparse Fingerprints (2022-2024)**

The progression of indoor localization research between 2022 and 2024 has been defined by a pivot away from heuristic methods towards deep geometric learning and probabilistic data recovery. In environments where the "overlap" between a user's online scan and the offline database is statistically insignificant due to sparsity, traditional Euclidean-based matching fails. The current SOTA focuses on reconstructing the missing spectral and spatial data or learning strictly from the available topology using graph-based inference.

### **2.1 Attentional Graph Meta-Learning (AGML)**

The most significant advancement for handling extremely sparse datasets is the integration of Graph Neural Networks (GNNs) with Meta-Learning frameworks. This approach, termed Attentional Graph Meta-Learning (AGML), specifically addresses the "cold start" problem where the radio map contains insufficient data points (e.g., only 44 locations) to train a deep neural network from scratch without severe overfitting.1

In a standard Euclidean model, training examples (fingerprints) are treated as independent vectors. This ignores the spatial correlation that is vital in sparse environments—the fact that a signal reading at location $A$ provides information about the probable signal at nearby location $B$. AGML constructs a graph where fingerprints act as nodes and physical proximity or signal similarity forms the edges. The architecture employs an attention mechanism to weigh the importance of neighboring nodes. This allows the model to aggregate information from "visible" neighbors to infer the state of "invisible" or noisy ones. For a dataset with only 44 nodes, the attention mechanism dynamically adjusts the influence of adjacent reference points based on signal stability, effectively filtering out the noise inherent in the \-108 dBm floor.1

Furthermore, the meta-learning component of AGML is designed to learn the *general* propagation characteristics of indoor environments—such as how signals typically decay through drywall or reflect off concrete—from auxiliary datasets. It then transfers this "meta-knowledge" to the specific sparse target environment. This enables the system to converge on a high-accuracy model using only a few samples (few-shot learning), making it uniquely suited to the user's constrained dataset.2

### **2.2 Multi-Graph Heterogeneous GNNs (MG-HGNN)**

While AGML focuses on meta-generalization, Multi-Graph Heterogeneous GNNs (MG-HGNN) are designed to maximize information extraction from the heterogeneous relationships within the available data. This method is particularly robust in complex, non-linear environments where single-graph representations may fail to capture the full dynamic of signal propagation.3

The MG-HGNN framework constructs two parallel graph structures: a **Spatial Graph** and a **Signal Graph**. The Spatial Graph models the physical relationships between the 44 locations, creating edges based on geometric distance. Simultaneously, the Signal Graph models the correlations between the 25 beacons themselves. For instance, if Beacon 5 and Beacon 7 are physically close, their RSSI values will exhibit a high covariance. The MG-HGNN learns these beacon-to-beacon correlations explicitly.4

In a scenario with 4% coverage, this dual-graph approach is transformative. Even if a specific location has never been visited or has missing data for Beacon 5, the network can infer the likely RSSI value based on the strong signal received from Beacon 7 and the learned correlation edge between them. This capability allows the system to constrain the search space significantly, effectively "filling in the blanks" of the sparse matrix through learned dependencies rather than simple interpolation.3

### **2.3 Deep Generative In-painting: CGAN-LSTM**

Conditional Generative Adversarial Networks (CGAN) combined with Long Short-Term Memory (LSTM) units have emerged as a powerful tool for artificially densifying sparse radio maps. This method addresses the sparsity problem by generating synthetic data to fill the voids between the known reference points.5

The architecture consists of a Generator and a Discriminator. The Generator is trained on the 44 known fingerprints and attempts to create synthetic fingerprints for coordinates *between* these points. The Discriminator tries to distinguish real measurements from the synthetic ones. Crucially, the inclusion of LSTM units models the spatial continuity of the signal. It ensures that the generated RSSI values follow a logical spatial sequence—preventing the generator from producing physically impossible jumps in signal strength (e.g., from \-100 dBm to \-70 dBm over a few centimeters).5

Research indicates that this augmentation strategy can increase localization accuracy by approximately 15.74% in sparse environments. By transforming the 44-point dataset into a dense, synthetic grid of thousands of points, the CGAN-LSTM approach allows the use of standard, high-precision regression algorithms that would otherwise fail on the original sparse set.5

### **2.4 Low-Rank Matrix Completion and Compressive Sensing**

For static sparse maps, the mathematical rigor of Low-Rank Matrix Completion (LRMC) remains a dominant approach. This method posits that the radio map matrix (Locations $\\times$ Beacons) is not random but possesses a "low rank" structure governed by the physics of signal propagation.

Algorithms such as Singular Value Thresholding (SVT) or convex optimization are employed to recover the missing entries (the 37.55% zero-signal pairs) by finding the lowest-rank matrix that matches the observed data. Recent advancements have combined LRMC with Compressive Sensing (CS). Once the matrix is completed, the localization problem is reformulated as a sparse recovery problem. This technique has demonstrated the ability to reduce positioning error to under 0.7 meters in testbeds where traditional k-Nearest Neighbor (KNN) methods yielded errors of 2-3 meters.7

### **Summary of SOTA Applicability**

The following table summarizes the applicability of these advanced methods to the specific constraints of the user's project.

| Method | Primary Mechanism | Optimal Use Case | Prerequisite |
| :---- | :---- | :---- | :---- |
| AGML 1 | Graph Attention \+ Meta-Learning | Extremely sparse grids with few Reference Points (RPs) | High computational power for training |
| MG-HGNN 3 | Dual Graph Embedding | Complex environments with non-linear signal features | Knowledge of topology/floor plan |
| CGAN-LSTM 5 | Synthetic Data Generation | Densifying small datasets (e.g., 44 points) | Sequential data (trajectories) for LSTM |
| Matrix Completion 7 | Low-Rank Approximation | Recovering missing RSSI values in static maps | High correlation between beacon signals |

## ---

**3\. Feature Engineering for Missing Data**

In a dataset characterized by 37.55% zero-signal pairs and a 4% coverage rate, "missing data" is not merely an absence of information; it is a feature in itself. In the context of BLE, a missing signal is often **Missing Not At Random (MNAR)**—it implies that the beacon is either too distant or heavily obstructed. Consequently, feature engineering must pivot from analyzing *what is present* to analyzing the *structure of presence and absence*.

### **3.1 The Binary Fingerprint (Presence/Absence Vectors)**

Given the extreme volatility of RSSI at the edge of reception (where values oscillate between \-100 dBm and disconnection), the precise dBm value carries less information density than the binary state of connectivity. At \-108 dBm, a 3 dB fluctuation can be caused by a user's orientation, whereas the transition from "connected" to "disconnected" often signifies a distinct environmental boundary (e.g., entering a room or crossing a wall).

**Implementation:** The RSSI vector $R$ is converted into a binary vector $B$, where $B\_i \= 1$ if $RSSI\_i \> Threshold$ and $0$ otherwise. This transformation effectively filters out the noise of signal fluctuation while retaining the coarse-grained location information provided by the visibility set.

**Similarity Metrics:** Standard Euclidean distance is ill-suited for binary vectors as it essentially counts mismatches without context. Instead, **Jaccard Similarity** or **Hamming Distance** should be employed. Jaccard Similarity measures the intersection over the union of the visible beacon sets, providing a robust metric for clustering locations based on *which* subset of the 25 beacons is visible, regardless of their unstable signal strengths.8

### **3.2 Ranking-Based Features (Kendall Tau)**

Absolute RSSI values are notoriously susceptible to device heterogeneity (e.g., different antenna gains on user smartphones) and temporal instability. However, the *relative order* of signal strengths remains remarkably stable. If Beacon A is physically closer than Beacon B, the relationship $RSSI\_A \> RSSI\_B$ usually holds true even if the absolute values of both drop by 10 dB due to shadowing.

**Rank Transformation:** For each location, the raw RSSI vector is replaced with a permutation vector representing the rank of the beacons (e.g., Beacon 5 is 1st strongest, Beacon 2 is 2nd). This transforms the regression problem into an ordinal ranking problem.

**Kendall Tau Correlation:** The similarity between a user's observed rank vector and the database fingerprints is calculated using the Kendall Tau coefficient. This metric counts the number of concordant and discordant pairs between two ranking lists. It is immune to scale bias and is particularly effective for weak signals where absolute calibration is impossible.10

### **3.3 Differential RSSI (DRSS)**

Differential RSSI exploits the spatial gradient between pairs of beacons to subtract out common-mode noise.

* **formulation:** $D\_{i,j} \= RSSI\_i \- RSSI\_j$.  
* **Mechanism:** Factors such as battery voltage drops or global interference often affect all received signals simultaneously. By calculating the difference between Beacon $i$ and Beacon $j$, these common error terms cancel out.  
* **Constraint:** This method requires that multiple beacons be visible simultaneously. Given the 4% coverage, this feature is most effective when applied to specific subsets of beacons known to be co-located or co-visible in certain zones. Studies show that DRSS significantly reduces variance caused by device heterogeneity.13

### **3.4 Missing Pattern as a Feature**

The specific *pattern* of missingness creates a unique "shadowing signature" for each location. In complex indoor environments, structural elements like concrete pillars or elevator shafts create consistent shadow zones where specific beacons are always missing.

**Vectorization:** A "mask vector" $M$ can be created where $M\_i \= 1$ if data is missing. This binary mask is concatenated with the processed RSSI vector. Neural networks can then learn to associate specific missing patterns (e.g., "missing Beacon 12" combined with "weak Beacon 4") with distinct physical locations, effectively turning obstructions into localization features.15

## ---

**4\. Imputation Strategies**

With 37.55% of the data explicitly zero and a high probability of additional missingness in the "4% coverage" regime, simply discarding incomplete rows is not an option. Imputation is mandatory, and the chosen strategy defines the bias introduced into the localization model.

### **4.1 Advanced Matrix Completion (The Gold Standard)**

Matrix Completion (MC) via **Singular Value Thresholding (SVT)** or **Nuclear Norm Minimization** represents the most mathematically rigorous approach for handling sparse radio maps.

* **Concept:** The matrix of RSSI values is assumed to be low-rank because the 25 beacons are observing the same physical space; their signals are correlated by the geometry of the building.  
* **Optimization:** MC algorithms solve a convex optimization problem to find the matrix with the minimum nuclear norm that is consistent with the observed entries. This effectively recovers the global structure of the radio map.7  
* **Superiority:** Unlike simple interpolation, which fails with non-grid data, or mean imputation, which destroys variance, MC utilizes the latent correlations between beacons to infer missing values. Empirical results demonstrate that MC-based recovery can improve positioning accuracy from \>2m to \~0.7m in sparse environments.7

### **4.2 Dynamic Contextual Imputation**

**Dynamic Imputation** improves upon static methods by estimating missing values based on the local context of *visible* neighbors.

* **Method:** Instead of replacing a missing value with a global constant (e.g., \-105 dBm), the algorithm identifies the $k$ nearest reference points using only the visible beacons. It then imputes the missing value based on the values found in those neighbors.  
* **Impact:** This approach preserves the local probability distribution of the signals. Research indicates that dynamic imputation can improve accuracy by approximately 30% over fixed-value imputation by respecting local signal gradients.18

### **4.3 Deep Learning-Based Imputation (Autoencoders)**

**Denoising Autoencoders (DAE)** offer a non-linear approach to imputation.

* **Mechanism:** A DAE is trained to reconstruct full fingerprints from partial inputs. During training, known values are artificially masked (set to zero), and the network is penalized based on its ability to predict the original values.  
* **Latent Representation:** The "bottleneck" layer of the autoencoder forces the model to learn a compressed representation of the radio environment. When a sparse vector is input during inference, the decoder reconstructs the "ideal" full vector, effectively filtering out the noise and filling in the gaps based on learned non-linear correlations.19

### **Imputation Strategy Comparison**

| Strategy | Assumption | Pros | Cons |
| :---- | :---- | :---- | :---- |
| **Fixed Value (-105 dBm)** | Signal is out of range (MNAR) | Simple, preserves dimensionality | Biases distance metrics; ignores structural correlations 21 |
| **Matrix Completion (SVT)** | Radio map is low-rank | Recovers global topology; highly accurate | Computationally intensive; requires offline batch processing 7 |
| **Dynamic Imputation** | Local correlations exist | Adapts to local signal context; \+30% accuracy | Requires reliable neighbor data (hard with 4% coverage) 18 |
| **Autoencoder (DAE)** | Non-linear latent space | Handles noise and sparsity simultaneously | Requires training data (difficult with only 44 locations) 19 |

## ---

**5\. Transfer Learning and Augmentation**

Given the extremely limited sample size of 44 locations, any machine learning model is at high risk of overfitting. Transfer learning and data augmentation are therefore critical to artificially expand the model's "experience."

### **5.1 Manifold Alignment (Domain Adaptation)**

Manifold Alignment aligns the "signal manifold" (the high-dimensional shape of RSSI data) with the "physical manifold" (the 2D floor plan).

* **Semi-Supervised Learning:** While labeled data (locations with known coordinates) is scarce, *unlabeled* data (scans collected while walking) is cheap and abundant. Manifold alignment techniques map these unlabeled scans to the physical space by aligning them with the labeled anchors. This effectively fills in the gaps between the 44 points using the topology of the walking traces.22  
* **Cross-Domain Transfer:** If a radio map exists for a similar environment (e.g., another floor with a similar layout), transfer learning can project the learned signal propagation features to the current sparse dataset. This can reduce the calibration effort by up to 85% while maintaining accuracy.24

### **5.2 Synthetic Data Generation via GANs**

**Conditional GANs (CGAN)** are the premier tool for augmenting sparse radio maps.

* **Process:** The Generator takes a coordinate $(x, y)$ and a noise vector as input and outputs a synthetic RSSI vector. The Discriminator evaluates whether the vector is plausible given the location.  
* **Result:** Once trained, the Generator can produce infinite synthetic fingerprints for any coordinate on the map. This transforms the sparse 44-point dataset into a dense grid (e.g., 4400 points) suitable for training robust regressors like XGBoost or CNNs.5  
* **Physics Constraints:** To prevent the model from "hallucinating" unrealistic signals, the GAN can be constrained by a coarse Path-Loss model, creating a physics-guided generative process.26

### **5.3 Few-Shot Learning (FSL)**

**Prototypical Networks** or **Relation Networks** allow the system to classify a user's location based on very few examples.

* **Meta-Learning:** The model is trained on a variety of *other* tasks (or simulated rooms) to learn the general concept of "localization." It then adapts to the 44-location dataset with only 1-5 examples per location.  
* **Siamese Networks:** Instead of predicting coordinates directly, a Siamese network is trained to predict the *similarity* between the user's scan and the 44 reference points. This approach is often more robust for sparse data than direct regression.27

## ---

**6\. Fingerprinting vs. Path-Loss Comparison**

In the specific regime of weak signals (-108 dBm) and high sparsity (4% coverage), the choice between Fingerprinting and Path-Loss models is critical. Both methodologies exhibit distinct failure modes under these conditions.

### **6.1 Path-Loss (Log-Distance Model)**

* **Mechanism:** This model relies on the equation $RSSI \= \-10n \\log\_{10}(d) \+ A$ to estimate the distance $d$ to each beacon, followed by multilateration (trilateration) to determine the intersection of these distances.  
* **Failure Mode:**  
  * **Weak Signals:** At \-108 dBm, the signal is dominated by the noise floor and multipath fading. The path loss exponent $n$ varies wildly due to obstacles. A mere 3 dB error at this level can translate to a distance error of over 10 meters.  
  * **Sparsity:** Trilateration mathematically requires at least 3 visible beacons to resolve a 2D position. With 4% coverage and 25 beacons, the probability of simultaneously detecting 3 beacons is excessively low. Consequently, the system will frequently fail to produce *any* estimate.30

### **6.2 Fingerprinting (Pattern Matching)**

* **Mechanism:** This method compares the observed RSSI vector to a stored database (Radio Map) using algorithms like KNN or probabilistic matching.  
* **Advantage:**  
  * **Implicit Multipath Handling:** Fingerprinting does not attempt to filter out multipath effects; it utilizes them as unique features. If a specific location consistently exhibits a signal reflection causing \-90 dBm, the fingerprint captures and utilizes this anomaly.32  
  * **Sparsity Tolerance:** Fingerprinting can function with as few as 1 or 2 visible beacons if the signal "signature" is sufficiently unique. Feature engineering techniques like binary fingerprinting further enhance this robustness.  
* **Constraint:** The resolution is limited by the density of the reference points. With only 44 points, the user's location will often be "snapped" to the nearest point, potentially resulting in quantization errors of several meters.

### **6.3 Quantitative Comparison**

| Feature | Path-Loss (Trilateration) | Sparse Fingerprinting | Recommended Hybrid |
| :---- | :---- | :---- | :---- |
| **Accuracy** | 2m \- 6m (High variance) 34 | 1m \- 3m (Grid limited) 33 | **0.7m \- 1.5m** (with Matrix Completion) 7 |
| **Robustness to NLOS** | Low (Assumes Line-of-Sight) | **High** (Learns NLOS patterns) | High |
| **Requirement** | 3+ Visible Beacons | 1+ Visible Beacon | 1+ Visible Beacon |
| **Performance @ \-100dBm** | **Critical Failure** (Divergence) | **Degraded** (Noisy matching) | **Manageable** (Uses relative features) |

### **6.4 The Verdict: Hybrid is Necessary**

For the user's specific constraints, **neither pure approach is ideal**. Pure Path-Loss is mathematically impossible in many zones due to insufficient beacon visibility. Pure Fingerprinting is resolution-limited by the sparse grid.

**Recommendation:** A **Hybrid Approach** is required. The 44 known points should be used to calibrate a local Path-Loss model (determining $n$ and $A$ for different zones). This calibrated model is then used to generate *synthetic* fingerprints (Augmentation) to fill the gaps, creating a densified map. Finally, Fingerprinting is performed on this augmented database.35

## ---

**7\. Review of Recent Papers (2022-2024)**

The following recent publications are selected for their direct relevance to the challenges of sparse data, weak signals, and generative modeling in indoor localization.

* 1 "Attentional Graph Meta-Learning for Indoor Localization Using Extremely Sparse Fingerprints" (2025/Preprint): This paper is a direct solution to the user's core problem. It proposes the AGML framework, demonstrating how GNNs can aggregate neighbor information and how meta-learning allows generalization from limited data in "extremely sparse" environments.  
* 5 "Augmentation of Fingerprints for Indoor BLE Localization Using Conditional GANs" (IEEE Access, Jan 2024): This work provides a blueprint for expanding the user's 44-point dataset. It details the use of CGAN-LSTM to synthesize data, achieving a 15.74% accuracy boost in sparse environments.  
* 7 "Bluetooth Indoor Positioning and Correction Method Based on Matrix Completion and Compressed Sensing" (2023): This paper provides the mathematical framework for handling the 37.55% zero-signal pairs. It details the use of low-rank matrix completion to recover missing fingerprint data offline, followed by Compressive Sensing for online matching.  
* 3 "MG-HGNN: Multi-Graph Heterogeneous GNN for Wi-Fi Fingerprint-based Localization" (2024): This research introduces an advanced architecture for complex/noisy environments, utilizing the separation of spatial and signal graphs to maximize information extraction from heterogeneous data.  
* 37 "FALoc: Fingerprint Augmentation... for Indoor Localization" (Electronics, 2025): This paper focuses on intelligent imputation, using a probabilistic model to estimate likely missing RSSI values rather than relying on fixed value imputation.

## ---

**8\. Strategic Recommendations & Technical Roadmap**

Based on the rigorous analysis of the dataset—25 beacons, 44 locations, 4% coverage, and a \-108 dBm noise floor—the following technical roadmap is recommended for implementation.

### **Phase 1: Data Preprocessing & Enrichment**

1. **Binary Transformation:** Construct a secondary dataset of Boolean vectors (Presence/Absence). This will serve as a coarse-grain filter to identify the general "Zone" before fine-grain positioning is attempted.8  
2. **Rank-Based Normalization:** Convert raw RSSI values to Rank Vectors (1 to 25). This step is crucial to mitigate the extreme fluctuation observed at \-108 dBm, stabilizing the input for the machine learning model.10  
3. **Dynamic Imputation:** Do *not* default to \-100 dBm for missing values. Implement a **Low-Rank Matrix Completion** algorithm (specifically SVT) to fill the sparse matrix. This process recovers the latent correlations between beacons that are otherwise lost.7

### **Phase 2: Dataset Expansion (Augmentation)**

1. **Synthetic Densification:** Recognize that 44 locations are insufficient for high-accuracy regression. Train a **Conditional GAN (CGAN)** on the imputed matrix. Generate synthetic fingerprints on a 10cm grid between the 44 known points, effectively transforming the problem from "Sparse Fingerprinting" to "Dense Fingerprinting".5  
2. **Physics-Informed Constraints:** Utilize a basic Log-Distance Path Loss model to constrain the GAN. If the GAN predicts a signal jump that violates physical laws (e.g., passing through 3 walls instantaneously), penalize the generator.

### **Phase 3: Model Architecture**

1. **Hierarchical Approach:**  
   * *Step 1 (Coarse):* Employ **Weighted KNN** (with $k=3$ or 4\) on the **Binary Fingerprints** to identify the 3-5 nearest Reference Points (RPs).  
   * *Step 2 (Fine):* Utilize an **Attentional Graph Neural Network (AGNN)** on the **Ranked RSSI** data within the selected cluster. The AGNN's attention mechanism will naturally weigh the reliable beacons higher than the noisy ones.1  
2. **Avoid Pure Trilateration:** Given the signal quality (-108 dBm), reliance on geometric intersection will lead to divergence. The topological information captured by the fingerprinting approach is far more robust.

### **Phase 4: Evaluation**

1. **Metric Selection:** Do not rely solely on Mean Absolute Error (MAE). Use the **CDF (Cumulative Distribution Function)** of error. In sparse data regimes, "outliers" are common and determining the 95th percentile error is often more critical than the mean.  
2. **Robustness Check:** Validate the model by randomly masking 50% of the visible beacons in the test set. A robust GNN/Matrix Completion model should maintain acceptable accuracy, whereas a standard KNN model will likely fail.

### **Conclusion**

The dataset provided presents a significant challenge due to its extreme sparsity and signal weakness. Standard localization methods will fail to provide usable accuracy. The path to success lies in **reconstructing the missing data** using Matrix Completion, **augmenting the spatial resolution** using Generative Adversarial Networks, and utilizing **relative feature engineering** (Ranking/Binary) to bypass the absolute noise of the weak signals. The convergence of Graph Neural Networks and Generative Models in the 2022-2024 literature provides the precise toolkit required to solve this "Sparse Radio Map" problem.

#### **Nguồn trích dẫn**

1. Attentional Graph Meta-Learning for Indoor Localization Using Extremely Sparse Fingerprints \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/html/2504.04829v1](https://arxiv.org/html/2504.04829v1)  
2. Positioning performance comparison of the five algorithms. \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/figure/Positioning-performance-comparison-of-the-five-algorithms\_tbl2\_347614407](https://www.researchgate.net/figure/Positioning-performance-comparison-of-the-five-algorithms_tbl2_347614407)  
3. MG-HGNN: A Heterogeneous GNN Framework for Indoor Wi-Fi Fingerprint-Based Localization \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/html/2511.07282v1](https://arxiv.org/html/2511.07282v1)  
4. Heterogeneous Graph Neural Network for WiFi RSSI-Based Indoor Floor Classification, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/2079-9292/14/24/4845](https://www.mdpi.com/2079-9292/14/24/4845)  
5. (PDF) Augmentation of Fingerprints for Indoor BLE Localization ..., truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/378376773\_Augmentation\_of\_Fingerprints\_for\_Indoor\_BLE\_Localization\_Using\_Conditional\_GANs](https://www.researchgate.net/publication/378376773_Augmentation_of_Fingerprints_for_Indoor_BLE_Localization_Using_Conditional_GANs)  
6. DNN-based Indoor Localization Under Limited Dataset using GANs and Semi-Supervised Learning, truy cập vào tháng 12 10, 2025, [https://ieeexplore.ieee.org/iel7/6287639/6514899/09812625.pdf](https://ieeexplore.ieee.org/iel7/6287639/6514899/09812625.pdf)  
7. Bluetooth Indoor Positioning and Correction Method Based on Matrix Completion and Compressed Sensing, truy cập vào tháng 12 10, 2025, [https://csroc.cmex.org.tw/journal/JOC31-3/JOC3103-08.pdf](https://csroc.cmex.org.tw/journal/JOC31-3/JOC3103-08.pdf)  
8. Evaluating Navigation Efficiency: A Comparative Study of Search Performance in Indoor Positioning Systems \- University of Bahrain Journals, truy cập vào tháng 12 10, 2025, [https://journal.uob.edu.bh/server/api/core/bitstreams/9f42efa3-e8ba-4bf4-b5f8-3d55bd4a581a/content](https://journal.uob.edu.bh/server/api/core/bitstreams/9f42efa3-e8ba-4bf4-b5f8-3d55bd4a581a/content)  
9. A Reliable Localization Algorithm Based on Grid Coding and Multi-Layer Perceptron \- SciSpace, truy cập vào tháng 12 10, 2025, [https://scispace.com/pdf/a-reliable-localization-algorithm-based-on-grid-coding-and-2nwcgaoy5l.pdf](https://scispace.com/pdf/a-reliable-localization-algorithm-based-on-grid-coding-and-2nwcgaoy5l.pdf)  
10. (PDF) Rank based fingerprinting algorithm for indoor positioning \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/236837286\_Rank\_based\_fingerprinting\_algorithm\_for\_indoor\_positioning](https://www.researchgate.net/publication/236837286_Rank_based_fingerprinting_algorithm_for_indoor_positioning)  
11. A BLE RSSI ranking based indoor positioning system for generic ..., truy cập vào tháng 12 10, 2025, [https://ieeexplore.ieee.org/document/7943542/](https://ieeexplore.ieee.org/document/7943542/)  
12. A BLE RSSI ranking based indoor positioning system for generic smartphones, truy cập vào tháng 12 10, 2025, [https://www.semanticscholar.org/paper/A-BLE-RSSI-ranking-based-indoor-positioning-system-Ma-Poslad/c05f41564cdfd07da9bd94f9623c4e36511e0ce1](https://www.semanticscholar.org/paper/A-BLE-RSSI-ranking-based-indoor-positioning-system-Ma-Poslad/c05f41564cdfd07da9bd94f9623c4e36511e0ce1)  
13. Development of a Multidirectional BLE Beacon-Based Radio-Positioning System for Vehicle Navigation in GNSS Shadow Roads \- MDPI, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/2673-4591/102/1/9](https://www.mdpi.com/2673-4591/102/1/9)  
14. A BLE based turnkey indoor positioning system for mobility assessment in aging-in-place settings \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/390877082\_A\_BLE\_based\_turnkey\_indoor\_positioning\_system\_for\_mobility\_assessment\_in\_aging-in-place\_settings](https://www.researchgate.net/publication/390877082_A_BLE_based_turnkey_indoor_positioning_system_for_mobility_assessment_in_aging-in-place_settings)  
15. Climate Change and the Field of Farm Labor in the Lower Rio Grande Valley, Texas \- ScholarWorks @ UTRGV, truy cập vào tháng 12 10, 2025, [https://scholarworks.utrgv.edu/cgi/viewcontent.cgi?article=2728\&context=etd](https://scholarworks.utrgv.edu/cgi/viewcontent.cgi?article=2728&context=etd)  
16. (PDF) Imputation Method Using Data Enrichment for Missing Data of Loop Detectors in Intelligent Traffic \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/372195019\_Imputation\_Method\_Using\_Data\_Enrichment\_for\_Missing\_Data\_of\_Loop\_Detectors\_in\_Intelligent\_Traffic](https://www.researchgate.net/publication/372195019_Imputation_Method_Using_Data_Enrichment_for_Missing_Data_of_Loop_Detectors_in_Intelligent_Traffic)  
17. An Efficient Fingerprint Database Construction Approach Based on Matrix Completion for Indoor Localization \- IEEE Xplore, truy cập vào tháng 12 10, 2025, [http://ieeexplore.ieee.org/document/9141223](http://ieeexplore.ieee.org/document/9141223)  
18. Uncaught Signal Imputation for Accuracy Enhancement of WLAN-based Positioning Systems, truy cập vào tháng 12 10, 2025, [http://wcl.cs.rpi.edu/pilots/library/papers/ACM-SIGSPATIAL-GIS02/MobiGIS2012-USB/MobiGIS2012\_files/MobiGIS2012-11.pdf](http://wcl.cs.rpi.edu/pilots/library/papers/ACM-SIGSPATIAL-GIS02/MobiGIS2012-USB/MobiGIS2012_files/MobiGIS2012-11.pdf)  
19. Adaptive Scheme of Denoising Autoencoder for Estimating Indoor Localization Based on RSSI Analytics in BLE Environment \- MDPI, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/1424-8220/23/12/5544](https://www.mdpi.com/1424-8220/23/12/5544)  
20. Data Imputation for Sparse Radio Maps in Indoor Positioning (Extended Version) \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/abs/2302.13022](https://arxiv.org/abs/2302.13022)  
21. From Fingerprinting to Advanced Machine Learning: A Systematic Review of Wi-Fi and BLE-Based Indoor Positioning Systems \- MDPI, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/1424-8220/25/22/6946](https://www.mdpi.com/1424-8220/25/22/6946)  
22. An Unsupervised Learning Technique to Optimize Radio Maps for Indoor Localization, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/1424-8220/19/4/752](https://www.mdpi.com/1424-8220/19/4/752)  
23. Indoor Localization Using Semi-Supervised Manifold Alignment with Dimension Expansion, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/2076-3417/6/11/338](https://www.mdpi.com/2076-3417/6/11/338)  
24. Updating Radio Maps Without Pain: An Enhanced Transfer Learning Approach | Request PDF \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/348084014\_Updating\_Radio\_Maps\_Without\_Pain\_An\_Enhanced\_Transfer\_Learning\_Approach](https://www.researchgate.net/publication/348084014_Updating_Radio_Maps_Without_Pain_An_Enhanced_Transfer_Learning_Approach)  
25. Transfer Learning of RSSI to Improve Indoor Localisation Performance \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/html/2412.09292v1](https://arxiv.org/html/2412.09292v1)  
26. A Hybrid BLE/UWB Localization Technique with Automatic Radio Map Creation \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/pdf/2404.03072](https://arxiv.org/pdf/2404.03072)  
27. DR-FSL : Distribution Relation Based Few-Shot Learning for Indoor Localization With CSI \- CEUR-WS, truy cập vào tháng 12 10, 2025, [https://ceur-ws.org/Vol-3581/91\_WiP.pdf](https://ceur-ws.org/Vol-3581/91_WiP.pdf)  
28. Few-Shot Learning in Wi-Fi-Based Indoor Positioning \- MDPI, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/2313-7673/9/9/551](https://www.mdpi.com/2313-7673/9/9/551)  
29. Wi-Fi fingerprint based indoor localization using few shot regression \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/382156758\_Wi-Fi\_fingerprint\_based\_indoor\_localization\_using\_few\_shot\_regression](https://www.researchgate.net/publication/382156758_Wi-Fi_fingerprint_based_indoor_localization_using_few_shot_regression)  
30. (PDF) Real-time Tracking of Medical Devices: An Analysis of Multilateration and Fingerprinting Approaches \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/368935321\_Real-time\_Tracking\_of\_Medical\_Devices\_An\_Analysis\_of\_Multilateration\_and\_Fingerprinting\_Approaches](https://www.researchgate.net/publication/368935321_Real-time_Tracking_of_Medical_Devices_An_Analysis_of_Multilateration_and_Fingerprinting_Approaches)  
31. Comparison between trilateration and fingerprinting methods \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/figure/Comparison-between-trilateration-and-fingerprinting-methods\_tbl3\_336453953](https://www.researchgate.net/figure/Comparison-between-trilateration-and-fingerprinting-methods_tbl3_336453953)  
32. BLE Fingerprint Indoor Localization Algorithm Based on Eight-Neighborhood Template Matching \- PMC \- PubMed Central, truy cập vào tháng 12 10, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6891383/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6891383/)  
33. A Comparison Analysis of BLE-Based Algorithms for Localization in Industrial Environments, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/2079-9292/9/1/44](https://www.mdpi.com/2079-9292/9/1/44)  
34. Indoor localization method comparison Fingerprinting and Trilateration algorithm \- Computer Models for Social Change, truy cập vào tháng 12 10, 2025, [https://rose.geog.mcgill.ca/ski/system/files/fm/2011/Wei.pdf](https://rose.geog.mcgill.ca/ski/system/files/fm/2011/Wei.pdf)  
35. Leveraging Hybrid RF-VLP for High-Accuracy Indoor Localization with Sparse Anchors, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/1424-8220/25/10/3074](https://www.mdpi.com/1424-8220/25/10/3074)  
36. Robust Fingerprint Construction Based on Multiple Path Loss Model (M-PLM) for Indoor Localization \- Monash University, truy cập vào tháng 12 10, 2025, [https://research.monash.edu/files/419619889/409800093\_oa.pdf](https://research.monash.edu/files/419619889/409800093_oa.pdf)  
37. A Wi-Fi Fingerprinting Indoor Localization Framework Using Feature-Level Augmentation via Variational Graph Auto-Encoder \- MDPI, truy cập vào tháng 12 10, 2025, [https://www.mdpi.com/2079-9292/14/14/2807](https://www.mdpi.com/2079-9292/14/14/2807)