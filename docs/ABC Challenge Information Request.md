---
noteId: "77d63940d5cf11f080694155c34f83e4"
tags: []

---

# **Strategic Analysis of the Activity and Behavior Computing (ABC) Challenge: Methodologies, Technical Evolution, and Future Directions (2019–2026)**

## **1\. Introduction: The Paradigm Shift in Pervasive Healthcare Computing**

The convergence of ubiquitous sensing, machine learning, and healthcare has precipitated a fundamental shift in how human behavior is quantified and understood. At the epicenter of this transformation lies the Activity and Behavior Computing (ABC) Challenge, an annual competition organized by Sozolab at the Kyushu Institute of Technology and affiliated with the premier UbiComp conference and HASCA workshop. Since its inception, the ABC Challenge has served as a rigorous testbed for "in-the-wild" Human Activity Recognition (HAR), distinguishing itself from other competitions by prioritizing the messiness of real-world data over the sterility of controlled laboratory environments.

As participants prepare for the 8th ABC Challenge in 2026, it is imperative to understand the historical trajectory of this competition. The challenge has evolved from simple locomotion recognition to complex, high-level behavioral forecasting—such as predicting nurse care activities hour-by-hour or forecasting the "wearing-off" phenomenon in Parkinson’s disease patients. This report provides an exhaustive analysis of the winning solutions, key technical methodologies, and seminal literature produced by the ABC community between 2019 and 2025\. By dissecting the contributions of leading researchers like Garcia et al. and Okuda et al., and analyzing the performance metrics of top teams, this document aims to provide a strategic roadmap for future competitors.

The significance of the ABC Challenge extends beyond academic benchmarking. It addresses critical societal issues, particularly the global shortage of caregivers and the aging population. By automating the recognition of nursing activities and the monitoring of chronic diseases, the solutions developed within this framework promise to optimize hospital workflows, reduce documentation burdens, and enhance patient safety.1 The following sections detail the technical odyssey of the ABC Challenge, synthesizing insights from granular accelerometer data, multi-modal sensor fusion, and the recent integration of Generative AI.

## **2\. The ABC Challenge Chronicles: A Longitudinal Analysis (2019–2025)**

The history of the ABC Challenge is characterized by a deliberate escalation in difficulty. Each year, organizers have introduced new layers of complexity—moving from labeled lab data to unlabelled field data, and from classification to future prediction.

### **2.1 The Nurse Care Era (2019–2022)**

The initial phase of the ABC Challenge (2019–2022) focused intensively on the nursing domain. This was driven by the practical need to automate the generation of nursing care records, a task that consumes a significant portion of a caregiver's shift.

#### **2.1.1 ABC 2019: Proving Feasibility in Controlled Environments**

The inaugural Nurse Care Activity Recognition Challenge (2019) set out to answer a fundamental question: Can complex nursing activities be recognized using wearable sensors?

* **Task Overview:** Participants were tasked with recognizing 12 standard nursing activities, including "Vital Signs Measurement," "Blood Collection," "Oral Care," and "Diaper Exchange".3  
* **Data Composition:** The challenge utilized the CARE-COM Nurse Care Activity Dataset. Data was collected in a controlled "recording studio" environment involving 8 subjects. The sensor suite was extensive, including accelerometers, motion capture systems, and indoor location sensors.3  
* **Key Findings:** The results highlighted the limitations of using acceleration data alone for complex activities. While locomotion (walking, running) is easily distinguishable, nursing tasks often involve subtle hand movements or static postures that look identical in accelerometer traces. For instance, distinguishing "Blood Glucose Measurement" from "Blood Collection" proved notoriously difficult due to the kinematic similarity of the tasks.3

#### **2.1.2 ABC 2020: The "Lab to Field" Domain Shift**

The 2020 challenge introduced a critical constraint: domain adaptation.

* **Task Overview:** Participants had to create models that could recognize activities in real-life settings (nursing homes) despite being trained primarily on laboratory data.4  
* **The "Field" Problem:** In real-world nursing, labels are often missing or imprecise because nurses prioritize patient care over data annotation. The training set included both lab and field data, but the test set was exclusively field data from the same users.4  
* **Outcome:** This challenge exposed the fragility of models trained in sterile environments. Algorithms that achieved near-perfect accuracy in the lab saw significant performance degradation when exposed to the noise, interruptions, and multitasking inherent in actual nursing shifts.4

#### **2.1.3 ABC 2021: Big Data and Class Imbalance**

Titled "Can We Do from Big Data?", the 2021 challenge focused on utilizing larger, noisier datasets to improve recognition rates in the face of extreme class imbalance.

* **Winner Analysis:** **Team "Not a Fan of Local Minima"** emerged as the winner, achieving an accuracy of **92%** on the validation set, which significantly outperformed the baseline of 87%. Their final score, calculated as the average of F1-score and Accuracy, was **55.5%**.6  
* **Runner-Up:** **Team Alpha** followed closely with an average score of **54.5%**.6  
* **Technical Insight:** The winning approach demonstrated that traditional machine learning algorithms, specifically **Random Forest**, could outperform deep learning baselines when dealing with tabular feature data derived from sensors. The challenge underscored that "Big Data" in this context often means "Imbalanced Data," where the vast majority of samples represent "Walking" or "Standing," drowning out critical but rare activities like "Emergency Assistance".6

#### **2.1.4 ABC 2022: The Shift to Prediction**

The 2022 challenge marked a pivotal shift from *recognition* (what is happening now?) to *forecasting* (what will happen next?).

* **Task Overview:** Participants were required to predict the future occurrence of nurse care activities hour-by-hour using both care records and accelerometer data.2  
* **Winning Solution:**  
  * **Team:** **"Not a Fan of Local Minima"** (Repeat Winners).  
  * **Methodology:** The team made a strategic decision to prioritize care records over accelerometer data. They recognized that nursing schedules follow a semantic logic (e.g., medication rounds happen at specific times) that is better captured by historical logs than by motion sensors. They utilized a **Random Forest** classifier combined with extensive time-series feature engineering.7  
  * **Score:** The team achieved the highest average score of **55.5%**, topping the baseline F1 score of **32.4%**.7  
* **Strategic Insight:** This challenge proved that in long-horizon forecasting, semantic history is often more predictive than immediate sensor readings. The low baseline score (0.324) highlighted the extreme difficulty of predicting human behavior in unstructured environments.7

### **2.2 The Clinical and Generative Era (2023–2025)**

Recent iterations of the ABC Challenge have diversified beyond general nursing to focus on specific pathologies (Parkinson's Disease), mental health (Depression), and the integration of Generative AI.

#### **2.2.1 ABC 2023: Parkinson’s Disease and Heatstroke Prevention**

This year featured two high-impact tracks focusing on specialized healthcare applications.

* **Track 1: Forecasting Wearing-Off in Parkinson’s Disease:**  
  * **Problem:** Predicting the "wearing-off" phenomenon, where medication effectiveness wanes and symptoms re-emerge, using wrist-worn fitness trackers and smartphone symptom logs.8  
  * **Winning Team:** **Team BAUCVL** secured 1st place with a solution titled "Forecasting Wearing-Off in Parkinson's Disease: An Ensemble Learning Approach Using Wearable Data".9  
  * **Runner-Up:** **Team Convergence** employed a Stacked Super Learner approach.9  
  * **Performance:** The winning solutions achieved remarkable accuracy, with some post-challenge improvements reaching **98.06%** accuracy and an F1-score of **87.65%** by utilizing advanced data augmentation techniques like SMOTE and ADASYN.8  
* **Track 2: Heatstroke Prevention:**  
  * **Winner:** **Team Ahadvisionlab** (University of East London).10  
  * **Methodology:** This track required the integration of physiological data (heart rate, body temperature) to forecast thermal comfort and prevent heatstroke in outdoor workers.10

#### **2.2.2 ABC 2024: Generative AI and Video Skeletons**

Reflecting the global explosion of Generative AI, the 2024 challenge mandated the use of GenAI or Large Language Models (LLMs) in the recognition pipeline.

* **Task:** Recognition of Nurse Training Activity (Endotracheal Suctioning) using Skeleton and Video Data.11  
* **Constraint:** Participants had to use GenAI creatively, such as for synthesizing missing training data or augmenting feature sets.11  
* **Winners:**  
  * **Champion:** **Team Seahawk** with the solution "Recognition of Nurse Activities in Endotracheal Suctioning Procedures: A Comparative Analysis Using **LightGBM** and Other Algorithm".12  
  * **Runner-Up:** **Team Sequoia**, utilizing a **ResNet-3** deep learning model.12  
  * **Best Paper:** **Team bun-bo**, recognized for their "Pose Estimation and Ensemble Learning Approach".12

#### **2.2.3 ABC 2025: Mental Health and Foundation Models**

The 2025 challenge expanded into four distinct tracks, emphasizing mental health and the use of foundation models.

* **Track 1: BeyondSmile (Depression Detection):**  
  * **Task:** Detect depression through facial behaviors and head gestures.  
  * **Winner:** **Team Persistence** (AIUB) with the paper "Facial Behavior-Based Depression Detection with Bi-LSTM".  
  * **Score:** Achieved **0.77 AUROC** (Universal Model) and **0.88 AUROC** (Hybrid Model).13  
* **Track 2: Silent Speech Decoding:**  
  * **Winner:** **Team Persistence** (Qatar University/IIT Roorkee). They applied deep learning techniques to EEG data to decode silent speech.15  
* **Track 3: Virtual Data Generation:**  
  * **Winner:** **Team Willow Swamp** (Osaka University). Their solution focused on interpolation algorithms and data augmentation to generate virtual sensor data for industrial activity recognition.15  
* **Track 4: Parkinson’s Normal vs. Unusual:**  
  * **Winner:** **Team HCMUT\_chillguy** (Ho Chi Minh City University of Technology), utilizing accelerometer-based multi-domain feature extraction.15

## ---

**3\. Key Techniques Used by Top Teams**

A comparative analysis of the winning solutions from 2019 to 2025 reveals a convergence on specific high-performance techniques. While deep learning attracts significant attention, ensemble methods remain the dominant force for tabular sensor data.

### **3.1 Ensemble Learning and Decision Trees**

Winning teams consistently favor ensemble methods for their robustness to noise and ability to handle the non-linear relationships in sensor feature spaces.

* **Random Forest (RF):** This algorithm was the cornerstone of **Team "Not a Fan of Local Minima"** (2021, 2022 winners) and **Team BAUCVL** (2023 winner). RF's ability to handle high-dimensional feature sets without extensive scaling or normalization makes it ideal for the "bag-of-features" approach often used in accelerometer processing.6  
* **Gradient Boosting (LightGBM/XGBoost):** **Team Seahawk** (2024 winner) leveraged **LightGBM** to win the Endotracheal Suctioning challenge. Gradient boosting machines are particularly effective at mining hard-to-classify samples (the "long tail") by iteratively correcting the errors of previous trees. This is crucial for recognizing rare nursing activities.12  
* **Stacked Generalization:** **Team Convergence** (2023) utilized a **Stacked Super Learner**, which combines predictions from multiple base models (e.g., Logistic Regression, SVM, RF) using a meta-learner. This approach often yields marginal but decisive performance gains over any single model.9

### **3.2 Deep Learning Architectures**

For tasks involving complex temporal dependencies or video data, deep learning remains superior.

* **Bi-LSTM (Bidirectional Long Short-Term Memory):** Employed by **Team Persistence** (2025 winner) for depression detection. Bi-LSTMs process data in both forward and backward directions, allowing the model to understand the full context of a facial gesture or physiological signal sequence. This is essential for detecting the subtle, prolonged signs of depression.13  
* **ResNet:** **Team Sequoia** (2024) applied Residual Networks (ResNet) to skeleton data. By treating time-series sensor data as an image (spectrograms) or using 1D convolutions, ResNets can capture hierarchical spatial-temporal patterns that manual feature engineering might miss.12  
* **ST-GCN (Spatio-Temporal Graph Convolutional Networks):** Used by **Team TDU-DSML** (2019), ST-GCNs model the human body as a graph structure (joints as nodes, bones as edges). This topology-aware approach is the state-of-the-art for skeleton-based activity recognition.3

### **3.3 Data Augmentation and Generative AI**

The scarcity of labeled data in healthcare has driven the adoption of advanced augmentation strategies.

* **SMOTE & ADASYN:** In the 2023 Parkinson's challenge, top teams utilized Synthetic Minority Over-sampling Technique (SMOTE) and Adaptive Synthetic Sampling (ADASYN) to generate synthetic samples for the minority class (e.g., "Wearing-Off" episodes), balancing the dataset and preventing classifier bias.8  
* **Generative Adversarial Networks (GANs):** The 2024 and 2025 challenges explicitly encouraged the use of GenAI. Techniques like **CTGAN** (Conditional Tabular GAN) and Variational Autoencoders (VAEs) are now used to synthesize entirely new "virtual" subjects, allowing models to train on datasets orders of magnitude larger than the physical collection.11

## ---

**4\. Seminal Literature from the ABC Challenge: Garcia et al. and Okuda et al.**

Two research groups have been particularly influential in shaping the theoretical and practical frameworks of the ABC Challenge. Their papers provide essential reading for any prospective participant.

### **4.1 Garcia et al.: The Prediction Paradigm**

**Key Paper:** *Summary of the Fourth Nurse Care Activity Recognition Challenge \- Predicting Future Activities*.2

* **Context:** This paper analyzes the 2022 challenge, which pivoted from recognition to prediction.  
* **Methodological Contribution:** Garcia et al. established that predicting *future* activities requires a fundamental shift in feature engineering. Unlike recognition, which relies on *current* motion data, prediction relies on *semantic history*. The paper demonstrates that models using care records alone often outperformed those using accelerometer data, as nursing routines follow a logical script (e.g., "Vital Signs" \-\> "Documentation" \-\> "Medication").7  
* **Metric Innovation:** The paper introduces rigorous evaluation metrics for imbalanced prediction, specifically advocating for the **Average F1 Score** (macro-averaged) to prevent models from "cheating" by only predicting the majority class. It also discusses the "False Head Rate" to quantify the risk of misclassifying rare, critical events as common ones.2

### **4.2 Okuda et al.: Infrastructure and Data Integrity**

**Key Papers:**

1. *A Mobile App for Nursing Activity Recognition (FonLog)*.19  
2. *Activity Prediction Method for Nursing Care Records with Missing Entries*.20  
3. *A Relabeling Approach to Signal Patterns for Beacon-based Indoor Localization in Nursing Care Facility*.21  
* **Technical Contribution:** Ryuichiro Okuda’s work addresses the "dirty data" reality of field deployments.  
  * **FonLog:** Okuda developed the **FonLog** application, the primary tool used for data collection in the ABC challenges. This app integrates activity labeling with sensor logging, enabling the large-scale collection of the HASC and CARE-COM datasets.2  
  * **Missing Entry Prediction:** In 20, Okuda proposed a correction module for nursing care records. Recognizing that nurses often forget to log activities, this model uses binary occurrence time series to *reconstruct* missing entries before they are used for training. This preprocessing step is critical for cleaning the noisy labels found in the challenge datasets.20  
  * **BLE Relabeling:** In 21, Okuda introduced a data augmentation technique for indoor localization. By analyzing Received Signal Strength (RSS) patterns, his method "relabels" data from one room to simulate another geometrically similar room. This technique addresses the data scarcity problem in BLE fingerprinting, yielding a **6-8% improvement in F1 scores**.21

## ---

**5\. Domain-Specific Methodologies**

Success in the ABC Challenge requires mastering three distinct technical domains: BLE Localization, Accelerometer-based HAR, and Multi-Modal Fusion.

### **5.1 Common Approaches for BLE-Based Localization**

Indoor localization is essential for tracking nurse workflows and ensuring patient safety.

* **Fingerprinting:** The standard approach involves creating a radio map of RSSI vectors. During the online phase, the observed vector is matched against the map using k-Nearest Neighbors (kNN) or probabilistic methods.  
* **Relabeling Augmentation:** As proposed by Okuda et al., utilizing signal patterns from data-rich locations to augment data-poor locations is a key strategy for improving robustness.21  
* **Weighted Centroid Localization (WCL):** This geometric method calculates the user's position as the weighted average of the coordinates of detected beacons, where weights are inversely proportional to the signal distance.

### **5.2 Common Approaches for Activity Recognition with Accelerometer Data**

This remains the core task of most ABC challenges.

* **Windowing Strategies:** Data is typically segmented into sliding windows (e.g., 2–5 seconds) with 50% overlap. Adaptive window sizing is an advanced technique used to capture activities of varying durations (e.g., a quick "fall" vs. a long "medication round").  
* **Feature Engineering:**  
  * **Time-Domain:** Mean, variance, skewness, kurtosis, and zero-crossing rate.  
  * **Frequency-Domain:** Fast Fourier Transform (FFT) coefficients and spectral energy are crucial for distinguishing rhythmic activities (walking) from non-rhythmic ones (standing).  
  * **Jerk:** The derivative of acceleration (Jerk) is highly predictive of sudden movements.  
* **Deep Learning:** 1D-CNNs are increasingly used to learn features directly from raw waveforms, avoiding the need for manual feature selection.3

### **5.3 Common Approaches for Multi-Modal Sensor Fusion**

The most competitive teams fuse data from disparate sources to create a holistic view of the subject.

* **Early Fusion:** Concatenating raw feature vectors (e.g., accelerometer \+ gyroscope \+ BLE) before feeding them into a classifier. This is simple but assumes all modalities are synchronized and available.  
* **Late Fusion:** Training separate models for each modality (e.g., a ResNet for video, a Random Forest for sensors) and combining their outputs using weighted voting or a meta-learner. **Team Persistence** successfully used a hybrid model approach in 2025\.14  
* **Context-Aware Fusion:** This advanced technique uses semantic data (care records) to gate or weight the sensor predictions. For example, if the care record indicates "10:00 AM: Medication," the model might suppress the probability of "Sleeping" even if the accelerometer shows low motion.2

## ---

**6\. Performance Benchmarks: Typical Baseline F1 Scores**

Understanding the baseline performance is critical for assessing the efficacy of a proposed solution. The following table synthesizes the baseline and winning scores across key challenge years.

| Challenge Year | Task Domain | Baseline Score | Winning Score | Metric | Key Insight |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **2019** | Nurse Activity Recognition | **43.1%** (RF), **46.5%** (CNN) | **80.2%** | Accuracy | Motion Capture data significantly boosted accuracy over accelerometers alone.3 |
| **2021** | Big Data / Imbalance | **32.4%** | **55.5%** | Avg (F1 \+ Acc) | Handling class imbalance was the primary differentiator; simple accuracy is misleading here.6 |
| **2022** | Future Activity Prediction | **32.4% (0.324)** | **55.5%** | Avg (F1 \+ Acc) | Prediction is inherently harder than recognition. The low baseline reflects the high entropy of human behavior.7 |
| **2023** | Parkinson's Wearing-Off | **70.0% \- 71.7%** (Literature) | **\~93.2% \- 98.0%** | Accuracy | Physiological signals (Heart Rate) provided strong predictors, allowing for very high accuracy compared to nursing tasks.8 |
| **2024** | Endotracheal Suctioning | N/A (Low due to complexity) | High (Comparative) | F1 Score | Generative AI augmentation was key to handling the complex, multi-step nature of suctioning procedures.12 |
| **2025** | Depression Detection | N/A | **0.77 \- 0.88** | AUROC | Hybrid models fusing universal and personalized features achieved the best results.14 |

**Analysis of Baselines:**

* **The Prediction Gap:** The stark contrast between the prediction baseline (0.324) and the Parkinson's baseline (\~0.70) illustrates that defining the problem space is as important as the algorithm. Predicting "what happens next" (2022) is significantly harder than classifying "what is happening now" (2023).  
* **The Generalization Gap:** In 2021, baseline models that performed well in validation often collapsed on the test set (dropping from \~70% to \~13%), highlighting the critical need for domain adaptation techniques that can handle the variability of "in-the-wild" data.23

## ---

**7\. Conclusion and Strategic Recommendations for 2026**

The ABC Challenge has evolved into a premier venue for high-stakes, real-world computing. It rewards participants who can look beyond the raw data to understand the semantic and operational context of the domain.

**Strategic Recommendations for ABC 2026:**

1. **Prioritize Context Over Sensors:** As demonstrated by the 2022 winners, understanding the *business logic* (e.g., nursing workflows, medication schedules) can often yield better predictions than sophisticated sensor processing. Do not ignore the semantic data (care records).  
2. **Master the Ensemble:** While Deep Learning is essential for video/audio, **Random Forest** and **Gradient Boosting (LightGBM)** remain the champions for tabular sensor data. A robust solution will likely involve a stacked ensemble of these methods.  
3. **Address Imbalance Aggressively:** You cannot win an ABC challenge with a standard loss function. Implement **SMOTE**, **ADASYN**, or cost-sensitive learning to ensure your model learns the rare, critical classes.8  
4. **Leverage Generative AI:** The trend is clear. Use GenAI (CTGAN, VAEs) to augment your data, particularly for rare classes or missing modalities. This is no longer optional; it is a requirement for state-of-the-art performance.11  
5. **Study the Infrastructure:** Review the works of **Okuda et al.** to understand the data collection pipeline (FonLog). Understanding *how* the data was collected often reveals biases or artifacts that can be exploited for better performance.

By synthesizing these technical insights with a deep understanding of the Garcia and Okuda frameworks, participants will be well-positioned to tackle the complexities of the 8th ABC Challenge in 2026\.

### **Citations**

1

#### **Nguồn trích dẫn**

1. HASC Corpus: Large Scale Human Activity Corpus for the Real-World Activity Understandings \- Opportunity, truy cập vào tháng 12 10, 2025, [http://www.opportunity-project.eu/system/files/docs/Workshop/Kawaguchi-HASC\_Corpus.pdf](http://www.opportunity-project.eu/system/files/docs/Workshop/Kawaguchi-HASC_Corpus.pdf)  
2. Summary of the Fourth Nurse Care Activity Recognition Challenge \-Predicting Future Activities \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/378781773\_Summary\_of\_the\_Fourth\_Nurse\_Care\_Activity\_Recognition\_Challenge\_-Predicting\_Future\_Activities](https://www.researchgate.net/publication/378781773_Summary_of_the_Fourth_Nurse_Care_Activity_Recognition_Challenge_-Predicting_Future_Activities)  
3. (PDF) Nurse care activity recognition challenge: summary and results \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/335765627\_Nurse\_care\_activity\_recognition\_challenge\_summary\_and\_results](https://www.researchgate.net/publication/335765627_Nurse_care_activity_recognition_challenge_summary_and_results)  
4. About Second Nurse Care Activity Recognition Challenge, truy cập vào tháng 12 10, 2025, [https://abc-research.github.io/nurse2020/learn/](https://abc-research.github.io/nurse2020/learn/)  
5. ABC2024 \- AUTOCARE LLC, truy cập vào tháng 12 10, 2025, [https://autocare.ai/abc2024](https://autocare.ai/abc2024)  
6. Summary of the Third Nurse Care Activity Recognition Challenge \- Can We Do from the Field Data? | Request PDF \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/354824699\_Summary\_of\_the\_Third\_Nurse\_Care\_Activity\_Recognition\_Challenge\_-\_Can\_We\_Do\_from\_the\_Field\_Data](https://www.researchgate.net/publication/354824699_Summary_of_the_Third_Nurse_Care_Activity_Recognition_Challenge_-_Can_We_Do_from_the_Field_Data)  
7. Summary of the Fourth Nurse Care Activity Recognition Challenge \-Predi \- Taylor & Francis eBooks, truy cập vào tháng 12 10, 2025, [https://www.taylorfrancis.com/chapters/edit/10.1201/9781003371540-29/summary-fourth-nurse-care-activity-recognition-challenge-predicting-future-activities-defry-hamdhana-christina-garcia-nazmun-nahid-haru-kaneko-sayeda-shamma-alia-tahera-hossain-sozo-inoue](https://www.taylorfrancis.com/chapters/edit/10.1201/9781003371540-29/summary-fourth-nurse-care-activity-recognition-challenge-predicting-future-activities-defry-hamdhana-christina-garcia-nazmun-nahid-haru-kaneko-sayeda-shamma-alia-tahera-hossain-sozo-inoue)  
8. Predicting Wearing-Off in Parkinson's Disease Patients Using Multimodal Time-Series Data and Ensemble Learning \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/394918666\_Predicting\_Wearing-Off\_in\_Parkinson's\_Disease\_Patients\_Using\_Multimodal\_Time-Series\_Data\_and\_Ensemble\_Learning](https://www.researchgate.net/publication/394918666_Predicting_Wearing-Off_in_Parkinson's_Disease_Patients_Using_Multimodal_Time-Series_Data_and_Ensemble_Learning)  
9. Results \- 5th ABC Challenge, truy cập vào tháng 12 10, 2025, [https://abc-research.github.io/challenge2023/results/](https://abc-research.github.io/challenge2023/results/)  
10. AtiqAhad Vision Lab., truy cập vào tháng 12 10, 2025, [https://ahadvisionlab.com/](https://ahadvisionlab.com/)  
11. 6th ABC Challenge \- GitHub Pages, truy cập vào tháng 12 10, 2025, [https://abc-research.github.io/challenge2024/](https://abc-research.github.io/challenge2024/)  
12. Results \- 6th ABC Challenge \- GitHub Pages, truy cập vào tháng 12 10, 2025, [https://abc-research.github.io/challenge2024/results/](https://abc-research.github.io/challenge2024/results/)  
13. 1st Place and Best Paper Award at the 7th International Conference on Activity and Behavior Computing (ABC) 2025 | AIUB, truy cập vào tháng 12 10, 2025, [https://www.aiub.edu/1st-place-and-best-paper-award-at-the-7th-international-conference-on-activity-and-behavior-computing-abc-2025](https://www.aiub.edu/1st-place-and-best-paper-award-at-the-7th-international-conference-on-activity-and-behavior-computing-abc-2025)  
14. Faculty Profiles \- Christina Alvarez Garcia, truy cập vào tháng 12 10, 2025, [https://hyokadb02.jimu.kyutech.ac.jp/html/100001806\_en.html](https://hyokadb02.jimu.kyutech.ac.jp/html/100001806_en.html)  
15. ABC2025 \- AUTOCARE LLC, truy cập vào tháng 12 10, 2025, [https://autocare.ai/abc2025](https://autocare.ai/abc2025)  
16. International Journal of Activity and Behavior Computing \- J-Stage, truy cập vào tháng 12 10, 2025, [https://www.jstage.jst.go.jp/browse/ijabc/2025/2/\_contents/-char/en](https://www.jstage.jst.go.jp/browse/ijabc/2025/2/_contents/-char/en)  
17. Faculty Profiles \- INOUE Sozo \- 九州工業大学, truy cập vào tháng 12 10, 2025, [https://hyokadb02.jimu.kyutech.ac.jp/html/140\_en.html](https://hyokadb02.jimu.kyutech.ac.jp/html/140_en.html)  
18. Nurse Care Activity Recognition: A Cost-Sensitive Ensemble Approach to Handle Imbalanced Class Problem in the Wild | Request PDF \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/354823803\_Nurse\_Care\_Activity\_Recognition\_A\_Cost-Sensitive\_Ensemble\_Approach\_to\_Handle\_Imbalanced\_Class\_Problem\_in\_the\_Wild](https://www.researchgate.net/publication/354823803_Nurse_Care_Activity_Recognition_A_Cost-Sensitive_Ensemble_Approach_to_Handle_Imbalanced_Class_Problem_in_the_Wild)  
19. A Mobile App for Nursing Activity Recognition | Request PDF, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/328679475\_A\_Mobile\_App\_for\_Nursing\_Activity\_Recognition](https://www.researchgate.net/publication/328679475_A_Mobile_App_for_Nursing_Activity_Recognition)  
20. Activity Prediction Method for Nursing Care Records with Missing Entries \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/391229263\_Activity\_Prediction\_Method\_for\_Nursing\_Care\_Records\_with\_Missing\_Entries](https://www.researchgate.net/publication/391229263_Activity_Prediction_Method_for_Nursing_Care_Records_with_Missing_Entries)  
21. Relabeling for Indoor Localization Using Stationary Beacons in Nursing Care Facilities, truy cập vào tháng 12 10, 2025, [https://www.preprints.org/manuscript/202312.0998](https://www.preprints.org/manuscript/202312.0998)  
22. Haru Kaneko, truy cập vào tháng 12 10, 2025, [https://haruu11113.github.io/](https://haruu11113.github.io/)  
23. Predicting User-specific Future Activities using LSTM-based Multi-label Classification \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/html/2211.03100](https://arxiv.org/html/2211.03100)