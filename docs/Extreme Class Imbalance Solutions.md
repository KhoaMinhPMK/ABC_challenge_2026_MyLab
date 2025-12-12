---
noteId: "c36c3a60d5cc11f080694155c34f83e4"
tags: []

---

# **Dynamics of Extreme Class Imbalance: A Theoretical and Practical Compendium on High-Gini Multi-Class Classification**

## **1\. Introduction: The Anatomy of Extreme Imbalance**

The challenge of training deep neural networks on datasets exhibiting extreme class imbalance represents one of the most persistent hurdles in modern machine learning. The specific problem space defined by a 44-class ontology, a Gini coefficient of 0.783, and a maximal imbalance ratio of 2301:1 transcends the difficulty of standard academic benchmarks. While the machine learning community has extensively studied long-tailed recognition, the majority of existing literature focuses on imbalance ratios ranging from 100:1 to 1000:1, such as those found in CIFAR-100-LT or ImageNet-LT. An imbalance ratio of 2301:1 introduces a magnitude of sparsity where the tail classes are not merely underrepresented but are statistically negligible during standard gradient descent optimization. This scenario necessitates a rigorous re-examination of loss landscapes, optimization dynamics, and the fundamental statistical assumptions governing Empirical Risk Minimization (ERM).

### **1.1 The Statistical Implications of Gini 0.783**

The Gini coefficient, historically utilized in economics to measure income inequality, serves as a potent metric for characterizing the skewness of a label distribution. A Gini coefficient of zero indicates perfect equality (all classes have equal sample counts), while a coefficient of one indicates maximal inequality. In the context of visual recognition or tabular classification, a Gini coefficient of 0.783 1 signifies a distribution that is profoundly skewed, likely following a steep Power Law or Zipfian distribution.

In a dataset with 44 classes, a Gini of 0.783 suggests that the "head" of the distribution—the most frequent classes—likely accounts for the overwhelming majority of the total training samples. Conversely, the "tail"—the least frequent classes—possesses a sample density that approaches the regime of few-shot or even one-shot learning. The implications for training are severe. In standard Stochastic Gradient Descent (SGD), the total gradient is a linear superposition of the gradients computed from individual samples. When the head classes dominate the sample count by three orders of magnitude (2301x), the magnitude of the gradient updates is almost entirely determined by the head classes. The optimization trajectory moves exclusively to minimize the error on the majority, effectively ignoring the minority classes. This phenomenon, often described as the "Seesaw Dilemma" 2, results in a model that achieves high global accuracy by sacrificing the tail, yielding a trivial solution where the minority classes are universally misclassified as the majority.

### **1.2 The Metric Divergence: Macro F1 vs. Accuracy**

The specific requirement to optimize for the Macro F1 metric fundamentally alters the optimization objective compared to standard accuracy. Accuracy (or Micro F1 in multi-class settings) is a frequency-weighted metric; a model can achieve 99% accuracy on a dataset with a 2301:1 imbalance by simply predicting the majority class for every input. However, Macro F1 treats every class as an equal entity, regardless of its support size.3 The contribution of the class with 1 sample to the final Macro F1 score is identical to the contribution of the class with 2301 samples.

This dichotomy creates a direct conflict in the loss landscape. Cross-Entropy loss, which approximates the negative log-likelihood, is naturally aligned with accuracy maximization. To optimize Macro F1, the model must essentially "hallucinate" importance for the tail classes, assigning them a gradient weight that is disproportionate to their statistical presence. This artificial inflation of importance increases the variance of the gradient estimates for the tail classes, leading to instability. If a tail class has only one sample, and its weight is boosted by a factor of 2301 to match the head, the model is prone to memorizing that single sample (overfitting) rather than learning a generalized representation. Consequently, the pursuit of a high Macro F1 score on high-Gini data is a balancing act between bias (ignoring the tail) and variance (overfitting the tail).

This report synthesizes methodologies ranging from foundational loss engineering to state-of-the-art parameter decomposition techniques from the 2023–2025 literature. It provides a comprehensive roadmap for navigating this extreme imbalance, detailing the mathematical derivations, implementation nuances, and strategic deployments of Focal Loss, Class-Balanced Loss, LDAM-DRW, and cutting-edge approaches like PaCo, LGLA, and MORE.

## ---

**2\. Foundational Cost-Sensitive Learning Frameworks**

The most immediate intervention for addressing class imbalance is cost-sensitive learning, which modifies the loss function to penalize errors on minority classes more heavily than errors on majority classes. This approach aims to flatten the effective loss landscape, ensuring that the gradient contributions from all classes are roughly equivalent despite the disparity in sample counts.

### **2.1 Focal Loss: Mitigating Sample Hardness**

Focal Loss, originally proposed for dense object detection in the context of the RetinaNet architecture, introduced a paradigm shift from class-level re-weighting to sample-level re-weighting. In scenarios of extreme imbalance, the majority of training examples are "easy negatives"—samples that the model can correctly classify with high confidence early in the training process. For a dataset with a 2301x imbalance, the cumulative loss of these thousands of easy examples can overwhelm the rare, informative signals from the hard, misclassified tail samples.

The standard Cross-Entropy (CE) loss for a sample with ground truth class $y$ and predicted probability $p\_t$ is defined as $CE(p\_t) \= \-\\log(p\_t)$. Focal Loss introduces a modulating factor $(1 \- p\_t)^\\gamma$ to this equation:

$$FL(p\_t) \= \-\\alpha\_t (1 \- p\_t)^\\gamma \\log(p\_t)$$  
Here, $\\gamma \\geq 0$ is the focusing parameter, and $\\alpha\_t$ is a balancing variant. The mechanism of the modulating factor is crucial. As the probability $p\_t$ of the correct class approaches 1 (indicating an easy, well-classified example), the factor $(1 \- p\_t)^\\gamma$ approaches 0\. This effectively down-weights the loss contribution of easy examples.5

In the context of the user's problem, where the head class samples are abundant, the model will rapidly learn to classify them with high confidence ($p\_t \\approx 0.99$). In standard CE, the loss $-\\log(0.99) \\approx 0.01$ is small, but when summed over 2300 samples, it generates a significant gradient. In Focal Loss with $\\gamma=2$, the weight becomes $(1 \- 0.99)^2 \= 0.0001$, suppressing the signal by four orders of magnitude. This allows the optimizer to focus on the tail classes, which likely have lower $p\_t$.

Research indicates that for high imbalance, a $\\gamma$ value of 2.0 is a robust baseline, though higher values (e.g., $\\gamma=3$ or $\\gamma=4$) may be necessary given the extreme 2301x ratio. The parameter $\\alpha\_t$ is typically set to the inverse class frequency, but as we will explore in the next section, simple inverse frequency is often suboptimal for datasets with complex internal structures.

### **2.2 Class-Balanced (CB) Loss: The Theory of Effective Samples**

One of the limitations of standard inverse-frequency weighting is the assumption that each data point carries independent information. In real-world visual data, this assumption rarely holds. As the sample size of a class increases, the marginal information gain of an additional sample diminishes due to redundancy. A class with 2300 images likely contains many near-duplicates or samples with very similar feature representations. The **Class-Balanced Loss** 7 addresses this by introducing the concept of the "Effective Number of Samples."

The theoretical framework posits that each class represents a data manifold with a certain volume $N$. As we sample $n$ instances, we cover a portion of this volume. The effective number of samples, denoted as $E\_n$, is the expected volume covered by $n$ samples. Through a derivation involving random covering problems, the authors propose the formula:

$$E\_n \= \\frac{1 \- \\beta^n}{1 \- \\beta}$$  
Here, $\\beta \\in, the effective number of samples closely tracks the actual count for small $n$ (tail classes) but starts to saturate for very large $n$ (head classes). This prevents the weights for the tail classes from exploding to unstable levels (e.g., 2300\) while still providing significant boost.

Combining this with Focal Loss leads to the **CB-Focal Loss**, a potent baseline for the user's problem:

$$\\text{CB-Focal}(p\_t, y) \= \\frac{1 \- \\beta}{1 \- \\beta^{n\_y}} \\left( \- (1 \- p\_t)^\\gamma \\log(p\_t) \\right)$$  
This formulation simultaneously handles the inter-class imbalance (via the CB term) and the intra-class hard-sample mining (via the Focal term).

## ---

**3\. Margin-Aware Regularization Techniques**

While re-weighting modifies the gradient magnitude, it does not explicitly alter the geometric structure of the learned feature space. In standard softmax training, the decision boundary between two classes is determined by the equality of their logits. For imbalanced data, the tail classes often have much smaller norms in the feature space, causing the decision boundary to collapse towards the tail, reducing the volume of the feature space allocated to the minority. Margin-aware methods aim to explicitly enforce a larger safety margin for tail classes.

### **3.1 LDAM: Label-Distribution-Aware Margin**

The **Label-Distribution-Aware Margin (LDAM)** loss 5 is derived from the generalization error bounds of margin-based classifiers. The central theoretical insight is that the generalization gap for a class is proportional to $1/\\sqrt{n\_y}$. Therefore, to ensure equal generalization performance across all classes, the margin for class $y$, denoted as $\\Delta\_y$, should be inversely proportional to the fourth root of its sample count $n\_y$:

$$\\Delta\_y \= \\frac{C}{n\_y^{1/4}}$$  
The LDAM loss modifies the standard softmax cross-entropy by subtracting this margin from the logit of the ground truth class:

$$\\mathcal{L}\_{LDAM}((x, y); f) \= \-\\log \\frac{e^{s(z\_y \- \\Delta\_y)}}{e^{s(z\_y \- \\Delta\_y)} \+ \\sum\_{j \\neq y} e^{s z\_j}}$$  
Here, $s$ is a scaling factor (inverse temperature) that controls the sharpness of the distribution. By subtracting $\\Delta\_y$ from the true class logit $z\_y$, the loss function requires the model to produce a logit $z\_y$ that is significantly larger than the logits of other classes $z\_j$ to achieve a low loss. Since $\\Delta\_y$ is larger for tail classes, the model is forced to learn a much stronger, more robust feature representation for the minority classes to satisfy this margin requirement.

Comparison with Focal Loss:  
LDAM differs fundamentally from Focal Loss. Focal Loss focuses on hard samples regardless of class, whereas LDAM enforces a structural constraint based on class frequency. Empirical studies 12 have shown that LDAM often produces more separable feature spaces for long-tailed data compared to pure re-weighting.

### **3.2 DRW: The Necessity of Deferred Re-Weighting**

A critical component of the LDAM methodology is the training schedule known as Deferred Re-Weighting (DRW).10  
Deep neural networks learn hierarchical features. The early layers capture generic patterns (edges, textures), while deeper layers capture semantic concepts. If strong re-weighting (like CB-Loss) or large margins (LDAM) are applied from the very first epoch, the model's feature extractor can become corrupted. The noise from the up-weighted tail samples (which might be outliers) prevents the model from learning a robust generic feature extractor.  
DRW proposes a two-stage training process:

1. **Stage 1 (Feature Learning):** Train the model using standard ERM (Cross-Entropy) or vanilla LDAM without re-weighting. This allows the model to leverage the abundant head class data to learn high-quality, generalizable features.  
2. **Stage 2 (Margin/Weight Adaptation):** After the learning rate decays (e.g., at epoch 80 of 100), apply the re-weighting scheme (or the class-balanced weights). At this stage, the feature extractor is fixed or stable, and the optimization focuses on adjusting the decision boundaries (the classifier layer) to balance the class performance.

For the user's 2301x imbalance, DRW is not optional; it is mandatory. Attempting to apply a weight of \~2300 to tail samples at epoch 1 will almost certainly lead to gradient explosion or convergence to a degenerate solution.

## ---

**4\. The Logit Adjustment Revolution (2021-2024)**

The most significant theoretical advancement in long-tailed recognition in recent years is the move towards **Logit Adjustment (LA)**. This family of methods stems from a statistical perspective on minimizing balanced error.

### **4.1 Statistical Grounding and Fisher Consistency**

Menon et al. 14 formalized the problem by proving that standard softmax cross-entropy is Fisher consistent for the accuracy metric (minimizing misclassification rate). However, to minimize the *balanced error* (which corresponds to maximizing Macro Recall or Macro F1), the optimal decision rule must account for the class priors.

The theoretical optimal decision boundary for balanced error requires adjusting the logits $f(x)$ by the class priors $\\pi\_y$:

$$f^\*(x) \= f(x) \- \\tau \\log \\pi\_y$$

where $\\pi\_y \= n\_y / N$.  
This leads to two practical algorithms:

1. **Post-Hoc Logit Adjustment:** Train a standard model with Cross-Entropy. During inference, subtract $\\tau \\log \\pi\_y$ from the predicted logits. This penalizes head classes (where $\\pi\_y$ is large) and boosts tail classes.  
2. Logit Adjusted Loss: Incorporate the adjustment into the training loss:

   $$\\mathcal{L}\_{LA} \= \-\\log \\frac{e^{f\_y(x) \+ \\tau \\log \\pi\_y}}{\\sum\_j e^{f\_j(x) \+ \\tau \\log \\pi\_j}}$$

   Note the sign change (+ vs \-) depending on whether it's applied to the logit target or the interference terms. The core idea is to force the model to output logits that naturally compensate for the prior frequency.

### **4.2 LGLA: Local and Global Logit Adjustments (2023)**

While standard LA applies a global adjustment based on dataset-wide frequencies, **Local and Global Logit Adjustments (LGLA)** 16 argues that this is insufficient for complex datasets. LGLA employs an ensemble-of-experts approach.

* **Global Expert:** Handles the overall long-tailed distribution using standard LA.  
* **Local Experts:** The classes are divided into subsets (e.g., Many, Medium, Few). "Local" experts are trained to distinguish classes *within* these subsets.  
* **Mechanism:** This divide-and-conquer strategy reduces the effective imbalance ratio seen by the local experts. An expert trained only on "Few" classes sees a much more balanced distribution (e.g., ratio 10:1 instead of 2301:1).  
* Adaptive Angular Weighted (AAW) Loss: LGLA combines this with a specialized loss that mines hard samples, further refining the decision boundaries.  
  For the user's 44-class problem, LGLA suggests that training a monolithic model might be inferior to training a small ensemble where one head focuses on distinguishing the tail classes from each other, unburdened by the head classes.

### **4.3 Generalized Logit Adjustment (GLA)**

A limitation of standard LA is the assumption that the test distribution is uniform (balanced). In reality, the test distribution might be unknown or shift. **Generalized Logit Adjustment (GLA)** 17 introduces a mechanism to estimate the optimal adjustment factor $\\tau$ using a validation set or through a meta-learning process. This is particularly relevant when using foundation models (like CLIP) zero-shot, where the "training prior" is implicit and unknown.

### **4.4 Gaussian Clouded Logit (GCL)**

**Gaussian Clouded Logit (GCL)** 19 (2024) attacks the problem from a geometric perspective. It addresses the "feature collapse" of tail classes—where tail samples cluster too tightly in the embedding space, lacking the variance needed for robust generalization. GCL injects Gaussian noise into the logits of the tail classes during training, effectively "clouding" or expanding their footprint in the decision space. This forces the classifier to push the decision boundaries further away from the tail centers, reserving more volume for them. This technique is computationally cheap and can be layered on top of LDAM or standard LA.

## ---

**5\. Advanced Model-Based Approaches: SOTA 2024-2025**

Moving beyond loss functions and logit manipulation, the most recent state-of-the-art (SOTA) research focuses on the model architecture and parameter space itself.

### **5.1 PaCo: Parametric Contrastive Learning**

Contrastive Learning (CL) has revolutionized representation learning, but standard methods like SimCLR or MoCo fail in long-tailed settings. In CL, positive pairs are formed by augmenting the same image. For tail classes, the lack of *other* distinct images in a batch makes it impossible to learn intra-class variance.

Parametric Contrastive Learning (PaCo) 20 introduces a set of learnable class centers ($C \= \\{c\_1, \\dots, c\_K\\}$) into the contrastive loss.

$$\\mathcal{L}\_{PaCo} \= \-\\log \\frac{\\exp(z \\cdot c\_y / \\tau)}{\\exp(z \\cdot c\_y / \\tau) \+ \\sum\_{j \\neq y} \\exp(z \\cdot c\_j / \\tau) \+ \\sum\_{k \\in \\text{neg}} \\exp(z \\cdot k / \\tau)}$$

Instead of just contrasting a sample $z$ against other batch samples, PaCo contrasts $z$ against the learnable center $c\_y$. This ensures that even if a tail class appears only once in an epoch, it is constantly being pushed towards its stable center $c\_y$ and pulled away from head class centers. This "parametric" support acts as a persistent memory of the class concept, preventing the tail representations from collapsing. PaCo currently holds SOTA results on iNaturalist 2018 (a high-Gini dataset) and is highly recommended for the user's problem.

### **5.2 MORE: Model Rebalancing with Sinusoidal Scheduling**

**MORE (Model Rebalancing)** 22 (Late 2024\) introduces a novel hypothesis: the parameter space itself is imbalanced. The weights of the neural network are overwhelmingly utilized to encode features of the majority classes.

MORE proposes explicitly decomposing the model weights $\\theta$ into two components:

1. $\\theta\_g$: Generic parameters, shared and dominated by head classes.  
2. $\\theta\_t$: Tail-specific parameters, a low-rank component designed to capture minority features.

The training process is governed by a **Sinusoidal Reweighting Schedule**.

* **Early Training:** The loss weight $\\alpha(t)$ for the rebalancing term is low. The model focuses on $\\theta\_g$, learning high-quality generic features (shapes, edges) from the abundant data.  
* **Late Training:** The schedule $\\alpha(t)$ ramps up sinusoidally. The optimization shifts focus to $\\theta\_t$, fine-tuning the specific parameters needed to discriminate tail classes without degrading the generic features.

The reweighting factor follows the equation:

$$\\alpha(t) \= \\frac{1}{2} \\left( 1 \- \\cos\\left( \\frac{t \\pi}{T\_{max}} \\right) \\right)$$

This smooth transition is crucial. Unlike the step-function of DRW, the sinusoidal schedule avoids the shock of sudden objective changes, leading to smoother convergence and higher Macro F1 scores.

## ---

**6\. PyTorch Implementation Details and Engineering Guidelines**

Implementing these complex loss functions requires careful attention to numerical stability and tensor operations. Below are robust PyTorch implementations for the key techniques discussed.

### **6.1 Robust LDAM Loss with Margin Calculation**

This implementation handles the dynamic margin generation based on the user's specific class counts.

Python

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np

class LDAMLoss(nn.Module):  
    def \_\_init\_\_(self, cls\_num\_list, max\_m=0.5, weight=None, s=30):  
        """  
        LDAM Loss with class-dependent margins.  
        Args:  
            cls\_num\_list (list): List of sample counts per class \[N1, N2,... N44\]  
            max\_m (float): The maximum margin value (hyperparameter).  
            weight (torch.Tensor): Optional class weights for re-weighting.  
            s (float): Scaling factor (inverse temperature).  
        """  
        super(LDAMLoss, self).\_\_init\_\_()  
          
        \# Calculate margins: Delta\_y \~ 1 / n\_y^(1/4)  
        m\_list \= 1.0 / np.sqrt(np.sqrt(cls\_num\_list))  
        \# Normalize margins so the largest margin is max\_m  
        m\_list \= m\_list \* (max\_m / np.max(m\_list))  
          
        \# Register as buffer to move with model to GPU  
        self.register\_buffer('m\_list', torch.FloatTensor(m\_list))  
        self.s \= s  
        self.weight \= weight

    def forward(self, x, target):  
        \# x: logits \[batch\_size, num\_classes\]  
        \# target: labels \[batch\_size\]  
          
        index \= torch.zeros\_like(x, dtype=torch.uint8)  
        \# Create one-hot mask for ground truth  
        index.scatter\_(1, target.data.view(-1, 1), 1)  
          
        index\_float \= index.type(torch.cuda.FloatTensor)  
          
        \# Select margins for the current batch samples  
        \# batch\_m shape: \[batch\_size, 1\]  
        batch\_m \= torch.matmul(self.m\_list\[None, :\], index\_float.transpose(0, 1))  
        batch\_m \= batch\_m.view((-1, 1))  
          
        \# Apply margin only to the ground truth logit: z\_y \- Delta\_y  
        x\_m \= x \- batch\_m  
          
        \# Combine: use margin-adjusted logit for GT, original for others  
        output \= torch.where(index.bool(), x\_m, x)  
          
        \# Scale logits and compute cross entropy  
        \# Note: self.weight can be passed here for Deferred Re-Weighting  
        return F.cross\_entropy(self.s \* output, target, weight=self.weight)

### **6.2 Class-Balanced Focal Loss (CB-Focal)**

This snippet integrates the "Effective Number of Samples" calculation directly into the Focal Loss alpha parameter.

Python

class CBFocalLoss(nn.Module):  
    def \_\_init\_\_(self, cls\_num\_list, beta=0.9999, gamma=2.0):  
        """  
        Class-Balanced Focal Loss.  
        Args:  
            cls\_num\_list: List of sample counts.  
            beta: Hyperparameter for effective number (0.9999 recommended for high imbalance).  
            gamma: Focusing parameter (2.0 default, try 3.0 or 4.0 for 2301x ratio).  
        """  
        super(CBFocalLoss, self).\_\_init\_\_()  
          
        \# 1\. Calculate Effective Number of Samples: E\_n \= (1 \- beta^n) / (1 \- beta)  
        effective\_num \= 1.0 \- np.power(beta, cls\_num\_list)  
        per\_cls\_weights \= (1.0 \- beta) / np.array(effective\_num)  
          
        \# Normalize weights to sum to num\_classes (keeps loss magnitude stable)  
        per\_cls\_weights \= per\_cls\_weights / np.sum(per\_cls\_weights) \* len(cls\_num\_list)  
          
        self.register\_buffer('alpha', torch.tensor(per\_cls\_weights).float())  
        self.gamma \= gamma

    def forward(self, logits, targets):  
        \# Calculate standard Cross Entropy (unreduced)  
        ce\_loss \= F.cross\_entropy(logits, targets, reduction='none')  
          
        \# Calculate probabilities of the correct class (p\_t)  
        pt \= torch.exp(-ce\_loss)  
          
        \# Gather class weights for the batch  
        alpha\_t \= self.alpha\[targets\]  
          
        \# Focal formulation: \-alpha \* (1 \- pt)^gamma \* log(pt)  
        focal\_loss \= alpha\_t \* (1 \- pt) \*\* self.gamma \* ce\_loss  
          
        return focal\_loss.mean()

### **6.3 Post-Hoc Logit Adjustment for Inference**

This function should be applied during the validation and testing phases to maximize Macro F1.

Python

def predict\_with\_logit\_adjustment(model, images, cls\_num\_list, tau=1.0):  
    """  
    Applies Post-Hoc Logit Adjustment to maximize Balanced Accuracy / Macro F1.  
    """  
    \# Calculate prior probabilities  
    prior \= np.array(cls\_num\_list) / np.sum(cls\_num\_list)  
    log\_prior \= torch.tensor(np.log(prior \+ 1e-8)).float().cuda()  
      
    model.eval()  
    with torch.no\_grad():  
        logits \= model(images)  
          
        \# Adjustment formula: f(x) \- tau \* log(pi)  
        \# Subtracting log(prior) boosts rare classes (since log(small) is negative large)  
        \# Note: Ensure the sign matches the logic. Here we want to INCREASE logits for rare classes.  
        \# Rare class: prior=0.0001 \-\> log(prior)=-9.2.   
        \# logit \- (1.0 \* \-9.2) \= logit \+ 9.2. This boosts the rare class.  
        adjusted\_logits \= logits \- tau \* log\_prior   
          
        predictions \= torch.argmax(adjusted\_logits, dim=1)  
          
    return predictions

## ---

**7\. Optimizing Macro F1 Directly**

The user's core metric is Macro F1. Standard losses optimize accuracy. While "Soft F1 Loss" exists 24, it is mathematically unstable as a primary training objective due to vanishing gradients when batches do not contain samples for all classes—a guarantee with 44 classes and 2301x imbalance.

### **7.1 The Soft F1 Fallacy**

Optimizing Soft F1 directly often leads to a phenomenon where the model predicts the majority class with probability 1.0 and others with 0.0 to stabilize the denominator of the F1 formula ($2TP / (2TP \+ FP \+ FN)$).26

### **7.2 The Threshold Tuning Strategy**

A more robust approach is to train with LDAM-DRW or PaCo (which creates a good feature space) and then optimize the decision thresholds post-hoc on the validation set to maximize Macro F1.  
Instead of argmax, we define a vector of thresholds $T \= \[t\_1, \\dots, t\_{44}\]$.

$$\\hat{y} \= \\text{argmax}\_c (p\_c / t\_c)$$

This is mathematically equivalent to Logit Adjustment but allows for fine-grained tuning specifically for F1 rather than Balanced Accuracy.

## ---

**8\. Strategic Roadmap for the User**

Given the extreme constraints (Ratio 2301x, Gini 0.783), a standard "train and hope" approach will fail. The following roadmap integrates the theoretical insights into a practical pipeline.

### **Phase 1: Data & Architecture Setup**

* **Backbone:** Use a **ResNet-50** or **EfficientNet-B0** pre-trained on ImageNet. Do not train from scratch; the head classes in ImageNet will provide the initial feature filters that the tail classes lack samples to learn.  
* **Classifier Head:** Replace the standard dot-product classifier (Linear layer) with a **Cosine Classifier**.  
  * Standard: $y \= w \\cdot x \+ b$. The magnitude $\\|x\\|$ and $\\|w\\|$ affect the logit. Head classes learn larger norm weights $\\|w\\|$, biasing the output.  
  * Cosine: $y \= \\frac{w \\cdot x}{\\|w\\| \\|x\\|}$. This normalizes the magnitudes, forcing the model to rely on angular alignment (semantic similarity) rather than frequency-based magnitude.

### **Phase 2: Training Strategy (Decoupled)**

Use the **Deferred Re-Weighting (DRW)** schedule.

* **Epochs 0-150:** Train with standard Cross-Entropy or vanilla LDAM. Do not use re-weighting. Let the model learn the visual features from the head classes.  
* **Epochs 151-200:**  
  * Freeze the backbone (optional, but recommended for extreme imbalance).  
  * Switch loss to **LDAM** with **Class-Balanced Weights** ($\\beta=0.9999$).  
  * This "fine-tunes" the decision boundaries without destroying the feature extractor.

### **Phase 3: SOTA Integration**

If Phase 2 yields insufficient Macro F1:

* Implement **PaCo**.20 The parametric centers act as "anchors" for the 1-sample classes, providing a stable gradient even when positive pairs are missing from the batch.  
* Implement **MORE** 22 with the sinusoidal schedule. This is particularly effective if the model seems to underfit the tail (low recall) while overfitting the head (high precision).

### **Phase 4: Inference Optimization**

Do not use raw softmax outputs.

* Apply **Logit Adjustment** with $\\tau$ tuned on the validation set.  
* Sweep $\\tau$ from 0.5 to 2.0 and select the value that maximizes **Macro F1**.

## **9\. Conclusion**

The problem of 2301x imbalance with a Gini of 0.783 is not merely a data engineering problem; it is a geometric and statistical one. Standard losses fail because they are statistically consistent with accuracy, not Macro F1. Over-sampling fails because it induces variance (overfitting) on the tail. The solution lies in **geometric regularization** (LDAM/PaCo) to reserve feature space volume for the tail, combined with **probabilistic correction** (Logit Adjustment) to align the decision boundaries with the target metric. By adopting a decoupled training strategy with margin-aware losses and post-hoc adjustment, it is possible to achieve non-trivial Macro F1 scores even in the face of such extreme sparsity.

#### **Nguồn trích dẫn**

1. Gini coefficient \- Wikipedia, truy cập vào tháng 12 10, 2025, [https://en.wikipedia.org/wiki/Gini\_coefficient](https://en.wikipedia.org/wiki/Gini_coefficient)  
2. Long-Tailed Learning as Multi-Objective Optimization, truy cập vào tháng 12 10, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/28103/28211](https://ojs.aaai.org/index.php/AAAI/article/view/28103/28211)  
3. Optimal Thresholding of Classifiers to Maximize F1 Measure \- PMC \- NIH, truy cập vào tháng 12 10, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/)  
4. Macro average vs Weighted average \- GeeksforGeeks, truy cập vào tháng 12 10, 2025, [https://www.geeksforgeeks.org/machine-learning/macro-average-vs-weighted-average/](https://www.geeksforgeeks.org/machine-learning/macro-average-vs-weighted-average/)  
5. Learning Only When It Matters: Cost-Aware Long-Tailed Classification \- AAAI Publications, truy cập vào tháng 12 10, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/29133/30144](https://ojs.aaai.org/index.php/AAAI/article/view/29133/30144)  
6. Feature-Balanced Loss for Long-Tailed Visual Recognition \- COMP, HKBU, truy cập vào tháng 12 10, 2025, [https://www.comp.hkbu.edu.hk/\~ymc/papers/conference/ICME22-ID903.pdf](https://www.comp.hkbu.edu.hk/~ymc/papers/conference/ICME22-ID903.pdf)  
7. truy cập vào tháng 12 10, 2025, [https://openaccess.thecvf.com/content\_CVPR\_2019/papers/Cui\_Class-Balanced\_Loss\_Based\_on\_Effective\_Number\_of\_Samples\_CVPR\_2019\_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)  
8. Class-Balanced Loss Based on Effective Number of Samples \- ResearchGate, truy cập vào tháng 12 10, 2025, [https://www.researchgate.net/publication/330466215\_Class-Balanced\_Loss\_Based\_on\_Effective\_Number\_of\_Samples](https://www.researchgate.net/publication/330466215_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples)  
9. Enlarged Large Margin Loss for Imbalanced Classification \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/pdf/2306.09132](https://arxiv.org/pdf/2306.09132)  
10. Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss, truy cập vào tháng 12 10, 2025, [http://papers.neurips.cc/paper/8435-learning-imbalanced-datasets-with-label-distribution-aware-margin-loss.pdf](http://papers.neurips.cc/paper/8435-learning-imbalanced-datasets-with-label-distribution-aware-margin-loss.pdf)  
11. Learning Imbalanced Datasets with Label-Distribution-Aware ..., truy cập vào tháng 12 10, 2025, [https://arxiv.org/pdf/1906.07413](https://arxiv.org/pdf/1906.07413)  
12. Reviews: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss \- NIPS papers, truy cập vào tháng 12 10, 2025, [https://papers.nips.cc/paper\_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Reviews.html](https://papers.nips.cc/paper_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Reviews.html)  
13. \[Quick Review\] Long-tail learning via logit adjustment \- Liner, truy cập vào tháng 12 10, 2025, [https://liner.com/review/longtail-learning-via-logit-adjustment](https://liner.com/review/longtail-learning-via-logit-adjustment)  
14. Long-tail learning via logit adjustment \- SciSpace, truy cập vào tháng 12 10, 2025, [https://scispace.com/pdf/long-tail-learning-via-logit-adjustment-qmlstkat0q.pdf](https://scispace.com/pdf/long-tail-learning-via-logit-adjustment-qmlstkat0q.pdf)  
15. Local and Global Logit Adjustments for Long-Tailed Learning \- CVF Open Access, truy cập vào tháng 12 10, 2025, [https://openaccess.thecvf.com/content/ICCV2023/papers/Tao\_Local\_and\_Global\_Logit\_Adjustments\_for\_Long-Tailed\_Learning\_ICCV\_2023\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Tao_Local_and_Global_Logit_Adjustments_for_Long-Tailed_Learning_ICCV_2023_paper.pdf)  
16. Class and Attribute-Aware Logit Adjustment for Generalized Long-Tail Learning, truy cập vào tháng 12 10, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/34462/36617](https://ojs.aaai.org/index.php/AAAI/article/view/34462/36617)  
17. Generalized Logit Adjustment: Calibrating Fine-tuned Models by Removing Label Bias in Foundation Models | OpenReview, truy cập vào tháng 12 10, 2025, [https://openreview.net/forum?id=9qG6cMGUWk](https://openreview.net/forum?id=9qG6cMGUWk)  
18. Adjusting Logit in Gaussian Form for Long-Tailed Visual Recognition \- IEEE Xplore, truy cập vào tháng 12 10, 2025, [https://ieeexplore.ieee.org/iel7/9078688/10720652/10531112.pdf](https://ieeexplore.ieee.org/iel7/9078688/10720652/10531112.pdf)  
19. Parametric Contrastive Learning \- IEEE Xplore, truy cập vào tháng 12 10, 2025, [https://ieeexplore.ieee.org/iel7/9709627/9709628/09710141.pdf](https://ieeexplore.ieee.org/iel7/9709627/9709628/09710141.pdf)  
20. Parametric Contrastive Learning \- CVF Open Access, truy cập vào tháng 12 10, 2025, [https://openaccess.thecvf.com/content/ICCV2021/papers/Cui\_Parametric\_Contrastive\_Learning\_ICCV\_2021\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Parametric_Contrastive_Learning_ICCV_2021_paper.pdf)  
21. Long-tailed Recognition with Model Rebalancing \- arXiv, truy cập vào tháng 12 10, 2025, [https://arxiv.org/html/2510.08177v1](https://arxiv.org/html/2510.08177v1)  
22. Long-tailed Recognition with Model Rebalancing \- OpenReview, truy cập vào tháng 12 10, 2025, [https://openreview.net/attachment?id=jDKhljBQb8\&name=pdf](https://openreview.net/attachment?id=jDKhljBQb8&name=pdf)  
23. Implementing the Macro F1 Score in Keras: Do's and Don'ts \- Neptune.ai, truy cập vào tháng 12 10, 2025, [https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras](https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras)  
24. 6.4. Classification on imbalanced labels with focal loss \- skscope, truy cập vào tháng 12 10, 2025, [https://skscope.readthedocs.io/en/0.1.7/gallery/Miscellaneous/focal-loss-with-imbalanced-data.html](https://skscope.readthedocs.io/en/0.1.7/gallery/Miscellaneous/focal-loss-with-imbalanced-data.html)  
25. \[D\] Does directly optimizing the soft F1 loss make sense ? : r/MachineLearning \- Reddit, truy cập vào tháng 12 10, 2025, [https://www.reddit.com/r/MachineLearning/comments/ngk7w4/d\_does\_directly\_optimizing\_the\_soft\_f1\_loss\_make/](https://www.reddit.com/r/MachineLearning/comments/ngk7w4/d_does_directly_optimizing_the_soft_f1_loss_make/)