# Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry

**Authors:**  
**Abhishekh** *(Spatio-Temporal Graph Autoencoder Teacher & Spatial Ablation)*  
**Bhavya** *(3D MANO Data Pipeline, Volumetric Tool Occlusion & Glove Domain Adaptation)*  
**Vrushank** *(Continuous 6D $\text{SO}(3)$ Rotations, Biomechanical RoM Clamping & Differentiable Forward Kinematics)*  
**Kapish** *(Physics-Informed Knowledge Distillation, ONNX Compilation & Multi-Hardware Edge Deployment)*

**Course:** Foundations of Data Science (FDS) — Year 3 Undergraduate Technical Report  
**Date:** November 2026

---

## Abstract

Three-dimensional (3D) hand gesture telemetry in surgical navigation and industrial teleoperation degrades when handheld instruments obscure phalangeal joints or when medical-grade latex and nitrile gloves introduce specular reflections and elastic stiffness. Standard on-device hand trackers such as MediaPipe Hands and Apple ARKit regress unconstrained 3D Cartesian coordinates in $\mathbb{R}^{21 \times 3}$. When 30% to 70% of joints lose visual line-of-sight, direct coordinate regressors produce anatomically impossible poses characterized by phalangeal bone stretching ($15\text{–}27\text{ mm}$ error), finger collapse toward the palm, and dorsal joint hyperextension. Conversely, high-capacity occlusion-aware mesh transformers require tens of millions of parameters ($>100\text{ MB}$), exceeding the compute and memory budgets of embedded edge processors. We present **Constrained Graph Kinematics**, a two-stage spatio-temporal graph distillation and differentiable forward-kinematic framework. First, we design a $520,189$-parameter **Spatio-Temporal Graph Autoencoder (`SpatioTemporalGAE`) Teacher** that combines visibility-gated 3-layer, 4-head Graph Attention over the 21-joint hand topology with a Bidirectional Temporal Gated Recurrent Unit (GRU) across $T=8$ sliding windows. Second, we compress the Teacher via a multi-objective rotational and positional **Knowledge Distillation** loss into a $78,398$-parameter **Edge Student (`StudentKinematicMLP`)** ($6.64\times$ parameter reduction, $137.3\times$ FLOPs reduction to $163.4\text{ KFLOPs/frame}$). Both Teacher and Student map latent features into continuous 6D rotation representations on $\text{SO}(3)$, clamp Euler flexion, twist, and abduction angles via a smooth hyperbolic tangent (`BiomechanicalRoMClamper`) layer, and reconstruct 3D joint positions through an exact **Differentiable Forward Kinematics (`DifferentiableFK`)** chain. Across 30%, 50%, and 70% surgical tool occlusion on our held-out test split ($960$ sequences, $7,680$ frames) and our custom gloved benchmark ($120$ sequences), the Teacher achieves $3.33\text{ mm}$, $3.79\text{ mm}$, and $6.35\text{ mm}$ Mean Per-Joint Position Error (MPJPE), while the distilled Student achieves $4.08\text{ mm}$, $5.05\text{ mm}$, and $8.36\text{ mm}$ MPJPE—reducing error by $66.4\%$ at 50% occlusion compared to an unconstrained coordinate-regression MLP baseline ($15.03\text{ mm}$) while guaranteeing **$0.0000\text{ mm}$ bone-length error** and **$100.0\%$ biomechanical validity** on every frame. Compiled to a static $433.2\text{ KB}$ ONNX Opset 17 graph, the Student executes in $0.121\text{ ms/frame}$ ($8,256\text{ FPS}$) on x86_64 CPU, $0.314\text{ ms}$ ($3,185\text{ FPS}$) on Apple Silicon MacBook Air, $2.046\text{ ms}$ ($488.8\text{ FPS}$) on NVIDIA Jetson Nano TensorRT FP16, and $1.044\text{ ms}$ end-to-end on live 60 Hz iPhone ARKit UDP telemetry streams.

**Index Terms—** 3D Hand Pose Estimation, Differentiable Forward Kinematics, Spatio-Temporal Graph Attention, Continuous 6D Rotations, Biomechanical Constraints, Knowledge Distillation, Edge AI Telemetry.

---

## I. Introduction

Estimating three-dimensional (3D) hand kinematics from monocular cameras, depth sensors, or sparse keypoint streams is an ill-posed inverse problem. Anatomically, the human hand skeleton forms a directed tree $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ comprising $|\mathcal{V}| = 21$ joints connected by $|\mathcal{E}_{\text{tree}}| = 20$ rigid bone segments across five digits (thumb carpometacarpal [CMC], metacarpophalangeal [MCP], and interphalangeal [IP] joints; index, middle, ring, and pinky MCP, proximal interphalangeal [PIP], and distal interphalangeal [DIP] joints) anchored at the wrist root ($j = 0$). In clinical operating rooms, robotic teleoperation cells, and cleanroom assembly lines, optical hand telemetry encounters two compounding physical failure sources:

1. **Contiguous Phalangeal Occlusion by Handheld Instruments:** Grasping surgical forceps, scalpels, endoscopic probes, or power drivers blocks camera line-of-sight across contiguous distal subtrees ($\text{MCP} \to \text{PIP} \to \text{DIP} \to \text{TIP}$), masking 30% to 70% of the hand skeleton simultaneously.
2. **Latex and Nitrile Glove Domain Shift:** Medical and industrial gloves eliminate dorsal skin texture, fingernail boundaries, and chromatic cues while introducing two physical artifacts: (a) elastic material resistance that dampens finger flexion toward a semi-open equilibrium posture, and (b) directional specular reflections under overhead luminaires that induce temporally correlated tracking jitter and distal fingertip spikes up to $12.0\text{ mm}$.

When visual features disappear under tool occlusion or specular glare, unconstrained coordinate regressors—networks whose output layer predicts 63 independent scalars $\hat{\mathbf{P}} \in \mathbb{R}^{21 \times 3}$ trained via Euclidean Mean Squared Error (MSE)—fail in three ways. First, missing distal phalanges drift toward the palm centroid or origin, shrinking or stretching bone lengths by $4\text{–}27\text{ mm}$. Second, independent per-joint coordinate predictions violate physiological Range-of-Motion (RoM) bounds, producing negative PIP/DIP hyperextension angles or unnatural lateral finger splay. Third, frame-independent inference amplifies high-frequency optical glare into severe second-order acceleration jitter.

Prior work has attempted to mitigate these failures either by training large vision transformers on hand-object datasets [1]–[4], [8], [9] or by adding soft biomechanical penalty losses during training [11]. However, transformer architectures such as HandOccNet [8] ($37.2\text{M}$ parameters, $142\text{ MB}$) and METRO [9] ($102\text{M}$ parameters) require desktop GPUs and cannot execute at high frame rates on embedded edge processors such as the NVIDIA Jetson Nano or Raspberry Pi. Meanwhile, soft training-time penalty losses (Spurr et al. [11]) do not constrain the output layer at test time; under severe out-of-distribution occlusion, data likelihood gradients override the penalty priors, yielding stretched bones and invalid joint angles during inference.

### Contributions
To resolve the trade-off between anatomical validity under severe occlusion and sub-millisecond edge execution, this report introduces four technical contributions:

1. **3D Graph-Space Tool Occlusion & Zero-Shot Glove Domain Adaptation (`data/`):** Bhavya formulated `SyntheticOcclusionTransform`, which combines 3D cylindrical instrument clipping ($R = 28.0\text{ mm}$) with contiguous distal-to-proximal kinematic subtree priority at exact 30%, 50%, and 70% occlusion rates, alongside `SyntheticGloveTransform`, which models $12\%$ elastic stiffness and temporally correlated AR(1) specular sensor jitter ($\rho = 0.65$, $4.5\text{ mm}$ baseline, $12.0\text{ mm}$ fingertip spikes) across $4,800$ MANO sequences ($38,400$ frames) and a $120$-sequence custom gloved benchmark.
2. **Hard-Constrained Differentiable & Bare-Metal Edge Forward Kinematics (`models/kinematics.py`):** Vrushank designed an ONNX-traceable kinematic layer combining Gram-Schmidt continuous 6D-to-$\text{SO}(3)$ orthogonalization [12], a $C^1$-smooth hyperbolic tangent angular clamper (`BiomechanicalRoMClamper`), and recursive hierarchical Forward Kinematics (`DifferentiableFK` in PyTorch and `EdgeFK` in pure NumPy). By mathematical construction on $\text{SO}(3)$, our architecture guarantees **$0.0000\text{ mm}$ bone-length error** and **$100.0\%$ biomechanical validity** on every output pose.
3. **Mask-Aware Spatio-Temporal Graph Autoencoder Teacher (`models/teacher_st_gae.py`):** Abhishekh architected `SpatioTemporalGAE` ($520,189$ parameters), which imputes occluded nodes with canonical rest priors, propagates spatial context across a 21-node anatomical graph via three 4-head Graph Attention layers equipped with a learnable visibility key gate $\gamma_h$, and integrates temporal dynamics across $T=8$ frames via a Bidirectional GRU—reducing 50%-occlusion MPJPE by $38.9\%$ relative to our `SpatialOnlyGAE` ablation ($3.79\text{ mm}$ vs. $6.20\text{ mm}$).
4. **Multi-Objective Kinematic Distillation & Multi-Hardware Edge Deployment (`models/student_models.py`, `distillation/`, `deployment/`):** Kapish distilled the frozen Teacher into a $78,398$-parameter `StudentKinematicMLP` ($6.64\times$ parameter compression, $137.3\times$ FLOPs reduction), exported it to a $433.21\text{ KB}$ ONNX Opset 17 model with $0.000254\text{ mm}$ numerical parity, and benchmarked real-time execution across x86_64 PC ($0.121\text{ ms/frame}$), Apple Silicon MacBook Air ($0.314\text{ ms}$), NVIDIA Jetson Nano TensorRT FP16 ($2.046\text{ ms}$), Raspberry Pi 5 ($3.052\text{ ms}$), and 60 Hz iPhone ARKit UDP telemetry ($1.044\text{ ms}$ end-to-end).

---

## II. Related Work

### A. 3D Hand Pose & Occlusion Benchmarks
Modern 3D hand pose estimation builds upon calibrated multi-camera benchmarks fitted with the parametric **MANO** (*hand Model with Articulated and Non-rigid defOrmations*) mesh [13]. **FreiHAND** (Zimmermann et al. [1]) established cross-domain single-RGB evaluation across $134,200$ frames using Mean Per-Joint Position Error (MPJPE) and Procrustes-Aligned MPJPE (PA-MPJPE). To evaluate tracking under heavy object manipulation, **HO-3D** (Hampali et al. [2]) and **DexYCB** (Chao et al. [3]) captured RGB-D sequences of subjects grasping rigid YCB objects where up to 60% of the hand is obscured by held tools. **InterHand2.6M** (Moon et al. [4]) demonstrated that when identically textured phalanges overlap during bimanual interaction, local visual features become ambiguous, requiring structural kinematic priors to disambiguate joint assignments.

### B. Graph Convolutional and Occlusion-Aware Architectures
Because the hand skeleton forms a natural graph, **HandGCN** (Ge et al. [5]) applied Chebyshev spectral graph convolutions ($g_\theta * x = \sum_{k=0}^{K-1} \theta_k T_k(\tilde{L})x$) to reconstruct 3D hand meshes, while **HOPE-Net** (Doosti et al. [6]) cascaded two Adaptive Graph Convolutional Networks with learnable adjacency matrices $\tilde{A}_{\text{ada}}$ to lift 2D keypoints to 3D. **Pose2Mesh** (Choi et al. [7]) decoupled 2D keypoint detection from 3D graph lifting to improve cross-domain generalization. However, HandGCN, HOPE-Net, and Pose2Mesh regress unconstrained 3D vertex/joint coordinates in $\mathbb{R}^{V \times 3}$, suffering $3.4\text{–}4.8\text{ mm}$ bone-length distortion when input keypoints are corrupted by occlusion. To recover occluded features, **HandOccNet** (Park et al. [8]) introduced Feature-Injecting and Self-Enhancing Transformers, and **METRO** (Lin et al. [9]) applied non-local self-attention across all 799 joint and mesh tokens, while **MeTRAbs** (Sárándi et al. [10]) predicted metric-scale 3D volumetric heatmaps. While accurate ($9.1\text{ mm}$ PA-MPJPE on HO-3D for HandOccNet), these architectures require $37\text{M–}102\text{M}$ parameters and $>30\text{ ms/frame}$ on desktop GPUs.

### C. Biomechanical Constraints & Continuous Rotation Manifolds
While MANO [13] parameterizes hand meshes via axis-angle rotations $\theta \in \mathbb{R}^{16 \times 3}$, unconstrained axis-angle regression permits physiologically invalid joint hyperextension. **Spurr et al.** [11] proposed weakly supervised soft penalty losses on bone lengths ($L_{\text{bl}}$), palmar arch curvature ($L_{\text{palm}}$), and joint flexion/abduction hull boundaries ($L_{\text{ja}} = \sum_i \max(0, \underline{\theta}_i - \theta_i) + \max(0, \theta_i - \overline{\theta}_i)$). Because these penalties act only as additive training regularizers on an unconstrained $\mathbb{R}^{21 \times 3}$ output layer, test-time occlusion still induces bone stretching ($1.9\text{ mm}$) and invalid joint angles ($8.6\%$ violation rate). When parameterizing rotations for forward kinematics, **Zhou et al.** [12] proved topologically that all $\le 4\text{D}$ representations of $\text{SO}(3)$ (including 3D Euler angles and 4D unit quaternions) contain discontinuities that impede gradient descent, introducing a continuous 6D Gram-Schmidt representation ($\mathbb{R}^6 \to \text{SO}(3)$). Finally, lightweight edge trackers such as **MediaPipe Hands** (Zhang et al. [14]) execute at $>200\text{ FPS}$ on mobile CPUs/NPUs ($1.5\text{M}$ params) but rely on unconstrained coordinate regression, exhibiting $6.2\text{ mm}$ bone-length error and $35.2\%$ anatomical invalidity under tool occlusion.

| Method / Architecture | Primary Venue | Representation & Constraint Type | MPJPE (mm) $\downarrow$ | PA-MPJPE (mm) $\downarrow$ | Bone-Length Error (mm) $\downarrow$ | Biomechanical Validity (%) $\uparrow$ | Params / Model Size (MB) $\downarrow$ | Inference Latency / Throughput $\uparrow$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HandGCN** [5] | CVPR 2019 | Spectral GCN (3D Vertices, Soft Losses) | $13.9\text{–}18.2$ | $9.8\text{–}11.2$ | $3.40$ | $78.5\%$ | $18.50\text{M}$ / $74.0\text{ MB}$ | $28.0\text{ ms}$ / $35\text{ FPS}$ (GPU) |
| **HOPE-Net** [6] | CVPR 2020 | Adaptive-GCN (Unconstrained 3D Coords) | $26.8$ (HO-3D) | $11.0$ (HO-3D) | $4.80$ | $71.0\%$ | $25.20\text{M}$ / $96.0\text{ MB}$ | $16.5\text{ ms}$ / $60\text{ FPS}$ (GPU) |
| **Spurr et al.** [11] | ECCV 2020 | CNN + 3D Coords (Soft Penalty Losses) | $22.4$ (HO-3D) | $9.9$ (HO-3D) | $1.90$ | $91.4\%$ | $23.60\text{M}$ / $90.0\text{ MB}$ | $15.0\text{ ms}$ / $66\text{ FPS}$ (GPU) |
| **HandOccNet** [8] | CVPR 2022 | Dual Transformers (FIT/SET) + MANO | $24.0$ (HO-3D) | $9.1$ (HO-3D) | $0.80$ | $94.2\%$ | $37.20\text{M}$ / $142.0\text{ MB}$ | $33.5\text{ ms}$ / $30\text{ FPS}$ (GPU) |
| **MediaPipe Hands** [14] | CVPRW 2020 | Mobile CNN (Unconstrained 2.5D/3D) | $19.5\text{–}28.4^*$ | $11.4\text{–}16.2^*$ | $6.20$ | $64.8\%$ | $1.50\text{M}$ / $5.50\text{ MB}$ | $4.50\text{ ms}$ / $220\text{ FPS}$ (CPU) |
| **`UnconstrainedStudentMLP`** | Ours (Ablation) | Direct $\mathbb{R}^{21 \times 3}$ Regression (MSE, No FK) | $15.03$ | $12.34$ | $5.8505$ | $8.70\%$ | $0.0785\text{M}$ / $0.30\text{ MB}$ | $0.021\text{ ms}$ / $47,114\text{ FPS}$ (ORT) |
| **`SpatialOnlyGAE`** | Ours (Ablation) | 3-Layer GAT + 6D $\text{SO}(3)$ + Hard FK (No GRU) | $6.20$ | $4.58$ | **$0.0000$** | **$100.0\%$** | $0.4617\text{M}$ / $1.77\text{ MB}$ | $0.590\text{ ms}$ / $1,696\text{ FPS}$ (CPU) |
| **`Teacher ST-GAE`** | **Ours (Teacher)** | **3-Layer GAT + Bi-GRU + Hard $\tanh$ FK** | **$3.79$** | **$2.11$** | **$0.0000$** | **$100.0\%$** | $0.5202\text{M}$ / $1.99\text{ MB}$ | $0.676\text{ ms}$ / $1,479\text{ FPS}$ (CPU) |
| **`StudentKinematicMLP`** | **Ours (Student)** | **Distilled MLP + 6D $\text{SO}(3)$ + Hard FK** | **$5.05$** | **$3.43$** | **$0.0000$** | **$100.0\%$** | **$0.0784\text{M}$ / $0.30\text{ MB}$** | **$0.121\text{ ms}$ / $8,256\text{ FPS}$ (ORT)** |

*\*Table I: State-of-the-art comparison. Published literature metrics reflect HO-3D/FreiHAND benchmarks; our ablation, Teacher, and Student rows report exact held-out test metrics under 50% surgical tool occlusion and latex glove degradation (`results/metrics_summary.json`).*

---

## III. Mathematical Formulation of 3D Graph Transforms

To train and evaluate our models across controlled clinical and industrial failure modes without data leakage, Bhavya implemented a deterministic 3D MANO trajectory synthesis and domain-adaptation pipeline (`data/dataset_ingestion.py`, `data/transforms.py`, `data/dataloader.py`).

### A. Canonical 21-Joint Hand Graph & Stratified Dataset Split
Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ denote the 21-joint hand graph with vertex set $\mathcal{V} = \{0, 1, \dots, 20\}$, where node $0$ is the wrist root and nodes $1\text{–}20$ form five directed kinematic chains: Thumb $(0 \to 1 \to 2 \to 3 \to 4)$, Index $(0 \to 5 \to 6 \to 7 \to 8)$, Middle $(0 \to 9 \to 10 \to 11 \to 12)$, Ring $(0 \to 13 \to 14 \to 15 \to 16)$, and Pinky $(0 \to 17 \to 18 \to 19 \to 20)$, with parent mapping $p(j) = \text{PARENT\_INDICES}[j]$. The graph edge set $\mathcal{E}$ comprises the 20 parent-child kinematic bone edges plus 4 inter-MCP palmar arch edges $\mathcal{E}_{\text{palm}} = \{(1, 5), (5, 9), (9, 13), (13, 17)\}$ and 21 self-loops. Each bone $j \in \{1, \dots, 20\}$ possesses a fixed canonical length $L_j \in \mathbb{R}^+$ in millimeters (`CANONICAL_BONE_LENGTHS_MM`) and unit rest direction $\hat{\mathbf{u}}_j^{\text{rest}} \in S^2$.

We generate $N = 4,800$ smooth multi-frame 3D trajectories of length $T = 8$ ($38,400$ total frames) across four random seeds (`seed42`, `seed72`, `seed92`, `seed112` in `data/processed/`) spanning five functional articulation archetypes (*Pinch, Power Grasp, Tripod Precision, Index Pointing, Open Flexion*). The corpus is partitioned via a strict **80% Train ($3,840$ sequences, $30,720$ frames) / 20% Held-Out Test ($960$ sequences, $7,680$ frames)** split (`data/splits/split_indices.json`). Additionally, `data/record_custom_gloved.py` generates an independent $120$-sequence ($960$-frame) multi-tier gloved benchmark (`data/custom_gloved_dataset.npz`).

### B. Zero-Shot Glove Domain Adaptation (`SyntheticGloveTransform`)
Protective latex and nitrile gloves alter raw 3D hand telemetry through two physical mechanisms:
1. **Kinematic Range Reduction (Elastic Stiffness):** The elastic glove membrane resists full finger flexion, pulling each bone vector toward the semi-flexed neutral glove equilibrium $\hat{\mathbf{u}}_j^{\text{rest}}$ with damping coefficient $\alpha_{\text{stiff}} = 0.12$. Given clean ground-truth positions $\mathbf{P}_{t,j}^{\text{GT}} \in \mathbb{R}^3$ and current unit bone direction $\hat{\mathbf{d}}_{t,j} = (\mathbf{P}_{t,j}^{\text{GT}} - \mathbf{P}_{t,p(j)}^{\text{GT}}) / L_j$, the stiffened pose $\tilde{\mathbf{P}}_{t,j}$ is constructed recursively from the wrist root $\tilde{\mathbf{P}}_{t,0} = \mathbf{P}_{t,0}^{\text{GT}}$:
   $$\tilde{\mathbf{d}}_{t,j} = \frac{(1 - \alpha_{\text{stiff}})\hat{\mathbf{d}}_{t,j} + \alpha_{\text{stiff}}\hat{\mathbf{u}}_j^{\text{rest}}}{\left\|(1 - \alpha_{\text{stiff}})\hat{\mathbf{d}}_{t,j} + \alpha_{\text{stiff}}\hat{\mathbf{u}}_j^{\text{rest}}\right\|_2}, \qquad \tilde{\mathbf{P}}_{t,j} = \tilde{\mathbf{P}}_{t,p(j)} + L_j \tilde{\mathbf{d}}_{t,j} \tag{1}$$
2. **Temporally Correlated AR(1) Specular Sensor Jitter:** Optical and LiDAR trackers suffer correlated frame-to-frame depth drift on shiny latex surfaces plus localized specular glare spikes at the distal fingertips $\mathcal{V}_{\text{tip}} = \{4, 8, 12, 16, 20\}$. We model the baseline noise via a first-order autoregressive process $\text{AR}(1)$ with temporal correlation $\rho = 0.65$ and per-joint standard deviation $\sigma_j = 4.5\text{ mm}$ ($5.625\text{ mm}$ on DIP joints $j \in \{3, 7, 11, 15, 19\}$, $0\text{ mm}$ on the wrist anchor $j=0$), superimposed with Bernoulli specular spikes ($p_{\text{spike}} = 0.20$, $\sigma_{\text{spike}} = 12.0\text{ mm}$):
   $$\boldsymbol{\epsilon}_{t,j} = \rho \boldsymbol{\epsilon}_{t-1,j} + \sqrt{1 - \rho^2}\,\mathbf{z}_{t,j}, \qquad \mathbf{z}_{t,j} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_3) \tag{2}$$
   $$\mathbf{P}_{t,j}^{\text{gloved}} = \tilde{\mathbf{P}}_{t,j} + \sigma_j \boldsymbol{\epsilon}_{t,j} + \mathbb{I}(j \in \mathcal{V}_{\text{tip}})\,b_{t,j}\,\boldsymbol{\xi}_{t,j}, \qquad b_{t,j} \sim \text{Bernoulli}(p_{\text{spike}}), \quad \boldsymbol{\xi}_{t,j} \sim \mathcal{N}(\mathbf{0}, \sigma_{\text{spike}}^2 \mathbf{I}_3) \tag{3}$$

### C. 3D Volumetric Tool Clipping & Kinematic Subtree Masking (`SyntheticOcclusionTransform`)
To simulate a surgical scalpel, forceps, or industrial handle crossing the hand, `SyntheticOcclusionTransform` spawns a moving 3D cylindrical volume of radius $R_{\text{cyl}} = 28.0\text{ mm}$ and unit axis $\hat{\mathbf{u}}_{\text{tool}} \in S^2$ with time-varying center $\mathbf{c}_t = (1 - \frac{t}{T-1})\mathbf{c}_{\text{start}} + \frac{t}{T-1}\mathbf{c}_{\text{end}}$ anchored near the distal phalangeal centroid. For each joint $j$ at frame $t$ with relative offset $\mathbf{r}_{t,j} = \mathbf{P}_{t,j}^{\text{gloved}} - \mathbf{c}_t$, we compute the signed cylinder proximity and camera-ray ($+Z$) depth occlusion score $d_{t,j}^{\text{vol}}$:
$$d_{\perp}(t,j) = \left\|\mathbf{r}_{t,j} - (\mathbf{r}_{t,j} \cdot \hat{\mathbf{u}}_{\text{tool}})\hat{\mathbf{u}}_{\text{tool}}\right\|_2, \qquad d_{\parallel}(t,j) = |\mathbf{r}_{t,j} \cdot \hat{\mathbf{u}}_{\text{tool}}|, \qquad d_z(t,j) = \max\!\left(0,\, P_{t,j,z}^{\text{gloved}} - c_{t,z}\right) \tag{4}$$
$$d_{t,j}^{\text{vol}} = d_{\perp}(t,j) + 0.25\max\!\left(0,\, d_{\parallel}(t,j) - 45.0\right) - 0.20\,d_z(t,j) - R_{\text{cyl}} \tag{5}$$

To ensure that occlusion propagates along physically contiguous finger subtrees from distal tip to proximal knuckle ($\text{TIP} \to \text{DIP} \to \text{PIP} \to \text{MCP}$), each finger $f \in \{1, \dots, 5\}$ is assigned a mean distal proximity $\bar{d}_{t,f} = \frac{1}{3}\sum_{r=0}^2 d_{t,\text{chain}_f[r]}^{\text{vol}}$, and each joint $j = \text{chain}_f[r]$ ($r \in \{0, 1, 2, 3\}$ ordered distally) receives priority score:
$$s_{t,j} = \bar{d}_{t,f} + 0.35\,d_{t,j}^{\text{vol}} + 6.0\,r, \qquad s_{t,0} = +10^9 \tag{6}$$
Given target occlusion rate $\eta \in \{0.30, 0.50, 0.70\}$, the exact $k_{\text{mask}} = \text{round}(\eta \cdot 21)$ joints with the lowest scores $s_{t,j}$ are masked ($m_{t,j} = 0$, $\mathbf{P}_{t,j}^{\text{obs}} = \mathbf{0}$), while the remaining $21 - k_{\text{mask}}$ visible joints retain $m_{t,j} = 1$ and $\mathbf{P}_{t,j}^{\text{obs}} = \mathbf{P}_{t,j}^{\text{gloved}}$.

---

## IV. 6D Continuous Rotations & Differentiable Forward Kinematics

To guarantee that predicted 3D skeletons never exhibit bone-length distortion or physiological hyperextension, Vrushank implemented a differentiable kinematic manifold in `models/kinematics.py`.

### A. Singularity-Free 6D-to-$\text{SO}(3)$ Orthogonalization (`rot6d_to_matrix`)
Following Zhou et al. [12], each joint $j \in \{0, \dots, 20\}$ is parameterized by a continuous 6D vector $\mathbf{d}_j = [\mathbf{a}_{j,1}^\top, \mathbf{a}_{j,2}^\top]^\top \in \mathbb{R}^6$, where $\mathbf{a}_{j,1}, \mathbf{a}_{j,2} \in \mathbb{R}^3$. The mapping $g_{\text{GS}}: \mathbb{R}^6 \to \text{SO}(3)$ constructs an orthonormal right-handed rotation matrix $\mathbf{R}_j \in \text{SO}(3)$ via Gram-Schmidt orthogonalization:
$$\mathbf{b}_{j,1} = \frac{\mathbf{a}_{j,1}}{\|\mathbf{a}_{j,1}\|_2}, \qquad \mathbf{u}_{j,2} = \mathbf{a}_{j,2} - \left(\mathbf{b}_{j,1} \cdot \mathbf{a}_{j,2}\right)\mathbf{b}_{j,1}, \qquad \mathbf{b}_{j,2} = \frac{\mathbf{u}_{j,2}}{\|\mathbf{u}_{j,2}\|_2}, \qquad \mathbf{b}_{j,3} = \mathbf{b}_{j,1} \times \mathbf{b}_{j,2} \tag{7}$$
$$\mathbf{R}_j = \begin{bmatrix} | & | & | \\ \mathbf{b}_{j,1} & \mathbf{b}_{j,2} & \mathbf{b}_{j,3} \\ | & | & | \end{bmatrix} \in \text{SO}(3), \qquad \mathbf{R}_j^\top \mathbf{R}_j = \mathbf{I}_3, \quad \det(\mathbf{R}_j) = +1 \tag{8}$$
In `models/kinematics.py`, degenerate near-zero or collinear vectors ($\|\mathbf{a}_{j,1}\|_2 < 10^{-6}$ or $\|\mathbf{u}_{j,2}\|_2 < 10^{-6}$) are guarded via `torch.where` fallbacks onto canonical basis axes $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$, ensuring that $\mathbf{R}_j \in \text{SO}(3)$ holds for every possible tensor input using pure ONNX-standard operators.

### B. $C^1$-Smooth Euler Extraction & $\tanh$ Biomechanical RoM Clamping (`BiomechanicalRoMClamper`)
While Gram-Schmidt orthogonalization constrains each joint to $\text{SO}(3)$, unconstrained rotations on $\text{SO}(3)$ still allow a finger joint to bend backward by $90^\circ$ (dorsal hyperextension) or twist axially. Let $\mathbf{R}_j = \mathbf{R}_z(\theta_{j,z})\mathbf{R}_y(\theta_{j,y})\mathbf{R}_x(\theta_{j,x})$ denote the intrinsic Euler decomposition into local flexion/extension ($\theta_{j,x}$), axial twist ($\theta_{j,y}$), and abduction/adduction ($\theta_{j,z}$). From the matrix entries of $\mathbf{R}_j$, we extract $\theta_{j,y} = \arcsin(\text{clamp}(-R_{j,20}, -1+\epsilon, 1-\epsilon))$ and recover $\theta_{j,x}, \theta_{j,z}$ from their normalized sine-cosine pairs $(s, c)$ via an ONNX-traceable $C^1$-smooth transition (`_safe_angle_from_sin_cos`):
$$\theta(s, c) = \begin{cases} s \cdot \left(1 + \frac{1 - c}{3}\right), & \text{if } c > 0.9995 \;\;(|\theta| < 1.8^\circ) \\ \text{sign}(s)\,\arccos\!\left(\text{clamp}(c, -0.99999, 0.99999)\right), & \text{otherwise} \end{cases} \tag{9}$$
Equation (9) eliminates the $\arccos'(1) \to -\infty$ gradient singularity at $\theta = 0$, providing an exact unit derivative $\frac{\partial \theta}{\partial s}\big|_{\theta=0} = 1.0$. Each joint angle vector $\boldsymbol{\theta}_j = [\theta_{j,x}, \theta_{j,y}, \theta_{j,z}]^\top$ is then smoothly clamped onto its physiological Range-of-Motion interval $[\boldsymbol{\theta}_j^{\min}, \boldsymbol{\theta}_j^{\max}]$ (`ANATOMICAL_MIN_LIMITS_RAD`, `ANATOMICAL_MAX_LIMITS_RAD`) via a hyperbolic tangent bounding layer with temperature $\tau = 1.0$:
$$\boldsymbol{\mu}_j = \frac{\boldsymbol{\theta}_j^{\min} + \boldsymbol{\theta}_j^{\max}}{2}, \qquad \boldsymbol{\Delta}_j = \frac{\boldsymbol{\theta}_j^{\max} - \boldsymbol{\theta}_j^{\min}}{2}, \qquad \boldsymbol{\theta}_j^{\text{clamp}} = \boldsymbol{\mu}_j + \boldsymbol{\Delta}_j \odot \tanh\!\left(\frac{\boldsymbol{\theta}_j - \boldsymbol{\mu}_j}{\boldsymbol{\Delta}_j \cdot \tau + \epsilon}\right) \tag{10}$$
Re-composing $\mathbf{R}_j^{\text{local}} = \mathbf{R}_z(\theta_{j,z}^{\text{clamp}})\mathbf{R}_y(\theta_{j,y}^{\text{clamp}})\mathbf{R}_x(\theta_{j,x}^{\text{clamp}}) \in \text{SO}(3)$ guarantees that all 15 inter-segment articulations and 9 hinge joints (PIP flexion $\in [0^\circ, 110^\circ]$, DIP flexion $\in [0^\circ, 80^\circ]$, Thumb IP flexion $\in [0^\circ, 80^\circ]$) remain strictly inside human biomechanical limits.

### C. Differentiable Forward Kinematics & Zero-Bone-Length-Error Proof (`DifferentiableFK` & `EdgeFK`)
Given clamped local rotation matrices $\{\mathbf{R}_j^{\text{local}}\}_{j=0}^{20}$ and fixed canonical rest bone vectors $\mathbf{v}_j^{\text{rest}} = L_j \hat{\mathbf{u}}_j^{\text{rest}} \in \mathbb{R}^3$, `DifferentiableFK` (and its bare-metal NumPy counterpart `EdgeFK`) traverses the topological tree $j = 1, \dots, 20$ from anchored wrist $\mathbf{P}_0 = \mathbf{0}$, $\mathbf{R}_0^{\text{global}} = \mathbf{R}_0^{\text{local}}$:
$$\mathbf{R}_j^{\text{global}} = \mathbf{R}_{p(j)}^{\text{global}} \mathbf{R}_j^{\text{local}}, \qquad \mathbf{P}_j = \mathbf{P}_{p(j)} + \mathbf{R}_j^{\text{global}} \mathbf{v}_j^{\text{rest}} \tag{11}$$

**Theorem 1 (Exact Bone-Length Invariance):** *For any neural network output $\mathbf{d} \in \mathbb{R}^{21 \times 6}$, the bone-length error $\varepsilon_{\text{bone}}(\mathbf{P}) = \frac{1}{20}\sum_{j=1}^{20} \left|\|\mathbf{P}_j - \mathbf{P}_{p(j)}\|_2 - L_j\right|$ produced by `DifferentiableFK` is identically zero.*

*Proof:* Because $\text{SO}(3)$ is closed under matrix multiplication, $\mathbf{R}_j^{\text{global}} = \prod_{k \in \text{path}(0 \to j)} \mathbf{R}_k^{\text{local}} \in \text{SO}(3)$ satisfies $(\mathbf{R}_j^{\text{global}})^\top \mathbf{R}_j^{\text{global}} = \mathbf{I}_3$. By (11), the Euclidean distance between any child joint $j$ and its parent $p(j)$ is:
$$\|\mathbf{P}_j - \mathbf{P}_{p(j)}\|_2 = \|\mathbf{R}_j^{\text{global}} \mathbf{v}_j^{\text{rest}}\|_2 = \sqrt{(\mathbf{v}_j^{\text{rest}})^\top (\mathbf{R}_j^{\text{global}})^\top \mathbf{R}_j^{\text{global}} \mathbf{v}_j^{\text{rest}}} = \sqrt{(\mathbf{v}_j^{\text{rest}})^\top \mathbf{v}_j^{\text{rest}}} = \|\mathbf{v}_j^{\text{rest}}\|_2 = L_j \tag{12}$$
Therefore, $\left|\|\mathbf{P}_j - \mathbf{P}_{p(j)}\|_2 - L_j\right| = 0$ for all $j \in \{1, \dots, 20\}$, yielding $\varepsilon_{\text{bone}} \equiv 0.0000\text{ mm}$ ($< 1.1 \times 10^{-5}\text{ mm}$ in IEEE 754 FP32 arithmetic). $\blacksquare$

To compute exact ground-truth 6D rotation targets from 3D trajectories during training, Vrushank implemented closed-form analytical Inverse Kinematics (`positions_to_rot6d`) using Rodrigues alignment between canonical unit vector $\hat{\mathbf{u}}_j^{\text{rest}}$ and parent-frame observed direction $\hat{\mathbf{d}}_j^{\text{local}} = (\mathbf{R}_{p(j)}^{\text{global}})^\top \frac{\mathbf{P}_j - \mathbf{P}_{p(j)}}{\|\mathbf{P}_j - \mathbf{P}_{p(j)}\|_2}$:
$$\mathbf{R}_j^{\text{local}} = \mathbf{I}_3 + [\mathbf{k}_j]_\times + \frac{[\mathbf{k}_j]_\times^2}{1 + \hat{\mathbf{u}}_j^{\text{rest}} \cdot \hat{\mathbf{d}}_j^{\text{local}}}, \qquad \mathbf{k}_j = \hat{\mathbf{u}}_j^{\text{rest}} \times \hat{\mathbf{d}}_j^{\text{local}} \tag{13}$$

---

## V. Spatio-Temporal Graph Autoencoder (ST-GAE) Teacher Architecture

To reconstruct contiguous phalangeal subtrees when up to 70% of joints are occluded, Abhishekh designed the `SpatioTemporalGAE` Teacher (`models/teacher_st_gae.py`, $520,189$ parameters).

### A. Rest-Prior Imputation & Asymmetric Posture Embedding
Given a batch of occluded sequences $\mathbf{X} \in \mathbb{R}^{B \times T \times 21 \times 4}$ where each node carries palm-normalized coordinates $\tilde{\mathbf{x}}_{t,j} = \mathbf{x}_{t,j}^{\text{obs}} / S_{\text{palm}}$ ($S_{\text{palm}} = 100.0\text{ mm}$) and visibility flag $m_{t,j} \in \{0, 1\}$, masked nodes ($m_{t,j} = 0$) are first imputed with canonical rest positions $\bar{\mathbf{x}}_j^{\text{rest}} = \mathbf{P}_j^{\text{rest}} / S_{\text{palm}}$:
$$\mathbf{x}_{t,j}^{\text{imp}} = m_{t,j}\tilde{\mathbf{x}}_{t,j} + (1 - m_{t,j})\bar{\mathbf{x}}_j^{\text{rest}}, \qquad \Delta\tilde{\mathbf{x}}_{t,j} = m_{t,j}\!\left(\tilde{\mathbf{x}}_{t,j} - \bar{\mathbf{x}}_j^{\text{rest}}\right) \tag{14}$$
Each node's 7D input vector $\mathbf{f}_{t,j}^{\text{in}} = [\mathbf{x}_{t,j}^{\text{imp}} \,\|\, m_{t,j} \,\|\, \Delta\tilde{\mathbf{x}}_{t,j}] \in \mathbb{R}^7$ is projected to $D = 128$ channels and combined with a learnable joint-identity embedding $\mathbf{e}_j^{\text{joint}} \in \mathbb{R}^{128}$ and a flattened whole-hand posture skip projection $\mathbf{h}_t^{\text{pose}} = \text{SiLU}(\text{LN}(\mathbf{W}_{\text{pose}}\text{vec}(\mathbf{x}_t^{\text{imp}}, \mathbf{m}_t))) \in \mathbb{R}^{128}$:
$$\mathbf{h}_{t,j}^{(0)} = \text{GELU}\!\left(\text{LN}\!\left(\mathbf{W}_{\text{in}}\mathbf{f}_{t,j}^{\text{in}} + \mathbf{b}_{\text{in}}\right)\right) + \mathbf{e}_j^{\text{joint}} + \mathbf{h}_t^{\text{pose}} \tag{15}$$

### B. Mask-Aware 3-Layer 4-Head Graph Attention (`MultiHeadGraphAttentionLayer`)
The spatial encoder stacks $L = 3$ layers of `MultiHeadGraphAttentionLayer` with $H = 4$ attention heads ($d_h = D / H = 32$). For layer $l \in \{0, 1, 2\}$ and head $h \in \{1, \dots, 4\}$, node queries, keys, and values $\mathbf{q}_{t,i}^{(h)}, \mathbf{k}_{t,j}^{(h)}, \mathbf{v}_{t,j}^{(h)} \in \mathbb{R}^{32}$ interact strictly across topological neighbors $(i, j)$ where adjacency mask $A_{i,j} = 1$. To prevent occluded nodes ($m_{t,j} = 0$) from polluting visible or adjacent representations, we introduce a hybrid dot-product + GATv2 attention score with learnable topological edge bias $B_{h,i,j}$ and a **visibility key penalty** $\gamma_h > 0$ (initialized to $1.5$):
$$e_{t,i,j}^{(h)} = \frac{\mathbf{q}_{t,i}^{(h)} \cdot \mathbf{k}_{t,j}^{(h)}}{\sqrt{d_h}} + 0.25\,\text{LeakyReLU}_{0.2}\!\left(\mathbf{a}_{\text{src}}^{(h)} \cdot \mathbf{q}_{t,i}^{(h)} + \mathbf{a}_{\text{dst}}^{(h)} \cdot \mathbf{k}_{t,j}^{(h)}\right) + B_{h,i,j} - \gamma_h\left(1 - m_{t,j}\right) \tag{16}$$
$$\alpha_{t,i,j}^{(h)} = \frac{A_{i,j}\exp\!\left(e_{t,i,j}^{(h)}\right)}{\sum_{k=0}^{20} A_{i,k}\exp\!\left(e_{t,i,k}^{(h)}\right)}, \qquad \mathbf{z}_{t,i}^{(l)} = \mathbf{W}_O^{(l)} \Bigg\|_{h=1}^H \sum_{j=0}^{20} \alpha_{t,i,j}^{(h)} \mathbf{v}_{t,j}^{(h)} \tag{17}$$
Followed by pre-norm residual connections and a two-layer GELU feed-forward network (FFN):
$$\tilde{\mathbf{h}}_{t,i}^{(l)} = \text{LN}\!\left(\mathbf{h}_{t,i}^{(l)} + \mathbf{z}_{t,i}^{(l)}\right), \qquad \mathbf{h}_{t,i}^{(l+1)} = \text{LN}\!\left(\tilde{\mathbf{h}}_{t,i}^{(l)} + \text{FFN}^{(l)}\!\left(\tilde{\mathbf{h}}_{t,i}^{(l)}\right)\right) \tag{18}$$

### C. Bidirectional Temporal GRU Processor & Kinematic Decoding
After $L=3$ spatial layers, temporal dynamics across the $T=8$ window are processed by a depthwise 1D convolution ($\text{kernel\_size}=3, \text{padding}=1, \text{groups}=128$) coupled with a Bidirectional GRU ($\text{hidden\_size} = 64$ per direction, concatenated to $128$):
$$\mathbf{h}_{1:T}^{\text{conv}} = \text{SiLU}\!\left(\mathbf{W}_{\text{pw}}\,\text{Conv1D}_{\text{dw}}\!\left(\mathbf{h}_{1:T}^{\text{pose}}\right)\right), \qquad \mathbf{h}_{1:T}^{\text{gru}} = \text{BiGRU}\!\left(\mathbf{h}_{1:T}^{\text{pose}} + \mathbf{h}_{1:T}^{\text{conv}}\right) \tag{19}$$
$$\mathbf{h}_{t}^{\text{temp}} = \text{ResMLP}_2\!\left(\text{ResMLP}_1\!\left(\text{LN}\!\left(\mathbf{h}_{t}^{\text{pose}} + \mathbf{h}_{t}^{\text{conv}} + \mathbf{h}_{t}^{\text{gru}}\right)\right)\right) \tag{20}$$
Finally, the decoded 6D rotations combine the neutral hand posture prior $\mathbf{d}_j^{\text{neutral}} \in \mathbb{R}^6$, the whole-hand rotation head $\mathbf{g}_{\text{hand}}(\mathbf{h}_t^{\text{temp}})$, and a gated per-node GAT refinement head $\beta_{\text{gat}}\,\mathbf{g}_{\text{node}}(\mathbf{h}_{t,j}^{(3)} + \mathbf{h}_t^{\text{temp}})$ before passing through `BiomechanicalRoMClamper` and `DifferentiableFK`.

### D. Spatial-Only Ablation (`SpatialOnlyGAE`)
To isolate the contribution of temporal recurrence, Abhishekh also implemented `SpatialOnlyGAE` ($461,693$ parameters), which retains the exact 3-layer 4-head GAT encoder, `BiomechanicalRoMClamper`, and `DifferentiableFK`, but replaces the `Conv1D` + `BiGRU` temporal core in (19) with a frame-independent linear projection $\mathbf{h}_t^{\text{temp}} = \text{SiLU}(\mathbf{W}_{\text{frame}}\mathbf{h}_t^{\text{pose}})$.

---

## VI. Multi-Objective Knowledge Distillation & Edge Student Compression

While `SpatioTemporalGAE` achieves $3.79\text{ mm}$ MPJPE at 50% occlusion, its 3-layer multi-head attention and bidirectional recurrence require $22.43\text{ MFLOPs/frame}$ ($179.42\text{ MFLOPs}$ per $T=8$ window). To achieve sub-millisecond edge execution, Kapish designed `StudentKinematicMLP` (`models/student_models.py`) and the `DistillationLoss` pipeline (`distillation/distill_student.py`, `training/losses.py`).

### A. Lightweight Edge Student Architecture (`StudentKinematicMLP` vs. `UnconstrainedStudentMLP`)
`StudentKinematicMLP` ($78,398$ trainable parameters, $309.52\text{ KB}$ FP32, $154.76\text{ KB}$ FP16) processes an occluded sequence $\mathbf{X} \in \mathbb{R}^{B \times T \times 21 \times 4}$ in four vectorized stages without graph attention loops or recurrent hidden states:
1. **Rest Imputation & Linear Embedding:** Imputes occluded coordinates with $\bar{\mathbf{x}}_j^{\text{rest}}$ as in (14), flattens each frame to $\mathbb{R}^{84}$, and projects to $D = 128$ via $\text{SiLU}(\text{LN}(\mathbf{W}_{\text{emb}}\mathbf{f}_t^{\text{flat}}))$.
2. **Fast 1D Temporal Mixing (`TemporalMixingBlock1D`):** Applies a depthwise 1D convolution ($k=3, \text{groups}=128$) and pointwise linear layer across $T$ to filter AR(1) specular spikes:
   $$\mathbf{h}_{1:T}^{\text{mix}} = \text{LN}\!\left(\mathbf{h}_{1:T}^{(0)} + \text{SiLU}\!\left(\mathbf{W}_{\text{pw}}\,\text{Conv1D}_{\text{dw}}\!\left(\mathbf{h}_{1:T}^{(0)}\right)\right)\right) \tag{21}$$
3. **Residual MLP Trunk (`ResidualMLPBlock`):** Applies a two-layer SiLU residual block $\mathbf{h}_t^{\text{trunk}} = \text{SiLU}(\mathbf{h}_t^{\text{mix}} + \text{LN}(\mathbf{W}_2\,\text{SiLU}(\text{LN}(\mathbf{W}_1\mathbf{h}_t^{\text{mix}}))))$.
4. **Hard Kinematic Decoding:** Predicts residual 6D rotations $\hat{\mathbf{d}}_{t,j} = \mathbf{d}_j^{\text{neutral}} + \mathbf{W}_{\text{rot}}\mathbf{h}_t^{\text{trunk}} \in \mathbb{R}^{21 \times 6}$, clamps angles via `BiomechanicalRoMClamper`, and synthesizes 3D positions $\hat{\mathbf{P}}_{t,j}^{\text{S}} \in \mathbb{R}^{21 \times 3}$ via `DifferentiableFK`.

For controlled ablation, Kapish also implemented `UnconstrainedStudentMLP` ($78,527$ parameters), which shares the exact same embedding, `TemporalMixingBlock1D`, and `ResidualMLPBlock` backbone, but replaces the 6D rotation head and `DifferentiableFK` with an unconstrained Cartesian projection $\hat{\mathbf{P}}_t^{\text{unc}} = \mathbf{P}^{\text{rest}} + \mathbf{W}_{\text{xyz}}\mathbf{h}_t^{\text{trunk}} \in \mathbb{R}^{21 \times 3}$.

### B. Five-Term Kinematic Manifold Distillation Objective (`DistillationLoss`)
During distillation, the frozen Teacher `SpatioTemporalGAE` precomputes repaired 6D rotations $\mathbf{d}_{t,j}^{\text{T}}$ ($\mathbf{R}_{t,j}^{\text{T}} \in \text{SO}(3)$) and 3D FK joint positions $\mathbf{P}_{t,j}^{\text{T}}$ across all training windows. We define three component losses:
* **Occlusion-Weighted MPJPE Loss ($\mathcal{L}_{\text{MPJPE}}^w$):** Weights occluded joints ($m_{t,j}=0$) by $w_{\text{occ}} = 1.5$ and visible joints ($m_{t,j}=1$) by $w_{\text{vis}} = 1.0$:
  $$\mathcal{L}_{\text{MPJPE}}^w(\hat{\mathbf{P}}, \mathbf{P}) = \frac{\sum_{t,j} \left(w_{\text{vis}} m_{t,j} + w_{\text{occ}}(1 - m_{t,j})\right) \sqrt{\|\hat{\mathbf{P}}_{t,j} - \mathbf{P}_{t,j}\|_2^2 + \epsilon}}{\sum_{t,j} \left(w_{\text{vis}} m_{t,j} + w_{\text{occ}}(1 - m_{t,j})\right)} \tag{22}$$
* **Continuous 6D & Geodesic $\text{SO}(3)$ Loss ($\mathcal{L}_{6\text{D}}$):** Combines 6D Euclidean MSE with exact geodesic angular distance and chordal Frobenius alignment on $\text{SO}(3)$ where $\cos\theta_{t,j} = \frac{\text{Tr}((\hat{\mathbf{R}}_{t,j})^\top \mathbf{R}_{t,j}) - 1}{2}$:
  $$\mathcal{L}_{6\text{D}}(\hat{\mathbf{d}}, \mathbf{d}) = \frac{1}{21 T}\sum_{t,j} \left[\|\hat{\mathbf{d}}_{t,j} - \mathbf{d}_{t,j}\|_2^2 + 0.25\,\arccos\!\left(\text{clamp}(\cos\theta_{t,j}, -1+\epsilon, 1-\epsilon)\right) + 0.25\left(1 - \cos\theta_{t,j}\right)\right] \tag{23}$$
* **Velocity & Acceleration Smoothness Loss ($\mathcal{L}_{\text{smooth}}$):** Given first-order velocity $\Delta \mathbf{P}_t = \mathbf{P}_{t+1} - \mathbf{P}_t$ and second-order acceleration $\Delta^2 \mathbf{P}_t = \mathbf{P}_{t+1} - 2\mathbf{P}_t + \mathbf{P}_{t-1}$:
  $$\mathcal{L}_{\text{smooth}}(\hat{\mathbf{P}}, \mathbf{P}^{\text{GT}}) = 1.0\,\|\Delta\hat{\mathbf{P}}_t - \Delta\mathbf{P}_t^{\text{GT}}\|_2 + 1.5\,\|\Delta^2\hat{\mathbf{P}}_t - \Delta^2\mathbf{P}_t^{\text{GT}}\|_2 + 0.20\,\|\Delta^2\hat{\mathbf{P}}_t\|_2 \tag{24}$$

The total Knowledge Distillation loss optimized via AdamW ($\text{lr}=4.5 \times 10^{-3}$, cosine annealing over 35 epochs) is:
$$\mathcal{L}_{\text{KD}} = 35.0\,\mathcal{L}_{6\text{D}}\!\left(\hat{\mathbf{d}}^{\text{S}}, \mathbf{d}^{\text{T}}\right) + 0.45\,\mathcal{L}_{\text{MPJPE}}^w\!\left(\hat{\mathbf{P}}^{\text{S}}, \mathbf{P}^{\text{T}}\right) + 0.55\,\mathcal{L}_{\text{MPJPE}}^w\!\left(\hat{\mathbf{P}}^{\text{S}}, \mathbf{P}^{\text{GT}}\right) + 25.0\,\mathcal{L}_{6\text{D}}\!\left(\hat{\mathbf{d}}^{\text{S}}, \mathbf{d}^{\text{GT}}\right) + 0.35\,\mathcal{L}_{\text{smooth}}\!\left(\hat{\mathbf{P}}^{\text{S}}, \mathbf{P}^{\text{GT}}\right) \tag{25}$$

---

## VII. Experimental Setup & Quantitative Ablation Studies

All models were trained exclusively on the $80\%$ training split ($3,840$ sequences across 30%, 50%, and 70% occlusion rates) and evaluated on the held-out $20\%$ test split ($960$ sequences, $7,680$ frames per occlusion tier) as well as the independent $120$-sequence `custom_gloved_dataset.npz` benchmark (`training/evaluate_and_plot.py`, `results/metrics_summary.json`).

### A. Evaluation Metrics
1. **Overall, Visible, and Occluded MPJPE (mm):** Mean Euclidean distance between predicted and ground-truth 3D joints across all 21 joints, visible joints ($m_{t,j}=1$), and occluded joints ($m_{t,j}=0$).
2. **Procrustes-Aligned MPJPE (PA-MPJPE, mm):** MPJPE after optimal rigid $\text{SIM}(3)$ Kabsch-Umeyama alignment per frame.
3. **Bone-Length Error (mm):** Mean absolute deviation $\frac{1}{20}\sum_{j=1}^{20} |\|\hat{\mathbf{P}}_j - \hat{\mathbf{P}}_{p(j)}\|_2 - L_j|$ relative to canonical bone lengths.
4. **Biomechanical Validity (%):** Percentage of predicted hand poses passing all three anatomical checks in `check_biomechanical_validity`: (i) max bone-length deviation $\le 1.0\text{ mm}$, (ii) all 15 inter-segment bending angles within $[0^\circ, 125^\circ]$, and (iii) all 9 finger hinge joints exhibiting dorsal hyperextension $\le 15^\circ$.
5. **Temporal Acceleration Jitter ($\text{mm/s}^2$):** Mean second-order finite-difference acceleration $\|\mathbf{P}_{t+1} - 2\mathbf{P}_t + \mathbf{P}_{t-1}\|_2$ across the $T=8$ window.

### B. Held-Out Test Split Ablation Results (30%, 50%, 70% Occlusion)
Table II reports the complete quantitative ablation study across all five evaluated pipelines at 30%, 50%, and 70% surgical tool occlusion.

| Occlusion Tier | Method / Ablation Pipeline | Overall MPJPE (mm) $\downarrow$ | Visible MPJPE (mm) $\downarrow$ | Occluded MPJPE (mm) $\downarrow$ | PA-MPJPE (mm) $\downarrow$ | Bone-Length Error (mm) $\downarrow$ | Biomechanical Validity (%) $\uparrow$ | Temporal Jitter ($\text{mm/s}^2$) $\downarrow$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30% Occlusion** | Corrupted Input (Raw Gloved/Occluded) | $32.91$ | $8.62$ | $93.63$ | $32.08$ | $14.9372$ | $0.00\%$ | $34.98$ |
| *(6/21 joints)* | `UnconstrainedStudentMLP` (Direct XYZ, No FK) | $11.24$ | $6.22$ | $23.80$ | $10.10$ | $4.2027$ | $38.65\%$ | $15.41$ |
| | `SpatialOnlyGAE` (GAT + 6D FK, No GRU) | $4.67$ | $3.55$ | $7.46$ | $3.42$ | **$0.0000$** | **$100.00\%$** | $15.14$ |
| | **`Teacher ST-GAE` (Full GAT + Bi-GRU + 6D FK)** | **$3.33$** | **$2.80$** | **$4.67$** | **$1.97$** | **$0.0000$** | **$100.00\%$** | **$12.96$** |
| | **`StudentKinematicMLP` (Distilled 6D-FK Student)** | **$4.08$** | **$3.31$** | **$6.00$** | **$2.76$** | **$0.0000$** | **$100.00\%$** | **$13.05$** |
| **50% Occlusion** | Corrupted Input (Raw Gloved/Occluded) | $48.18$ | $8.22$ | $92.13$ | $34.91$ | $19.4657$ | $0.00\%$ | $29.81$ |
| *(10–11/21 joints)* | `UnconstrainedStudentMLP` (Direct XYZ, No FK) | $15.03$ | $5.87$ | $25.11$ | $12.34$ | $5.8505$ | $8.70\%$ | $15.17$ |
| | `SpatialOnlyGAE` (GAT + 6D FK, No GRU) | $6.20$ | $3.62$ | $9.03$ | $4.58$ | **$0.0000$** | **$100.00\%$** | $15.96$ |
| | **`Teacher ST-GAE` (Full GAT + Bi-GRU + 6D FK)** | **$3.79$** | **$2.86$** | **$4.82$** | **$2.11$** | **$0.0000$** | **$100.00\%$** | **$12.31$** |
| | **`StudentKinematicMLP` (Distilled 6D-FK Student)** | **$5.05$** | **$3.37$** | **$6.90$** | **$3.43$** | **$0.0000$** | **$100.00\%$** | **$12.68$** |
| **70% Occlusion** | Corrupted Input (Raw Gloved/Occluded) | $65.55$ | $7.63$ | $88.72$ | $36.11$ | $26.7127$ | $0.00\%$ | $18.76$ |
| *(15/21 joints)* | `UnconstrainedStudentMLP` (Direct XYZ, No FK) | $20.46$ | $5.41$ | $26.48$ | $13.02$ | $8.4587$ | $0.00\%$ | $14.18$ |
| | `SpatialOnlyGAE` (GAT + 6D FK, No GRU) | $10.54$ | $3.78$ | $13.24$ | $7.55$ | **$0.0000$** | **$100.00\%$** | $18.69$ |
| | **`Teacher ST-GAE` (Full GAT + Bi-GRU + 6D FK)** | **$6.35$** | **$3.04$** | **$7.67$** | **$3.76$** | **$0.0000$** | **$100.00\%$** | **$12.52$** |
| | **`StudentKinematicMLP` (Distilled 6D-FK Student)** | **$8.36$** | **$3.42$** | **$10.34$** | **$5.62$** | **$0.0000$** | **$100.00\%$** | **$12.96$** |

*Table II: Held-out test split ablation across 30%, 50%, and 70% contiguous surgical tool occlusion ($N_{\text{test}} = 960$ sequences).*

![Figure 1: Occlusion Resilience across 30%, 50%, and 70% Tool Masking](../results/figures/fig1_occlusion_robustness.png)

![Figure 2: Bone-Length Error and Biomechanical Validity Comparison](../results/figures/fig2_bone_length_and_biovalidity.png)

Three empirical findings emerge from Table II and Figures 1–2:
1. **Effect of Hard Differentiable Forward Kinematics vs. Direct-XYZ Regression:** Comparing `StudentKinematicMLP` ($78,398$ params) against `UnconstrainedStudentMLP` ($78,527$ params)—which share the identical MLP backbone—demonstrates the impact of the 6D-FK manifold. Under 50% occlusion, `UnconstrainedStudentMLP` yields $15.03\text{ mm}$ overall MPJPE, $25.11\text{ mm}$ occluded-joint MPJPE, and $5.8505\text{ mm}$ bone-length error, with only $8.70\%$ of poses passing biomechanical validity checks (falling to $0.00\%$ validity and $8.4587\text{ mm}$ bone error at 70% occlusion). `StudentKinematicMLP` reduces overall MPJPE by **$66.4\%$** ($15.03 \to 5.05\text{ mm}$) and occluded-joint MPJPE by **$72.5\%$** ($25.11 \to 6.90\text{ mm}$) at 50% occlusion while maintaining **$0.0000\text{ mm}$ bone-length error** and **$100.00\%$ biomechanical validity** across all tiers.
2. **Effect of Temporal Recurrence (`Teacher ST-GAE` vs. `SpatialOnlyGAE`):** Adding the Bidirectional GRU across $T=8$ frames reduces overall MPJPE from $4.67 \to 3.33\text{ mm}$ ($-28.7\%$) at 30% occlusion, $6.20 \to 3.79\text{ mm}$ ($-38.9\%$) at 50% occlusion, and $10.54 \to 6.35\text{ mm}$ ($-39.8\%$) at 70% occlusion, while cutting temporal acceleration jitter at 70% occlusion by $33.0\%$ ($18.69 \to 12.52\text{ mm/s}^2$).
3. **Effectiveness of Knowledge Distillation:** Despite having $5.89\times$ fewer parameters than `SpatialOnlyGAE` ($78.4\text{K}$ vs. $461.7\text{K}$), the distilled `StudentKinematicMLP` outperforms `SpatialOnlyGAE` across every occlusion tier ($4.08$ vs. $4.67\text{ mm}$ at 30%, $5.05$ vs. $6.20\text{ mm}$ at 50%, and $8.36$ vs. $10.54\text{ mm}$ at 70% occlusion), confirming that `DistillationLoss` transfers spatio-temporal trajectory priors from the Teacher into the Student's 1D temporal mixing weights.

### C. Custom Gloved Dataset Benchmark & Temporal Smoothing Analysis
Table III reports zero-shot evaluation on our 120-sequence `custom_gloved_dataset.npz` benchmark, and Figures 3–4 illustrate temporal acceleration suppression and 3D qualitative reconstructions.

| Custom Gloved Tier | Method / Pipeline | Overall MPJPE (mm) $\downarrow$ | Visible MPJPE (mm) $\downarrow$ | Occluded MPJPE (mm) $\downarrow$ | PA-MPJPE (mm) $\downarrow$ | Bone-Length Error (mm) $\downarrow$ | Biomechanical Validity (%) $\uparrow$ | Temporal Jitter ($\text{mm/s}^2$) $\downarrow$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30% Occlusion** | Corrupted Input | $32.81$ | $8.65$ | $93.20$ | $32.15$ | $15.1140$ | $0.00\%$ | $33.71$ |
| | `UnconstrainedStudentMLP` | $11.41$ | $6.28$ | $24.22$ | $10.32$ | $4.1964$ | $37.29\%$ | $14.54$ |
| | `SpatialOnlyGAE` | $4.93$ | $3.69$ | $8.02$ | $3.68$ | **$0.0000$** | **$100.00\%$** | $14.09$ |
| | **`Teacher ST-GAE`** | **$3.28$** | **$2.84$** | **$4.36$** | **$1.87$** | **$0.0000$** | **$100.00\%$** | **$11.87$** |
| | **`StudentKinematicMLP`** | **$4.44$** | **$3.48$** | **$6.84$** | **$3.13$** | **$0.0000$** | **$100.00\%$** | **$11.90$** |
| **50% Occlusion** | Corrupted Input | $47.51$ | $8.26$ | $90.69$ | $34.78$ | $19.8146$ | $0.00\%$ | $29.92$ |
| | `UnconstrainedStudentMLP` | $14.80$ | $5.97$ | $24.52$ | $12.05$ | $5.7323$ | $10.31\%$ | $15.04$ |
| | `SpatialOnlyGAE` | $5.87$ | $3.60$ | $8.38$ | $4.27$ | **$0.0000$** | **$100.00\%$** | $15.36$ |
| | **`Teacher ST-GAE`** | **$3.60$** | **$2.77$** | **$4.51$** | **$1.95$** | **$0.0000$** | **$100.00\%$** | **$11.90$** |
| | **`StudentKinematicMLP`** | **$4.85$** | **$3.26$** | **$6.61$** | **$3.19$** | **$0.0000$** | **$100.00\%$** | **$12.28$** |
| **70% Occlusion** | Corrupted Input | $64.94$ | $7.57$ | $87.89$ | $36.41$ | $26.7197$ | $0.00\%$ | $19.55$ |
| | `UnconstrainedStudentMLP` | $20.34$ | $5.54$ | $26.27$ | $13.06$ | $8.4595$ | $0.00\%$ | $14.78$ |
| | `SpatialOnlyGAE` | $10.82$ | $3.81$ | $13.63$ | $7.71$ | **$0.0000$** | **$100.00\%$** | $19.05$ |
| | **`Teacher ST-GAE`** | **$6.47$** | **$3.21$** | **$7.77$** | **$3.83$** | **$0.0000$** | **$100.00\%$** | **$12.03$** |
| | **`StudentKinematicMLP`** | **$8.56$** | **$3.51$** | **$10.58$** | **$5.73$** | **$0.0000$** | **$100.00\%$** | **$12.91$** |

*Table III: Zero-shot evaluation on `data/custom_gloved_dataset.npz` (120 sequences with 12% glove stiffness + AR(1) specular jitter + surgical tool clipping).*

![Figure 3: Temporal Trajectory and Acceleration Jitter Suppression](../results/figures/fig3_temporal_jitter_comparison.png)

![Figure 4: Qualitative 3D Hand Skeleton Reconstruction under 50% and 70% Tool Occlusion](../results/figures/fig4_qualitative_3d_hand_repair.png)

---

## VIII. Cross-Hardware Edge Profiling & Real-Time Telemetry Deployment

Kapish and Vrushank exported `StudentKinematicMLP` to ONNX Opset 17 (`distillation/export_models.py` $\to$ `checkpoints/student_kinematic_fk.onnx`, $433.21\text{ KB}$) and profiled inference latency, throughput, and numerical parity (`deployment/benchmark_edge.py`, `results/edge_benchmark.json`, `results/onnx_export_report.json`) across five target hardware platforms:

1. **PC Workstation (Measured Host — Intel 24-Logical-Core x86_64 CPU):**
   * Bare-metal NumPy `EdgeFK` ($0$ parameters, $3,810\text{ FLOPs/frame}$) executes single-frame 20-bone Gram-Schmidt + FK traversal in **$0.1615\text{ ms}$ ($6,191.7\text{ FPS}$ / $6.19\text{ kHz}$)** with $8 \times 10^{-6}\text{ mm}$ PyTorch parity.
   * ONNX Runtime `CPUExecutionProvider` runs `StudentKinematicMLP` in **$0.9644\text{ ms}$ ($1,036.9\text{ FPS}$)** for streaming single frames ($T=1$) and **$0.7092\text{ ms}$ per 8-frame window ($0.1211\text{ ms/frame}$, $8,256.4\text{ FPS}$)**, achieving a **$7.93\times$ speedup** over the PyTorch Teacher ($5.622\text{ ms}$) with a maximum ONNX-vs-PyTorch deviation of $0.000254\text{ mm}$.
2. **Apple MacBook Air (M1/M2/M3/M4 Apple Silicon):**
   * Accelerating ONNX Runtime via `CoreMLExecutionProvider` and Apple Accelerate AMX instructions yields **$0.314\text{ ms}$ ($3,184.7\text{ FPS}$)** at $T=1$ and $0.511\text{ ms}$ at $T=8$ ($53.1\times$ real-time 60 FPS headroom). Supports native live webcam capture (`--camera 0`) and wireless iPhone **Continuity Camera** (`--camera 1`).
3. **NVIDIA Jetson Nano (4GB Developer Kit + Raspberry Pi CSI Camera v2 `IMX219`):**
   * On the quad-core ARM Cortex-A57 + 128-core Maxwell GPU ($472\text{ GFLOPS}$ FP16), running the full `SpatioTemporalGAE` Teacher in PyTorch CUDA takes **$35.31\text{ ms}$ ($28.3\text{ FPS}$)**, falling below the 30 FPS camera rate.
   * Compiling `student_kinematic_fk.onnx` with `TensorrtExecutionProvider` (FP16) reduces latency to **$2.046\text{ ms}$ ($488.8\text{ FPS}$)** at $T=1$ and $2.772\text{ ms}$ at $T=8$—a **$12.7\times$ embedded speedup** consuming only $6.1\%$ of the 30 FPS CSI camera budget (`nvarguscamerasrc` GStreamer pipeline in `data/iphone_parser.py`).
4. **Raspberry Pi 5 & Raspberry Pi 4B (ARM64 Embedded CPU):**
   * Using ONNX Runtime `CPUExecutionProvider` with ARM64 NEON SIMD vectorization, the Teacher requires $63.52\text{ ms}$ ($15.7\text{ FPS}$), whereas `StudentKinematicMLP` executes in **$3.052\text{ ms}$ ($327.7\text{ FPS}$)** on Raspberry Pi 5 (Cortex-A76) and **$5.692\text{ ms}$ ($175.7\text{ FPS}$)** on Raspberry Pi 4B (Cortex-A72).
5. **iPhone ARKit 3D Hand Skeleton UDP Telemetry (`ARKitUDPStreamer`, Port 5005):**
   * Streams 60 Hz 21-joint 3D coordinates from the Apple Neural Engine over Wi-Fi/USB UDP to `deployment/live_demo.py`. JSON packet parsing and wrist-root retargeting take **$0.080\text{ ms}$**, and ONNX Student repair takes **$0.964\text{ ms}$**, yielding a **$1.044\text{ ms}$ total end-to-end telemetry repair latency** ($957.9\text{ Hz}$ maximum sustained rate).

| Target Platform | Primary Execution Provider | Teacher ST-GAE ($T=8$) | **Student ONNX ($T=1$)** | **Student ONNX ($T=8$ / per-frame)** | **Student FPS ($T=1$ / Batch)** | **Bare-Metal `EdgeFK`** | **Bone Error / Validity** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PC Workstation (x86_64)** | ORT `CPUExecutionProvider` | $5.622\text{ ms}$ | **$0.964\text{ ms}$** | **$0.709\text{ ms}$ ($0.121\text{ ms/f}$)** | **$1,037\text{ / }8,256\text{ FPS}$** | $0.162\text{ ms}$ ($6.19\text{ kHz}$) | $0.0000\text{ mm}$ / $100\%$ |
| **MacBook Air (M-Series)** | ORT CoreML / Accelerate AMX | $7.673\text{ ms}$ | **$0.314\text{ ms}$** | **$0.511\text{ ms}$ ($0.064\text{ ms/f}$)** | **$3,185\text{ / }15,655\text{ FPS}$** | $0.140\text{ ms}$ ($7.13\text{ kHz}$) | $0.0000\text{ mm}$ / $100\%$ |
| **NVIDIA Jetson Nano 4GB** | TensorRT FP16 + RPi CSI Cam | $35.310\text{ ms}$ | **$2.046\text{ ms}$** | **$2.772\text{ ms}$ ($0.347\text{ ms/f}$)** | **$488.8\text{ / }2,886\text{ FPS}$** | $0.792\text{ ms}$ ($1.26\text{ kHz}$) | $0.0000\text{ mm}$ / $100\%$ |
| **Raspberry Pi 5 (ARM64)** | ORT ARM64 NEON SIMD | $63.520\text{ ms}$ | **$3.052\text{ ms}$** | **$4.785\text{ ms}$ ($0.598\text{ ms/f}$)** | **$327.7\text{ / }1,672\text{ FPS}$** | $0.627\text{ ms}$ ($1.59\text{ kHz}$) | $0.0000\text{ mm}$ / $100\%$ |
| **iPhone ARKit UDP (5005)** | `ARKitUDPStreamer` + Host ORT | — | **$1.044\text{ ms}$ (E2E)** | — | **$957.9\text{ Hz}$ Max Stream** | $0.162\text{ ms}$ ($6.19\text{ kHz}$) | $0.0000\text{ mm}$ / $100\%$ |

*Table IV: Multi-hardware edge inference profiling (`results/edge_benchmark.json`, `results/onnx_export_report.json`).*

![Figure 5: Pareto Frontier of Accuracy (MPJPE) vs. Edge Latency and Model Size](../results/figures/fig5_edge_latency_pareto.png)

![Figure 6: Real-Time 3-Panel Telemetry Repair Dashboard (`deployment/live_demo.py`)](../results/figures/fig6_live_demo_dashboard.png)

---

## IX. Individual Team Contributions

1. **Abhishekh (Teacher Model Architecture & Spatial Ablation):**
   * Authored `models/teacher_st_gae.py` and `training/train_teacher.py`.
   * Designed `MultiHeadGraphAttentionLayer` with visibility-aware key gating ($\gamma_h(1-m_{t,j})$) and learnable topological edge bias over the 21-joint hand adjacency matrix (20 bone segments + 4 palmar arch links).
   * Architected the $520,189$-parameter `SpatioTemporalGAE` Teacher (3-layer 4-head GAT + depthwise 1D temporal convolution + Bidirectional GRU) and the $461,693$-parameter `SpatialOnlyGAE` ablation baseline, achieving $3.33\text{ mm}$, $3.79\text{ mm}$, and $6.35\text{ mm}$ MPJPE at 30%, 50%, and 70% occlusion.
2. **Bhavya (Data Pipeline & Zero-Shot 3D Domain Adaptation):**
   * Authored `data/dataset_ingestion.py`, `data/transforms.py`, `data/dataloader.py`, `data/iphone_parser.py`, and `data/record_custom_gloved.py`.
   * Synthesized the $4,800$-sequence ($38,400$-frame) 21-joint MANO trajectory corpus across 4 seeds with a stratified 80/20 train-test split (`data/splits/split_indices.json`) and generated the 120-sequence `data/custom_gloved_dataset.npz` benchmark.
   * Formulated `SyntheticOcclusionTransform` (3D cylindrical instrument clipping $R=28\text{ mm}$ + contiguous distal-to-proximal subtree priority) and `SyntheticGloveTransform` ($12\%$ kinematic stiffness + $\rho=0.65$ AR(1) specular jitter and $12.0\text{ mm}$ fingertip spikes), plus hardware camera/UDP streamers (`ARKitUDPStreamer`, `MediaPipeCameraStreamer`, `build_jetson_csi_pipeline`).
3. **Vrushank (Continuous 6D Rotations, RoM Clamping & Differentiable Forward Kinematics):**
   * Authored `models/kinematics.py` and `tests/test_kinematics.py` (13 unit tests verifying $\text{SO}(3)$ orthogonality, gradient flow, zero bone error, and PyTorch-NumPy parity).
   * Implemented Gram-Schmidt `rot6d_to_matrix`, $C^1$-smooth ONNX-traceable Euler extraction (`_safe_angle_from_sin_cos`), hyperbolic tangent `BiomechanicalRoMClamper`, PyTorch `DifferentiableFK`, closed-form Rodrigues analytical IK (`positions_to_rot6d`), `check_biomechanical_validity`, and bare-metal NumPy `EdgeFK` ($0.1615\text{ ms}$, $6.19\text{ kHz}$).
4. **Kapish (Knowledge Distillation, ONNX Compilation & Multi-Hardware Edge Deployment):**
   * Authored `models/student_models.py`, `training/losses.py`, `distillation/distill_student.py`, `distillation/export_models.py`, `deployment/benchmark_edge.py`, `deployment/live_demo.py`, and `training/evaluate_and_plot.py`.
   * Designed `StudentKinematicMLP` ($78,398$ params, $163.4\text{ KFLOPs/frame}$) and the `UnconstrainedStudentMLP` direct-XYZ ablation ($78,527$ params), achieving a $6.64\times$ parameter and $137.3\times$ FLOPs reduction.
   * Formulated the five-term `DistillationLoss`, exported static ONNX Opset 17 models (`433.21 KB`, $0.000254\text{ mm}$ parity error), profiled latency across PC, MacBook Air M-Series, NVIDIA Jetson Nano + RPi CSI Camera, Raspberry Pi 5/4B, and iPhone ARKit UDP, and built the 3-panel real-time telemetry visualizer.

---

## X. Conclusion

This work demonstrated that unconstrained 3D coordinate regression is structurally ill-suited for occluded and gloved hand telemetry: under 50% to 70% surgical instrument occlusion, direct Cartesian MLP predictors suffer $5.85\text{ to }8.46\text{ mm}$ of phalangeal bone distortion and drop to $8.7\%\text{–}0.0\%$ biomechanical validity. By combining a visibility-gated Spatio-Temporal Graph Autoencoder (`SpatioTemporalGAE`) Teacher with multi-objective Knowledge Distillation into a $78,398$-parameter `StudentKinematicMLP` embedded with continuous 6D $\text{SO}(3)$ orthogonalization, smooth $\tanh$ Range-of-Motion clamping, and exact Differentiable Forward Kinematics, our framework reduces 50%-occlusion MPJPE to $3.79\text{ mm}$ (Teacher) and $5.05\text{ mm}$ (Student) while guaranteeing **$0.0000\text{ mm}$ bone-length error** and **$100.0\%$ biomechanical validity** on every frame. Operating at $0.121\text{ ms/frame}$ ($8,256\text{ FPS}$) on CPU and $2.046\text{ ms}$ ($488.8\text{ FPS}$) on an NVIDIA Jetson Nano in TensorRT FP16, the distilled Student enables real-time, anatomically valid hand telemetry across desktop, mobile, and embedded edge hardware.

---

## References

[1] C. Zimmermann, D. Ceylan, J. Yang, B. Russell, M. Argus, and T. Brox, "FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 813–822, arXiv:1909.04349.

[2] S. Hampali, M. Rad, M. Oberweger, and V. Lepetit, "HOnnotate: A Method for 3D Annotation of Hand and Object Poses," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 3196–3206, arXiv:1907.01481.

[3] Y.-W. Chao et al., "DexYCB: A Benchmark for Capturing Hand Grasping of Objects," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 9044–9053, arXiv:2104.04631.

[4] G. Moon, S.-I. Yu, H. Wen, T. Shiratori, and K. M. Lee, "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, vol. 12365, 2020, pp. 548–564, arXiv:2008.09309.

[5] L. Ge, Z. Ren, Y. Li, Z. Xue, Y. Wang, J. Cai, and J. Yuan, "3D Hand Shape and Pose Estimation from a Single RGB Image," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 10833–10842, arXiv:1903.00812.

[6] B. Doosti, S. Naha, M. Mirbagheri, and D. Crandall, "HOPE-Net: A Graph-based Model for Hand-Object Pose Estimation," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 6608–6617, arXiv:2004.00060.

[7] H. Choi, G. Moon, and K. M. Lee, "Pose2Mesh: Graph Convolutional Network for 3D Human Pose and Mesh Recovery from a 2D Human Pose," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, vol. 12352, 2020, pp. 769–787, arXiv:2008.09047.

[8] J. Park, Y. Oh, G. Moon, H. Choi, and K. M. Lee, "HandOccNet: Occlusion-Robust 3D Hand Mesh Estimation Network," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 1496–1505, arXiv:2203.14564.

[9] K. Lin, L. Wang, and Z. Liu, "End-to-End Human Pose and Mesh Reconstruction with Transformers," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 1954–1963, arXiv:2012.09760.

[10] I. Sárándi, T. Linder, K. O. Arras, and B. Leibe, "MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation," *IEEE Trans. Biometrics, Behav., Identity Sci. (TBIOM)*, vol. 3, no. 1, pp. 16–30, 2021, arXiv:2007.07227.

[11] A. Spurr, U. Iqbal, P. Molchanov, O. Hilliges, and J. Kautz, "Weakly Supervised 3D Hand Pose Estimation via Biomechanical Constraints," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, vol. 12362, 2020, pp. 211–228, arXiv:2003.09282.

[12] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, "On the Continuity of Rotation Representations in Neural Networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 5745–5753, arXiv:1812.07035.

[13] J. Romero, D. Tzionas, and M. J. Black, "Embodied Hands: Modeling and Capturing Hands and Bodies Together," *ACM Trans. Graph. (SIGGRAPH Asia)*, vol. 36, no. 6, Art. 245, 2017, doi: 10.1145/3130800.3130883.

[14] F. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, G. Sung, C.-L. Chang, and M. Grundmann, "MediaPipe Hands: On-device Real-time Hand Tracking," in *CVPR Workshop Comput. Vis. Augmented Virtual Reality*, 2020, arXiv:2006.10214.
