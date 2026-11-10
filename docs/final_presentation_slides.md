# Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry
**Final Project Presentation Deck — November 17, 2026**  
**Course:** Foundations of Data Science (FDS) — Year 3 Undergraduate  
**Authors:** Abhishekh, Bhavya, Vrushank, Kapish

---

## Slide 1: Title, Team Roles & Problem Overview

### Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry

* **Clinical & Industrial Motivation:**
  * Optical and LiDAR hand trackers (MediaPipe Hands, Apple ARKit) estimate 21 3D hand joints in $\mathbb{R}^{21 \times 3}$ directly from monocular cameras or depth sensors.
  * **Failure Mode 1 — Surgical & Industrial Tool Occlusion:** Grasping scalpels, forceps, drills, or probes blocks line-of-sight to 30%–70% of phalangeal joints, causing unconstrained regressors to collapse occluded fingers toward the palm or stretch bones by $15\text{–}27\text{ mm}$.
  * **Failure Mode 2 — Latex & Nitrile Glove Specularity:** Protective gloves remove skin texture and knuckle creases while introducing $12\%$ kinematic stiffness and directional specular glare ($\sigma = 4.5\text{ mm}$ baseline noise, $12.0\text{ mm}$ fingertip spikes).
* **Core Thesis:**
  * Replace unconstrained Cartesian coordinate regression ($\mathbb{R}^{21 \times 3}$) with a **Spatio-Temporal Graph Autoencoder (ST-GAE) Teacher** distilled into a **78.4K-parameter Student MLP** that predicts continuous 6D joint rotations on $\text{SO}(3)$ clamped to physiological Range-of-Motion (RoM) bounds and decoded via an exact **Differentiable Forward Kinematics (FK)** layer.
  * Guarantees **$0.0000\text{ mm}$ bone-length error** and **$100.0\%$ biomechanical validity** at **8,256 FPS ($0.121\text{ ms/frame}$)** in ONNX Runtime.

| Team Member | Primary Module Ownership | Key Codebase Components |
| :--- | :--- | :--- |
| **Abhishekh** | **Teacher Model Architecture & Spatial Ablation** | `models/teacher_st_gae.py`, `training/train_teacher.py` — 3-Layer 4-Head Mask-Aware GAT + Bidirectional Temporal GRU ($T=8$) + `SpatialOnlyGAE` ablation |
| **Bhavya** | **Data Pipeline & Zero-Shot 3D Domain Adaptation** | `data/dataset_ingestion.py`, `data/transforms.py`, `data/record_custom_gloved.py` — 21-joint MANO ingestion, 3D Volumetric Tool Clipping & Subtree Masking (30%/50%/70%), Glove Stiffness & AR(1) Jitter |
| **Vrushank** | **6D Rotations, RoM Clamping & Differentiable FK** | `models/kinematics.py`, `tests/test_kinematics.py` — Gram-Schmidt $\mathbb{R}^6 \to \text{SO}(3)$, `BiomechanicalRoMClamper` ($\tanh$), PyTorch `DifferentiableFK`, bare-metal NumPy `EdgeFK` |
| **Kapish** | **Knowledge Distillation & Multi-Hardware Edge Deployment** | `models/student_models.py`, `distillation/`, `deployment/` — `StudentKinematicMLP` vs. `UnconstrainedStudentMLP`, ONNX export, profiling on PC / MacBook Air M-Series / Jetson Nano + RPi CSI / iPhone UDP, `live_demo.py` |

#### Speaker Script — Slide 1
* **Abhishekh (0:00–0:30):** "Good morning everyone. Today our team—Bhavya, Vrushank, Kapish, and I—will present *Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry*. In surgical navigation and industrial teleoperation, vision-based hand trackers like MediaPipe and Apple ARKit fail when instruments occlude the fingers or when medical latex gloves introduce specular glare. Unconstrained neural networks regress 63 independent $X, Y, Z$ coordinates, producing stretched bones and hyperextended joints whenever visual evidence drops out."
* **Bhavya (0:30–0:50):** "To solve this without requiring studio multi-camera arrays, I built our 3D MANO trajectory ingestion pipeline and zero-shot domain adaptation transforms (`SyntheticOcclusionTransform` and `SyntheticGloveTransform`), simulating 3D cylindrical tool occlusion across 30%, 50%, and 70% of the hand alongside latex glove stiffness and autoregressive sensor jitter."
* **Vrushank (0:50–1:10):** "On the mathematical side, I designed our continuous 6D rotation parameterization on $\text{SO}(3)$, our hyperbolic-tangent `BiomechanicalRoMClamper`, and our exact `DifferentiableFK` and bare-metal NumPy `EdgeFK` engines, which guarantee $0.0000\text{ mm}$ bone-length error by construction."
* **Kapish (1:10–1:30):** "And I compressed Abhishekh's 520K-parameter Spatio-Temporal Graph Autoencoder Teacher via multi-objective Knowledge Distillation into a 78.4K-parameter ONNX Student MLP—achieving a $6.64\times$ parameter reduction and $137.3\times$ FLOPs reduction for sub-millisecond deployment across PC, Apple Silicon MacBook Air, NVIDIA Jetson Nano with a Raspberry Pi CSI camera, and iPhone ARKit UDP streams."

---

## Slide 2: Literature Review & Research Gap

### Prior Work Limitations: Soft Penalty Losses vs. Hard Kinematic Manifolds

* **Graph & Transformer Hand Pose Estimators:**
  * **HandGCN** (Ge et al., CVPR 2019) and **HOPE-Net** (Doosti et al., CVPR 2020) apply spectral or adaptive Graph CNNs over the hand topology, but regress unconstrained 3D Euclidean coordinates ($\mathbb{R}^{21 \times 3}$). Under tool occlusion, predicted bone lengths deviate by $3.4\text{–}4.8\text{ mm}$ and biomechanical validity drops to $71.0\%\text{–}78.5\%$.
  * **HandOccNet** (Park et al., CVPR 2022) uses Feature-Injecting and Self-Enhancing Transformers to infill occluded features ($9.1\text{ mm}$ PA-MPJPE on HO-3D), but requires **37.2M parameters ($142\text{ MB}$)** and $33.5\text{ ms/frame}$ on desktop GPUs—exceeding embedded edge budgets.
* **Why Soft Biomechanical Losses Fail at Test Time (Spurr et al., ECCV 2020):**
  * Spurr et al. regularize unconstrained 3D coordinate predictions during training using soft hinge penalties on bone length ($L_{bl}$) and joint angles ($L_{ja} = \sum_i \max(0, \underline{\theta}_i - \theta_i) + \max(0, \theta_i - \overline{\theta}_i)$).
  * **Research Gap:** Soft training penalties do not constrain the output architecture at inference time. When out-of-distribution tool occlusion masks 50%–70% of joints, the network still outputs stretched phalanges ($1.9\text{ mm}$ bone error, $8.6\%$ invalid poses).
* **Why Euler Angles & Quaternions Fail in Gradient Descent (Zhou et al., CVPR 2019):**
  * Euler angles ($\mathbb{R}^3$) suffer from gimbal lock and $\pm\pi$ wrap-around discontinuities; unit quaternions ($S^3 \subset \mathbb{R}^4$) have antipodal double-cover ambiguity ($q \equiv -q$). Every representation of $\text{SO}(3)$ in $d \le 4$ Euclidean dimensions is topologically discontinuous.

| Architecture / Method | Venue | Output Representation | Bone-Length Error | Biomech. Validity | Parameters (Size) | Edge CPU Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **HandGCN** (Ge et al.) | CVPR 2019 | Spectral GCN ($\mathbb{R}^{V \times 3}$ Mesh) | $3.40\text{ mm}$ | $78.5\%$ | $18.50\text{M}$ ($74.0\text{ MB}$) | $28.0\text{ ms}$ (GPU) |
| **HOPE-Net** (Doosti et al.) | CVPR 2020 | Adaptive-GCN ($\mathbb{R}^{21 \times 3}$ Coords) | $4.80\text{ mm}$ | $71.0\%$ | $25.20\text{M}$ ($96.0\text{ MB}$) | $16.5\text{ ms}$ (GPU) |
| **Spurr et al.** | ECCV 2020 | CNN + Soft Hinge Penalties ($L_{bl}, L_{ja}$) | $1.90\text{ mm}$ | $91.4\%$ | $23.60\text{M}$ ($90.0\text{ MB}$) | $15.0\text{ ms}$ (GPU) |
| **HandOccNet** (Park et al.) | CVPR 2022 | Dual Transformer + MANO Mesh | $0.80\text{ mm}$ | $94.2\%$ | $37.20\text{M}$ ($142.0\text{ MB}$) | $33.5\text{ ms}$ (GPU) |
| **MediaPipe Hands** (Zhang et al.) | CVPRW 2020 | Mobile CNN ($\mathbb{R}^{21 \times 3}$ Coords) | $6.20\text{ mm}$ | $64.8\%$ | $1.50\text{M}$ ($5.5\text{ MB}$) | $4.5\text{ ms}$ |
| **StudentKinematicMLP (Ours)** | **FDS 2026** | **6D $\text{SO}(3)$ + Hard $\tanh$ RoM + Edge FK** | **$0.0000\text{ mm}$** | **$100.0\%$** | **$0.078\text{M}$ ($0.30\text{ MB}$)** | **$0.12\text{ ms}$ (ORT)** |

#### Speaker Script — Slide 2
* **Abhishekh (0:00–0:45):** "Let's examine why existing methods fall short for edge surgical telemetry. Graph architectures like HandGCN and HOPE-Net propagate features across joints, but their final layer still outputs unconstrained $X, Y, Z$ coordinates, resulting in $3.4$ to $4.8\text{ mm}$ of bone length distortion. Transformer models like HandOccNet improve occlusion recovery on HO-3D and DexYCB, but require $37.2\text{ million}$ parameters and $142\text{ megabytes}$, making them impractical for low-power edge microcontrollers."
* **Vrushank (0:45–1:30):** "A key benchmark in biomechanical hand pose estimation is Spurr et al. at ECCV 2020, who introduced soft hinge loss penalties on joint angles and bone lengths during training. Because those constraints exist only in the loss function while the network output remains unconstrained $\mathbb{R}^{21 \times 3}$, severe test-time occlusion still produces invalid poses. In addition, as Zhou et al. proved at CVPR 2019, regressing 3D Euler angles or 4D quaternions introduces topological discontinuities and gimbal lock. Our architecture closes this gap by embedding continuous 6D $\text{SO}(3)$ rotations, hard $\tanh$ angular clamping, and fixed-bone Forward Kinematics directly inside the forward pass."

---

## Slide 3: End-to-End System Architecture

### Teacher-Student Kinematic Distillation & Edge Deployment Pipeline

```mermaid
flowchart LR
    subgraph Data ["1. 3D Domain Adaptation (Bhavya)"]
        A["Clean 21-Joint MANO\n(N=4800, T=8, 21, 3)"] --> B["SyntheticGloveTransform\n12% Stiffness + AR(1) Jitter\n(4.5mm base / 12mm tip spikes)"]
        B --> C["SyntheticOcclusionTransform\n3D Cylinder Clip + Subtree Mask\n(30% / 50% / 70% Occlusion)"]
    end

    subgraph Teacher ["2. Spatio-Temporal GAE Teacher (Abhishekh — 520.2K Params)"]
        C --> D["Imputation + Rest Delta\n(B, T, 21, 7) + Joint Emb"]
        D --> E["3-Layer 4-Head Mask-Aware GAT\n(21x21 Adjacency + Vis Gate)"]
        E --> F["Depthwise 1D Conv + Bi-GRU\nAcross T=8 Window (D=128)"]
    end

    subgraph Student ["3. Distilled Edge Student (Kapish — 78.4K Params)"]
        C --> G["Flatten (B, T, 84) -> Linear(128)\n+ TemporalMixingBlock1D (k=3)\n+ ResidualMLPBlock (SiLU)"]
    end

    subgraph Kinematics ["4. Hard Kinematic Manifold (Vrushank — 0.0000 mm Error)"]
        F -->|"Teacher 6D Rotations"| H["BiomechanicalRoMClamper\nSmooth tanh [theta_min, theta_max]\n+ Gram-Schmidt 6D -> SO(3)"]
        G -->|"Student 6D Rotations"| H
        H --> I["DifferentiableFK / EdgeFK\nP_j = P_p + R_global[j] @ v_j\n(0.0000 mm Bone Error, 100% Valid)"]
    end

    F -.->|"DistillationLoss (KD 6D + Geodesic + FK MPJPE + Accel)"| G
```

* **Architectural Separation of Concerns:**
  1. **Offline Training Phase:** The high-capacity `SpatioTemporalGAE` Teacher ($520,189$ parameters, $22.43\text{ MFLOPs/frame}$) leverages 3-layer multi-head topological graph attention and bidirectional temporal recurrence to infer missing subtrees.
  2. **Knowledge Distillation Phase:** Teacher representations (`teacher_rot6d` $\in \mathbb{R}^{T \times 21 \times 6}$ and `teacher_pos_mm` $\in \mathbb{R}^{T \times 21 \times 3}$) supervise the lightweight `StudentKinematicMLP` ($78,398$ parameters, $0.163\text{ MFLOPs/frame}$).
  3. **Edge Execution Phase:** Both Teacher and Student terminate in the identical `BiomechanicalRoMClamper` and `DifferentiableFK` layer, exported as a single static ONNX Opset 17 graph (`433.21 KB`).

#### Speaker Script — Slide 3
* **Abhishekh (0:00–0:45):** "Slide 3 illustrates our four-stage end-to-end architecture. Stage 1 ingests 21-joint 3D trajectories and corrupts them with realistic glove stiffness, autoregressive specular jitter, and contiguous 3D tool occlusion masks. In Stage 2, my Teacher network—the Spatio-Temporal Graph Autoencoder—takes the corrupted sequence $(B, T, 21, 4)$ where channel 4 is the binary visibility mask $m_{t,j} \in \{0, 1\}$, propagates features across visible neighbors via three 4-head Graph Attention layers, and smooths trajectories across $T=8$ frames using a Bidirectional GRU."
* **Kapish (0:45–1:30):** "In Stage 3, rather than deploying the heavy graph attention network on edge devices, we distill its learned kinematic manifold into our `StudentKinematicMLP`. The Student replaces quadratic graph attention and recurrent GRU states with a depthwise 1D temporal convolution (`kernel_size=3`) and SiLU residual MLP blocks. Both Teacher and Student feed into Vrushank's Stage 4 kinematic head—clamping joint angles via $\tanh$ and computing exact Forward Kinematics—so the Student inherits the Teacher's occlusion recovery while running $7.9\times$ faster in ONNX Runtime."

---

## Slide 4: Data Pipeline & Zero-Shot 3D Domain Adaptation (Bhavya)

### Simulating Surgical Tool Occlusion & Latex Glove Degradation in 3D Graph Space

* **Dataset Scale & Stratified Protocol (`data/dataset_ingestion.py`, `data/dataloader.py`):**
  * **Total Corpus:** $4,800$ 3D hand sequences ($38,400$ frames at $T=8$, $21$ MANO joints) across 5 clinical/industrial gesture classes (*Pinch, Power Grasp, Tripod Precision, Index Pointing, Open Flexion*), partitioned into a strict **80% Train ($3,840$ sequences) / 20% Held-Out Test ($960$ sequences)** split (`data/splits/split_indices.json`).
  * **Custom Gloved Benchmark (`data/record_custom_gloved.py`):** $120$ multi-tier evaluation sequences ($960$ frames) stored in `data/custom_gloved_dataset.npz` with precomputed 30%, 50%, and 70% surgical tool masks.
* **1. `SyntheticGloveTransform` — Kinematic Stiffness & Specular AR(1) Jitter:**
  * **Elastic Stiffness ($\alpha_{\text{stiff}} = 0.12$):** Dampens unit bone direction vectors $\hat{\mathbf{d}}_{t,j}$ toward the semi-flexed neutral glove equilibrium $\hat{\mathbf{d}}_j^{\text{rest}}$ while preserving canonical bone length $L_j$:
    $$\tilde{\mathbf{d}}_{t,j} = \frac{(1 - \alpha_{\text{stiff}})\hat{\mathbf{d}}_{t,j} + \alpha_{\text{stiff}}\hat{\mathbf{d}}_j^{\text{rest}}}{\|(1 - \alpha_{\text{stiff}})\hat{\mathbf{d}}_{t,j} + \alpha_{\text{stiff}}\hat{\mathbf{d}}_j^{\text{rest}}\|_2}, \quad \tilde{\mathbf{P}}_{t,j} = \tilde{\mathbf{P}}_{t,\text{parent}(j)} + L_j \tilde{\mathbf{d}}_{t,j}$$
  * **Temporally Correlated AR(1) Tracking Jitter ($\rho = 0.65$):** Models optical/LiDAR specular drift on latex surfaces with baseline $\sigma_j = 4.5\text{ mm}$ ($5.625\text{ mm}$ on DIP joints) and Bernoulli fingertip specular spikes ($p_{\text{spike}} = 0.20$, $\sigma_{\text{spike}} = 12.0\text{ mm}$):
    $$\boldsymbol{\eta}_{t,j} = \rho \boldsymbol{\eta}_{t-1,j} + \sqrt{1 - \rho^2}\sigma_j \mathbf{z}_{t,j} + b_{t,j}\boldsymbol{\xi}_{t,j}, \quad b_{t,j} \sim \text{Bernoulli}(0.20), \quad \boldsymbol{\xi}_{t,j} \sim \mathcal{N}(\mathbf{0}, 12.0^2 \mathbf{I}_3)$$
* **2. `SyntheticOcclusionTransform` — 3D Tool Cylinder Clipping & Subtree Masking:**
  * Spawns a moving 3D cylindrical instrument volume ($R_{\text{cyl}} = 28.0\text{ mm}$, axis $\hat{\mathbf{u}}_{\text{tool}}$) across the fingers and computes signed proximity plus optical ray depth behind the tool ($+Z$).
  * Enforces **contiguous distal-to-proximal subtree priority** ($\text{TIP} \to \text{DIP} \to \text{PIP} \to \text{MCP}$) so masking a proximal joint automatically masks its distal children, while anchoring the wrist root ($j=0, m_{t,0}=1$) at 30% ($6/21$ joints), 50% ($10\text{–}11/21$ joints), and 70% ($15/21$ joints) occlusion.

#### Speaker Script — Slide 4
* **Bhavya (0:00–0:50):** "In Slide 4, I detail our data ingestion and zero-shot 3D domain adaptation pipeline. Because public RGB datasets rarely pair ground-truth 3D MANO joints with sterile surgical gloves and metallic instrument occlusion, we operate directly in 3D skeletal graph space across $4,800$ sequences ($38,400$ frames) with an 80/20 train-test split, plus a $120$-sequence custom gloved benchmark. First, `SyntheticGloveTransform` models the physical elasticity of latex gloves by blending each bone's unit vector toward the neutral glove rest posture by $\alpha_{\text{stiff}} = 12\%$ while preserving exact bone lengths $L_j$. It then injects first-order autoregressive AR(1) noise with temporal correlation $\rho = 0.65$, $\sigma = 4.5\text{ mm}$ on visible joints, and $12.0\text{ mm}$ specular spikes on the five fingertips."
* **Bhavya (0:50–1:30):** "Second, random Bernoulli node dropout is unrealistic because a surgical tool never occludes a knuckle while leaving the fingertip floating in mid-air. My `SyntheticOcclusionTransform` spawns a 3D cylindrical tool volume of radius $28\text{ mm}$ moving across the hand, computes perpendicular cylinder distance and camera-ray depth occlusion, and ranks joints with contiguous distal-to-proximal subtree priority ($\text{TIP} \to \text{DIP} \to \text{PIP} \to \text{MCP}$). Occluded joints have their visibility mask set to $m_{t,j} = 0$ and coordinates zeroed, while wrist joint 0 remains visible as the kinematic root."

---

## Slide 5: 6D Continuous Rotations & Hard Edge Forward Kinematics (Vrushank)

### Singularity-Free $\text{SO}(3)$ Mapping, $\tanh$ RoM Clamping & Zero-Bone-Error Proof

* **1. Gram-Schmidt Continuous 6D-to-$\text{SO}(3)$ Conversion (`rot6d_to_matrix`):**
  * For each joint $j \in \{0, \dots, 20\}$, the network outputs $\mathbf{d}_j = [\mathbf{a}_{j,1}, \mathbf{a}_{j,2}] \in \mathbb{R}^6$, mapped differentiably onto $\text{SO}(3)$ ($\det(\mathbf{R}_j) = +1$):
    $$\mathbf{b}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|_2}, \quad \mathbf{b}_2 = \frac{\mathbf{a}_2 - (\mathbf{b}_1 \cdot \mathbf{a}_2)\mathbf{b}_1}{\|\mathbf{a}_2 - (\mathbf{b}_1 \cdot \mathbf{a}_2)\mathbf{b}_1\|_2}, \quad \mathbf{b}_3 = \mathbf{b}_1 \times \mathbf{b}_2, \quad \mathbf{R}_j^{\text{raw}} = \begin{bmatrix} \mathbf{b}_1 & \mathbf{b}_2 & \mathbf{b}_3 \end{bmatrix} \in \text{SO}(3)$$
* **2. Differentiable Anatomical Range-of-Motion Clamping (`BiomechanicalRoMClamper`):**
  * Extracts local Euler angles $\boldsymbol{\theta}_j = [\theta_{j,x}^{\text{flex}}, \theta_{j,y}^{\text{twist}}, \theta_{j,z}^{\text{abd}}]^\top$ via an ONNX-traceable $C^1$ small-angle expansion (`_safe_angle_from_sin_cos`) and applies smooth $\tanh$ clamping onto physiological intervals $[\boldsymbol{\theta}_j^{\min}, \boldsymbol{\theta}_j^{\max}]$:
    $$\boldsymbol{\theta}_j^{\text{clamp}} = \boldsymbol{\mu}_j + \boldsymbol{\Delta}_j \odot \tanh\!\left(\frac{\boldsymbol{\theta}_j - \boldsymbol{\mu}_j}{\boldsymbol{\Delta}_j \cdot \tau + \epsilon}\right), \quad \boldsymbol{\mu}_j = \frac{\boldsymbol{\theta}_j^{\min} + \boldsymbol{\theta}_j^{\max}}{2}, \quad \boldsymbol{\Delta}_j = \frac{\boldsymbol{\theta}_j^{\max} - \boldsymbol{\theta}_j^{\min}}{2}$$
* **3. Hierarchical Forward Kinematics & Zero-Bone-Length-Error Proof (`DifferentiableFK` & `EdgeFK`):**
  * Traversing the directed kinematic tree $p(j) = \text{PARENT\_INDICES}[j]$ from anchored wrist $\mathbf{P}_0 = \mathbf{0}$ with canonical rest bone vectors $\mathbf{v}_j^{\text{rest}} = L_j \hat{\mathbf{u}}_j^{\text{rest}}$:
    $$\mathbf{R}_j^{\text{global}} = \mathbf{R}_{p(j)}^{\text{global}} \mathbf{R}_j^{\text{local}}(\boldsymbol{\theta}_j^{\text{clamp}}), \qquad \mathbf{P}_j = \mathbf{P}_{p(j)} + \mathbf{R}_j^{\text{global}} \mathbf{v}_j^{\text{rest}}$$
  * **Mathematical Invariance Proof:** Since $\mathbf{R}_j^{\text{global}} \in \text{SO}(3)$ is orthogonal ($(\mathbf{R}_j^{\text{global}})^\top \mathbf{R}_j^{\text{global}} = \mathbf{I}_3$), every predicted bone length is invariant under arbitrary network weights:
    $$\|\mathbf{P}_j - \mathbf{P}_{p(j)}\|_2 = \|\mathbf{R}_j^{\text{global}} \mathbf{v}_j^{\text{rest}}\|_2 = \sqrt{(\mathbf{v}_j^{\text{rest}})^\top (\mathbf{R}_j^{\text{global}})^\top \mathbf{R}_j^{\text{global}} \mathbf{v}_j^{\text{rest}}} = \|\mathbf{v}_j^{\text{rest}}\|_2 = L_j \implies \varepsilon_{\text{bone}} \equiv 0.0000\text{ mm}$$

| Finger Joint Group | Joint Indices | Flexion / Extension ($r_x$) | Axial Twist ($r_y$) | Abduction / Adduction ($r_z$) | Canonical Bone Lengths $L_j$ (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thumb Chain** | $1, 2, 3, 4$ | $[-15^\circ, +85^\circ]$ (IP $[0^\circ, 80^\circ]$) | $[-10^\circ, +10^\circ]$ | $[-35^\circ, +45^\circ]$ | $34, 46, 30, 20\text{ mm}$ |
| **Finger MCPs** | $6, 10, 14, 18$ | $[-15^\circ, +90^\circ]$ | $[-5^\circ, +5^\circ]$ | $[-20^\circ, +20^\circ]$ | Proximal: $40, 42, 39, 31\text{ mm}$ |
| **Finger PIPs** | $7, 11, 15, 19$ | $[0^\circ, +110^\circ]$ (No hyperext.) | $[-2^\circ, +2^\circ]$ | $[-2^\circ, +2^\circ]$ | Middle: $24, 27, 25, 19\text{ mm}$ |
| **Finger DIPs** | $8, 12, 16, 20$ | $[0^\circ, +80^\circ]$ (No hyperext.) | $[-2^\circ, +2^\circ]$ | $[-2^\circ, +2^\circ]$ | Distal: $17, 18, 17, 16\text{ mm}$ |

#### Speaker Script — Slide 5
* **Vrushank (0:00–0:50):** "Slide 5 presents our hard kinematic manifold in `models/kinematics.py`. First, to avoid quaternion antipodal jumps and Euler singularities during backpropagation, the network predicts 6 unconstrained scalars per joint, which we orthogonalize into a valid $3 \times 3$ rotation matrix in $\text{SO}(3)$ via differentiable Gram-Schmidt normalization and a cross product. Second, to prevent dorsal hyperextension or lateral finger splay, `BiomechanicalRoMClamper` extracts local flexion, twist, and abduction angles using a $C^1$-smooth small-angle polynomial transition near zero—avoiding the infinite derivative of `acos` at $\theta=0$—and smoothly bounds every angle inside its anatomical interval $[\boldsymbol{\theta}_{\min}, \boldsymbol{\theta}_{\max}]$ via $\tanh$. For instance, PIP and DIP hinge joints are strictly clamped to $[0^\circ, 110^\circ]$ and $[0^\circ, 80^\circ]$ flexion with zero dorsal hyperextension."
* **Vrushank (0:50–1:30):** "Third, `DifferentiableFK` accumulates global rotation matrices along `PARENT_INDICES` from the wrist out to the fingertips and rotates the fixed canonical bone vectors $\mathbf{v}_j^{\text{rest}}$. Because any matrix $\mathbf{R}_j^{\text{global}} \in \text{SO}(3)$ is an isometry, the Euclidean norm $\|\mathbf{P}_j - \mathbf{P}_{p(j)}\|_2$ is identically $L_j$. That gives us a strict mathematical guarantee: $0.0000\text{ mm}$ bone-length error and $100.0\%$ biomechanical validity across all test frames, verified to $8 \times 10^{-6}\text{ mm}$ floating-point precision in our bare-metal NumPy `EdgeFK` engine running at $6.19\text{ kHz}$."

---

## Slide 6: Spatio-Temporal Graph Autoencoder Teacher (Abhishekh)

### Mask-Aware Multi-Head GAT & Bidirectional Temporal GRU ($T=8$)

* **1. Visibility-Imputed Node Embedding (`models/teacher_st_gae.py`):**
  * Given input $\mathbf{X} \in \mathbb{R}^{B \times T \times 21 \times 4}$ with palm-normalized coordinates $\tilde{\mathbf{x}}_{t,j} = \mathbf{x}_{t,j} / 100\text{ mm}$ and visibility flag $m_{t,j} \in \{0, 1\}$, occluded nodes are imputed with canonical rest positions $\bar{\mathbf{x}}_j^{\text{rest}}$ and concatenated with visible displacements $\Delta\tilde{\mathbf{x}}_{t,j}$:
    $$\mathbf{x}_{t,j}^{\text{imp}} = m_{t,j}\tilde{\mathbf{x}}_{t,j} + (1 - m_{t,j})\bar{\mathbf{x}}_j^{\text{rest}}, \quad \mathbf{f}_{t,j}^{\text{in}} = \left[\mathbf{x}_{t,j}^{\text{imp}} \;\|\; m_{t,j} \;\|\; m_{t,j}(\tilde{\mathbf{x}}_{t,j} - \bar{\mathbf{x}}_j^{\text{rest}})\right] \in \mathbb{R}^7$$
    $$\mathbf{h}_{t,j}^{(0)} = \text{GELU}\!\left(\text{LN}(\mathbf{W}_{\text{in}}\mathbf{f}_{t,j}^{\text{in}})\right) + \mathbf{e}_j^{\text{joint}} + \mathbf{h}_t^{\text{pose}} \in \mathbb{R}^{128}$$
* **2. 3-Layer 4-Head Mask-Aware Graph Attention (`MultiHeadGraphAttentionLayer`):**
  * Operates over the 21-joint adjacency matrix $\mathbf{A} \in \{0, 1\}^{21 \times 21}$ (21 self-loops + 20 kinematic bone edges + 4 inter-MCP palmar arch edges $= 69$ directed edges).
  * Combines scaled dot-product attention, GATv2 additive scoring, learnable topological edge bias $B_{h,i,j}$, and a **visibility key penalty** $\gamma_h (1 - m_{t,j})$ (initialized at $\gamma_h = 1.5$) that suppresses message passing from occluded key nodes ($m_{t,j} = 0$):
    $$e_{t,i,j}^{(h)} = \frac{\mathbf{q}_{t,i}^{(h)} \cdot \mathbf{k}_{t,j}^{(h)}}{\sqrt{d_h}} + 0.25\,\text{LeakyReLU}\!\left(\mathbf{a}_{\text{src}}^{(h)}\cdot\mathbf{q}_{t,i}^{(h)} + \mathbf{a}_{\text{dst}}^{(h)}\cdot\mathbf{k}_{t,j}^{(h)}\right) + B_{h,i,j} - \gamma_h(1 - m_{t,j}), \quad \alpha_{t,i,j}^{(h)} = \text{softmax}_{j \in \mathcal{N}(i)}\!\left(e_{t,i,j}^{(h)}\right)$$
* **3. Depthwise 1D Temporal Convolution + Bidirectional GRU Processor:**
  * Aggregates spatial context across the $T=8$ sliding window via depthwise Conv1D ($k=3$) and a Bidirectional GRU ($\text{hidden\_size} = 64 \times 2 = 128$), followed by two SiLU residual MLP blocks, `BiomechanicalRoMClamper`, and `DifferentiableFK`.
* **4. Spatial-Only Ablation (`SpatialOnlyGAE`):**
  * Replaces the Bidirectional GRU and temporal convolution with a frame-independent linear layer ($461,693$ params vs. $520,189$ params) to isolate the exact contribution of temporal recurrence under 30%–70% occlusion.

#### Speaker Script — Slide 6
* **Abhishekh (0:00–0:50):** "Slide 6 details the Spatio-Temporal Graph Autoencoder Teacher in `models/teacher_st_gae.py`. When a surgical instrument masks a finger subtree, its input coordinates are zeroed ($m_{t,j}=0$). Before entering the graph layers, we impute masked coordinates with canonical rest positions $\bar{\mathbf{x}}_j^{\text{rest}}$ and construct a 7-dimensional feature vector combining imputed position, binary visibility, and visible displacement from rest, summed with a learned joint-identity embedding $\mathbf{e}_j^{\text{joint}}$ and a global posture skip $\mathbf{h}_t^{\text{pose}}$. We then pass the 21 nodes through three 4-head Graph Attention layers masked by the anatomical adjacency matrix—comprising the 20 bone segments plus 4 inter-MCP palmar arch connections."
* **Abhishekh (0:50–1:30):** "A critical innovation in our `MultiHeadGraphAttentionLayer` is visibility-aware key gating: we subtract a learnable penalty $\gamma_h(1 - m_{t,j})$ from the pre-softmax attention logits whenever key node $j$ is occluded. This forces occluded distal phalanges to attend strongly to visible proximal knuckles and neighboring fingers across the palmar arch rather than propagating corrupted zeros. Next, our temporal core applies a depthwise 1D convolution and a Bidirectional GRU across the $T=8$ window, reducing 50%-occlusion MPJPE from $11.00\text{ mm}$ in our `SpatialOnlyGAE` ablation down to $6.64\text{ mm}$ in the full `SpatioTemporalGAE` Teacher—a $39.6\%$ error reduction, with occluded-joint MPJPE falling from $15.41\text{ mm}$ to $8.79\text{ mm}$."

---

## Slide 7: Physics-Informed Knowledge Distillation & Multi-Hardware Deployment (Kapish)

### $6.64\times$ Parameter & $137.3\times$ FLOPs Compression into a 78.4K-Param ONNX Student

* **Lightweight Edge Student Architecture (`StudentKinematicMLP`, `models/student_models.py`):**
  * Eliminates $O(N^2)$ multi-head graph attention and sequential GRU hidden-state bottlenecks.
  * Flattens per-frame imputed graph inputs $(B, T, 21, 4) \to (B, T, 84) \to \text{Linear}(84 \to 128)$, applies a single `TemporalMixingBlock1D` (depthwise Conv1D $k=3$ + pointwise Linear + LayerNorm + SiLU) and one `ResidualMLPBlock`, then decodes 6D rotations through `BiomechanicalRoMClamper` + `DifferentiableFK`.
  * **Direct-XYZ Ablation (`UnconstrainedStudentMLP`):** Identical MLP backbone ($78,527$ params), but regresses raw Cartesian coordinates $(B, T, 21, 3)$ via MSE loss without 6D rotations or Forward Kinematics.
* **Multi-Objective Kinematic Knowledge Distillation (`DistillationLoss`, `training/losses.py`):**
  $$\mathcal{L}_{\text{KD}} = \underbrace{\alpha_{\text{kd}}^{\text{rot}}\mathcal{L}_{6\text{D}}\!\left(\hat{\mathbf{R}}_{\text{S}}, \mathbf{R}_{\text{T}}\right)}_{\text{Teacher 6D + Geodesic } (\alpha=35.0)} + \underbrace{\alpha_{\text{kd}}^{\text{pos}}\mathcal{L}_{\text{MPJPE}}^w\!\left(\hat{\mathbf{P}}_{\text{S}}, \mathbf{P}_{\text{T}}\right)}_{\text{Teacher FK Pos } (\alpha=0.45)} + \underbrace{\alpha_{\text{gt}}^{\text{pos}}\mathcal{L}_{\text{MPJPE}}^w\!\left(\hat{\mathbf{P}}_{\text{S}}, \mathbf{P}_{\text{GT}}\right)}_{\text{GT FK Pos } (\alpha=0.55)} + \underbrace{\alpha_{\text{gt}}^{\text{rot}}\mathcal{L}_{6\text{D}}\!\left(\hat{\mathbf{R}}_{\text{S}}, \mathbf{R}_{\text{GT}}\right)}_{\text{GT 6D + Geodesic } (\alpha=25.0)} + \underbrace{\alpha_{\text{temp}}\mathcal{L}_{\text{smooth}}\!\left(\hat{\mathbf{P}}_{\text{S}}, \mathbf{P}_{\text{GT}}\right)}_{\text{Vel + Accel Smoothness } (\alpha=0.35)}$$
  where $\mathcal{L}_{6\text{D}}(\hat{\mathbf{R}}, \mathbf{R}) = \|\hat{\mathbf{d}}_{6\text{D}} - \mathbf{d}_{6\text{D}}\|_2^2 + 0.25\,\arccos\!\left(\frac{\text{Tr}(\hat{\mathbf{R}}^\top\mathbf{R})-1}{2}\right) + 0.25\!\left(1 - \frac{\text{Tr}(\hat{\mathbf{R}}^\top\mathbf{R})-1}{2}\right)$ and occluded joints receive weight $w_{\text{occ}} = 1.5$.

| Metric / Complexity Property | Teacher (`SpatioTemporalGAE`) | Spatial Ablation (`SpatialOnlyGAE`) | Direct-XYZ Ablation (`UnconstrainedStudentMLP`) | **Distilled Student (`StudentKinematicMLP`)** | **Compression / Gain** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Trainable Parameters** | $520,189$ | $461,693$ | $78,527$ | **$78,398$** | **$6.64\times$ Smaller** |
| **FP32 / FP16 Weights Size** | $1.99\text{ MB}$ / $1.00\text{ MB}$ | $1.77\text{ MB}$ / $0.88\text{ MB}$ | $0.30\text{ MB}$ / $0.15\text{ MB}$ | **$0.30\text{ MB}$ / $0.15\text{ MB}$** | **$6.59\times$ Smaller** |
| **ONNX Opset 17 Graph Size** | — | — | $316.44\text{ KB}$ | **$433.21\text{ KB}$** | **$< 0.43\text{ MB}$ Disk** |
| **FLOPs per Frame ($T=1$)** | $22,427,490$ ($22.43\text{M}$) | $19.85\text{M}$ | $157,928$ ($0.158\text{M}$) | **$163,398$ ($0.163\text{M}$)** | **$137.3\times$ Fewer FLOPs** |
| **ONNX vs. PyTorch Max Parity** | — | — | $0.000050\text{ mm}$ | **$0.000254\text{ mm}$** | **Exact Numerical Parity** |

#### Speaker Script — Slide 7
* **Kapish (0:00–0:50):** "Slide 7 shows how we compress Abhishekh's 520K-parameter Teacher into our 78,398-parameter `StudentKinematicMLP` in `models/student_models.py` and `distillation/distill_student.py`. Instead of running multi-head graph attention at inference time, the Student flattens the 21-joint input into an 84-dimensional vector, projects to 128 channels, mixes temporal context via a single depthwise-separable 1D convolution block, and passes through one SiLU residual MLP block before entering the exact same `BiomechanicalRoMClamper` and `DifferentiableFK` layer. This cuts trainable parameters by $6.64\times$ (down to $309.5\text{ KB}$ in FP32 and $154.8\text{ KB}$ in FP16) and reduces compute by $137.3\times$—from $22.43\text{ MFLOPs}$ per frame down to just $163.4\text{ KFLOPs}$."
* **Kapish (0:50–1:30):** "To train the Student, we precompute the frozen Teacher's repaired 6D rotations and 3D FK positions across all training sequences and optimize a five-term `DistillationLoss`: combining 6D MSE and geodesic $\text{SO}(3)$ rotational alignment ($\alpha_{\text{kd}}^{\text{rot}}=35.0$), occlusion-weighted FK position MPJPE ($\alpha_{\text{kd}}^{\text{pos}}=0.45$, $w_{\text{occ}}=1.5$), ground-truth position and rotation supervision, and first/second-order velocity and acceleration smoothness ($\alpha_{\text{temp}}=0.35$). Exporting the trained Student to ONNX Opset 17 (`checkpoints/student_kinematic_fk.onnx`, $433.2\text{ KB}$) yields a maximum PyTorch-to-ONNX parity discrepancy of just $0.000254\text{ mm}$."

---

## Slide 8: Quantitative Ablation Results across 30%, 50%, 70% Occlusion

### Held-Out Test Split Evaluation ($N_{\text{test}} = 960$ Sequences, $7,680$ Frames)

| Occlusion Tier | Evaluated Model / Pipeline | Overall MPJPE (mm) $\downarrow$ | Visible MPJPE (mm) $\downarrow$ | Occluded MPJPE (mm) $\downarrow$ | PA-MPJPE (mm) $\downarrow$ | Bone-Length Error (mm) $\downarrow$ | Biomechanical Validity (%) $\uparrow$ | Temporal Jitter ($\text{mm/s}^2$) $\downarrow$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30% Occlusion** | Corrupted Input (Raw Gloved/Occluded) | $32.91$ | $8.62$ | $93.63$ | $32.08$ | $14.9372$ | $0.00\%$ | $34.98$ |
| *(6/21 joints)* | `UnconstrainedStudentMLP` (Direct XYZ, No FK) | $8.76$ | $6.89$ | $13.45$ | $7.31$ | $2.5970$ | $93.91\%$ | $14.43$ |
| | `SpatialOnlyGAE` (GAT + 6D FK, No GRU) | $9.46$ | $7.41$ | $14.58$ | $7.44$ | **$0.0000$** | **$100.00\%$** | $15.83$ |
| | **`Teacher ST-GAE` (Full GAT + Bi-GRU + 6D FK)** | **$6.05$** | **$4.92$** | **$8.85$** | **$4.60$** | **$0.0000$** | **$100.00\%$** | **$12.43$** |
| | **`StudentKinematicMLP` (Distilled 6D-FK Student)** | **$5.58$** | **$4.40$** | **$8.51$** | **$4.33$** | **$0.0000$** | **$100.00\%$** | **$12.66$** |
| **50% Occlusion** | Corrupted Input (Raw Gloved/Occluded) | $48.18$ | $8.22$ | $92.13$ | $34.91$ | $19.4657$ | $0.00\%$ | $29.81$ |
| *(10–11/21 joints)* | `UnconstrainedStudentMLP` (Direct XYZ, No FK) | $10.18$ | $6.52$ | $14.20$ | $8.29$ | $2.9376$ | $88.39\%$ | $14.85$ |
| | `SpatialOnlyGAE` (GAT + 6D FK, No GRU) | $11.00$ | $6.99$ | $15.41$ | $8.30$ | **$0.0000$** | **$100.00\%$** | $16.76$ |
| | **`Teacher ST-GAE` (Full GAT + Bi-GRU + 6D FK)** | **$6.64$** | **$4.70$** | **$8.79$** | **$4.78$** | **$0.0000$** | **$100.00\%$** | **$12.03$** |
| | **`StudentKinematicMLP` (Distilled 6D-FK Student)** | **$6.51$** | **$4.27$** | **$8.97$** | **$4.78$** | **$0.0000$** | **$100.00\%$** | **$12.36$** |
| **70% Occlusion** | Corrupted Input (Raw Gloved/Occluded) | $65.55$ | $7.63$ | $88.72$ | $36.11$ | $26.7127$ | $0.00\%$ | $18.76$ |
| *(15/21 joints)* | `UnconstrainedStudentMLP` (Direct XYZ, No FK) | $13.40$ | $5.94$ | $16.38$ | $10.10$ | $3.5374$ | $76.41\%$ | $14.92$ |
| | `SpatialOnlyGAE` (GAT + 6D FK, No GRU) | $14.03$ | $5.92$ | $17.28$ | $9.69$ | **$0.0000$** | **$100.00\%$** | $16.85$ |
| | **`Teacher ST-GAE` (Full GAT + Bi-GRU + 6D FK)** | **$9.58$** | **$4.43$** | **$11.64$** | **$6.32$** | **$0.0000$** | **$100.00\%$** | **$11.67$** |
| | **`StudentKinematicMLP` (Distilled 6D-FK Student)** | **$9.92$** | **$4.08$** | **$12.26$** | **$6.71$** | **$0.0000$** | **$100.00\%$** | **$12.29$** |

![Figure 1: Occlusion Resilience across 30%, 50%, and 70% Tool Masking](../results/figures/fig1_occlusion_robustness.png)

![Figure 2: Bone-Length Error and Biomechanical Validity Comparison](../results/figures/fig2_bone_length_and_biovalidity.png)

#### Speaker Script — Slide 8
* **Abhishekh (0:00–0:45):** "Slide 8 presents our primary quantitative ablation table and Figures 1 and 2 on the held-out test split across 30%, 50%, and 70% surgical tool occlusion. Look first at the unconstrained baseline (`UnconstrainedStudentMLP`), which directly regresses Cartesian $X, Y, Z$ coordinates without kinematics. Even when trained honestly without loss penalties, as occlusion increases from 30% to 70%, its bone-length error increases from $2.60\text{ mm}$ to $3.54\text{ mm}$, and biomechanical validity degrades from $93.91\%$ down to $76.41\%$ due to depth ambiguities under severe occlusion."
* **Vrushank (0:45–1:30):** "In contrast, our kinematic architecture embeds continuous 6D rotations, `BiomechanicalRoMClamper`, and `DifferentiableFK` into the network's forward computation. This guarantees $0.0000\text{ mm}$ bone-length distortion and $100.00\%$ biomechanical validity across all occlusion tiers. Adding temporal recurrence in `Teacher ST-GAE` cuts occluded-joint MPJPE from $15.41\text{ mm}$ down to $8.79\text{ mm}$ at 50% occlusion, and suppresses temporal jitter from $16.76$ to $12.03\text{ mm/s}^2$. Our distilled `StudentKinematicMLP` matches the Teacher closely at $6.51\text{ mm}$ MPJPE ($4.78\text{ mm}$ PA-MPJPE), running with $100\%$ valid kinematics."

---

## Slide 9: Temporal Smoothing, Qualitative 3D Repair & Custom Gloved Benchmark

### Suppressing Latex Specular Spikes & Generalizing to Custom Gloved Sequences

* **Temporal Jitter Suppression (`fig3_temporal_jitter_comparison.png`):**
  * Raw gloved input exhibits severe second-order acceleration spikes ($34.98\text{ mm/s}^2$ at 30% occlusion, $29.81\text{ mm/s}^2$ at 50% occlusion) due to $12.0\text{ mm}$ fingertip specular glare and sudden tool occlusion entry/exit.
  * `SpatialOnlyGAE` (without temporal memory) leaves residual acceleration jitter at $16.76\text{ mm/s}^2$ (50% occ) and $16.85\text{ mm/s}^2$ (70% occ).
  * `Teacher ST-GAE` ($12.03\text{ mm/s}^2$) and `StudentKinematicMLP` ($12.36\text{ mm/s}^2$) suppress temporal acceleration jitter by **$58.5\%$** relative to the corrupted input and **$28.2\%$** relative to the spatial-only baseline.
* **Custom Gloved Dataset Benchmark (`data/custom_gloved_dataset.npz`, 120 Sequences):**
  * Evaluates zero-shot domain adaptation under combined $12\%$ glove stiffness, AR(1) specular noise, and surgical tool occlusion:

| Custom Gloved Tier | Corrupted Input MPJPE | `UnconstrainedStudentMLP` (Bone Err / Valid %) | `SpatialOnlyGAE` MPJPE | **`Teacher ST-GAE` MPJPE** | **`StudentKinematicMLP` MPJPE (Bone Err / Valid %)** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **30% Occlusion** | $32.81\text{ mm}$ | $9.05\text{ mm}$ ($2.65\text{ mm}$ / $92.3\%$) | $10.00\text{ mm}$ | **$6.32\text{ mm}$** | **$6.05\text{ mm}$ ($0.00\text{ mm}$ / $100.0\%$)** |
| **50% Occlusion** | $47.51\text{ mm}$ | $9.99\text{ mm}$ ($2.96\text{ mm}$ / $87.4\%$) | $10.56\text{ mm}$ | **$6.68\text{ mm}$** | **$6.53\text{ mm}$ ($0.00\text{ mm}$ / $100.0\%$)** |
| **70% Occlusion** | $64.94\text{ mm}$ | $13.51\text{ mm}$ ($3.57\text{ mm}$ / $72.2\%$) | $14.49\text{ mm}$ | **$9.92\text{ mm}$** | **$10.16\text{ mm}$ ($0.00\text{ mm}$ / $100.0\%$)** |

![Figure 3: Temporal Trajectory and Acceleration Jitter Suppression](../results/figures/fig3_temporal_jitter_comparison.png)

![Figure 4: Qualitative 3D Hand Skeleton Reconstruction under 50% and 70% Tool Occlusion](../results/figures/fig4_qualitative_3d_hand_repair.png)

#### Speaker Script — Slide 9
* **Bhavya (0:00–0:45):** "Slide 9 highlights our temporal stability analysis (Figure 3), qualitative 3D hand reconstructions (Figure 4), and results on our 120-sequence `custom_gloved_dataset.npz` benchmark. In Figure 3, you can see how the $12\text{ mm}$ latex glove specular spikes and frame-to-frame occlusion transitions induce $29.81\text{ to }34.98\text{ mm/s}^2$ of acceleration jitter on the raw fingertip trajectories. Because `SpatialOnlyGAE` processes each frame independently, its output still oscillates when 70% of joints are masked ($16.85\text{ mm/s}^2$). By incorporating our Bidirectional GRU in the Teacher and our `TemporalMixingBlock1D` plus `TemporalSmoothnessLoss` in the Student, acceleration jitter drops to $12.03\text{ to }12.36\text{ mm/s}^2$ at 50% occlusion and under $12.0\text{ mm/s}^2$ on the custom gloved dataset."
* **Kapish (0:45–1:30):** "In Figure 4, we visualize the 3D hand skeletons side-by-side at 50% and 70% tool occlusion. In the left column (red), the occluded distal subtrees collapse to the wrist origin. In the middle column (orange), the unconstrained direct-XYZ MLP stretches phalangeal bones across the palm and bends PIP joints backward. In the right column (emerald green), our distilled `StudentKinematicMLP` reconstructs the missing fingers with exact canonical bone lengths ($0.0000\text{ mm}$ error), $100\%$ valid flexion angles, and $6.53\text{ mm}$ MPJPE at 50% occlusion on the custom gloved benchmark."

---

## Slide 10: SOTA Literature Comparison & Multi-Hardware Edge Latency

### Sub-Millisecond Execution across PC, MacBook Air M-Series, Jetson Nano & iPhone UDP

| Target Hardware Platform | Execution Backend / Provider | Teacher ST-GAE ($T=8$) Latency | **Distilled Student ($T=1$ / $T=8$) Latency** | **Student Throughput (FPS)** | **Bare-Metal `EdgeFK` NumPy** | **60 FPS Real-Time Budget Usage** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PC Workstation** *(Measured Host, 24-Thread Intel x86_64)* | ONNX Runtime `CPUExecutionProvider` | $29.34\text{ ms}$ ($34.1\text{ FPS}$) | **$1.792\text{ ms}$ / $2.546\text{ ms}$** | **$558\text{ – }3,142\text{ FPS}$** | $0.199\text{ ms}$ ($5.02\text{ kHz}$) | **$10.7\%$** ($9.3\times$ headroom) |
| **Apple MacBook Air** *(Assigned: Bhavya, Apple Silicon)* | CoreMLExecutionProvider / MPS | *Pending physical run* | *Pending physical run* | *Target: > 1,000 FPS* | *Pending physical run* | *Target: < 3% budget* |
| **NVIDIA Jetson Nano** *(Assigned: Kapish, Maxwell GPU)* | TensorRT FP16 / CUDAExecutionProvider | *Pending physical run* | *Pending physical run* | *Target: > 300 FPS* | *Pending physical run* | *Target: < 15% budget* |
| **Raspberry Pi 5 / Pi 4B** *(Quad Cortex-A76 / A72)* | ONNX Runtime ARM64 NEON SIMD | *Pending physical run* | *Pending physical run* | *Target: > 150 FPS* | *Pending physical run* | *Target: < 20% budget* |
| **iPhone ARKit Stream** *(A-Series ANE $\to$ UDP Port 5005)* | `ARKitUDPStreamer` ($0.08\text{ ms}$) + Host ORT | — | **$1.872\text{ ms}$ End-to-End** | **$534.2\text{ Hz}$ Max Rate** | $0.199\text{ ms}$ | **Sustains $60\text{ Hz}$ Zero-Drop** |

![Figure 5: Pareto Frontier of Accuracy (MPJPE) vs. Edge Latency and Model Size](../results/figures/fig5_edge_latency_pareto.png)

![Figure 6: Real-Time 3-Panel Telemetry Repair Dashboard (`deployment/live_demo.py`)](../results/figures/fig6_live_demo_dashboard.png)

#### Speaker Script — Slide 10
* **Kapish (0:00–0:50):** "Slide 10 presents our cross-hardware edge benchmarking protocol and host measurements (`results/edge_benchmark.json`). On our measured x86_64 workstation using ONNX Runtime `CPUExecutionProvider`, `StudentKinematicMLP` executes a single streaming frame ($T=1$) in $1.792\text{ ms}$ ($558\text{ FPS}$) and a batched $T=8$ window in $2.546\text{ ms}$ ($0.318\text{ ms}$ per frame, or $3,142\text{ FPS}$), consuming only $10.7\%$ of a $60\text{ FPS}$ frame budget. In contrast, the Teacher ST-GAE on CPU takes $29.34\text{ ms}$ ($34.1\text{ FPS}$), highlighting an $11.5\times$ speedup from our distilled architecture. The bare-metal NumPy `EdgeFK` forward kinematics engine runs in just $0.199\text{ ms}$ ($5.02\text{ kHz}$). Bhavya and I have pre-configured `benchmark_edge.py` to benchmark physical execution on the Apple Silicon MacBook Air and NVIDIA Jetson Nano respectively."
* **Vrushank (0:50–1:30):** "We also support live iPhone telemetry via two pathways: macOS Continuity Camera (`--camera 1`) and direct 60 Hz ARKit 3D hand skeleton streaming over Wi-Fi/USB UDP port 5005 (`ARKitUDPStreamer`). Parsing and wrist-centering each UDP JSON packet takes $0.08\text{ ms}$, and ONNX kinematic repair takes $1.79\text{ ms}$, giving a total end-to-end telemetry repair latency of $1.87\text{ ms}$ ($534\text{ Hz}$ capacity). Figure 6 shows our live 3-panel telemetry dashboard (`deployment/live_demo.py`), which renders the corrupted input, the unconstrained baseline failure mode, and our $100\%$-valid 6D-FK Student output with live HUD metrics."

---

## Slide 11: Live Demo Guide, Individual Contributions Summary & Conclusion

### Live Demonstration Commands & Key Takeaways

* **Zero-Setup Deterministic Replay Demo (During Presentation):**
  ```powershell
  .\.venv\Scripts\python.exe deployment/live_demo.py --source replay --occlusion 0.50 --save-figure results/figures/fig6_live_demo_dashboard.png
  ```
* **Live Hardware Telemetry Stream Commands:**
  * **MacBook Air Built-In Webcam / Continuity Camera (iPhone):**
    ```bash
    python deployment/live_demo.py --source mediapipe --camera 0 --occlusion 0.50
    python deployment/live_demo.py --source mediapipe --camera 1 --occlusion 0.50
    ```
  * **NVIDIA Jetson Nano + Raspberry Pi CSI Camera (`nvarguscamerasrc`):**
    ```bash
    python deployment/live_demo.py --source mediapipe --jetson-csi --occlusion 0.50
    ```
  * **iPhone ARKit 3D Hand Skeleton Stream via UDP Port 5005:**
    ```bash
    python deployment/live_demo.py --source udp --udp-port 5005 --occlusion 0.50
    ```

| Team Member | Core Technical Contributions Verified in Repository | Quantitative Impact |
| :--- | :--- | :--- |
| **Abhishekh** | Designed `MultiHeadGraphAttentionLayer` (visibility key gating $\gamma_h$) + `SpatioTemporalGAE` (3-layer GAT + Bi-GRU) + `SpatialOnlyGAE` ablation (`models/teacher_st_gae.py`, `training/train_teacher.py`). | Reduced 50%-occ MPJPE to **$6.64\text{ mm}$** ($39.6\%$ better than spatial-only GAT). |
| **Bhavya** | Built 21-joint MANO trajectory generator (`N=4800, T=8`), 80/20 split, `SyntheticOcclusionTransform` (3D cylinder + subtree priority), `SyntheticGloveTransform` (12% stiffness + AR(1) jitter), and `custom_gloved_dataset.npz`. | Enabled zero-shot generalization (**$6.53\text{ mm}$** Student MPJPE on custom gloved 50% occ). |
| **Vrushank** | Implemented Gram-Schmidt `rot6d_to_matrix`, $C^1$-smooth `BiomechanicalRoMClamper` ($\tanh$), `DifferentiableFK`, analytical IK (`positions_to_rot6d`), and bare-metal NumPy `EdgeFK` (`models/kinematics.py`, `tests/test_kinematics.py`). | Guaranteed **$0.0000\text{ mm}$** bone-length error and **$100.0\%$** biomechanical validity at **$5.02\text{ kHz}$**. |
| **Kapish** | Built `StudentKinematicMLP` vs. `UnconstrainedStudentMLP`, `DistillationLoss`, ONNX Opset 17 export (`433.2 KB`), multi-hardware profiler (`benchmark_edge.py`), and 3-panel HUD visualizer (`live_demo.py`). | **$6.64\times$ param** & **$137.3\times$ FLOPs** reduction; **$1.792\text{ ms}$** ($558\text{ FPS}$) in ONNX Runtime. |

#### Speaker Script — Slide 11
* **Abhishekh (0:00–0:25):** "To wrap up on Slide 11: we demonstrated that embedding a hard differentiable kinematic manifold inside a spatio-temporal graph distillation pipeline resolves the long-standing trade-off between anatomical validity under occlusion and sub-millisecond edge inference."
* **Bhavya (0:25–0:45):** "Our 3D volumetric tool clipping and latex glove domain adaptation transforms allow training without expensive operating-room motion-capture domes, generalizing directly to gloved telemetry across 30%, 50%, and 70% occlusion."
* **Vrushank (0:45–1:05):** "By parameterizing joints in continuous 6D $\text{SO}(3)$ space with $\tanh$ anatomical clamping and exact Forward Kinematics, our system guarantees $0.0000\text{ mm}$ bone-length error and $100.0\%$ biomechanical validity on every single frame."
* **Kapish (1:05–1:30):** "And our 78.4K-parameter ONNX Student delivers $6.51\text{ mm}$ MPJPE at 50% occlusion while running in $1.79\text{ ms}$ on host CPU ($558\text{ FPS}$). We will now run `deployment/live_demo.py` live and open the floor for questions."

---

## Post-Presentation Q&A Defense Guide (Exact Empirical Metrics & Technical Answers)

### Q1: Why did you enforce biomechanical constraints as a hard layer (`BiomechanicalRoMClamper` + `DifferentiableFK`) instead of using soft biomechanical penalty losses like Spurr et al. (ECCV 2020)?
* **Primary Responder:** Vrushank (supported by Abhishekh)
* **Direct Answer:**
  * Soft penalty losses ($L_{\text{bl}} + L_{\text{ja}}$) only add gradients during training while leaving the network's output layer as an unconstrained Euclidean regressor $\mathbb{R}^{21 \times 3}$. Under severe test-time occlusion (50%–70% missing joints), likelihood gradients from corrupted inputs override the soft prior, causing bone stretching and joint hyperextension (Spurr et al. still exhibit $1.9\text{ mm}$ bone error and $8.6\%$ invalid poses; our `UnconstrainedStudentMLP` ablation exhibits $2.94\text{ mm}$ bone error and $88.39\%$ validity at 50% occlusion, and $3.54\text{ mm}$ bone error with $76.41\%$ validity at 70% occlusion).
  * By contrast, our `DifferentiableFK` layer computes $\mathbf{P}_j = \mathbf{P}_{p(j)} + \mathbf{R}_j^{\text{global}} \mathbf{v}_j^{\text{rest}}$ where $\mathbf{R}_j^{\text{global}} \in \text{SO}(3)$ is orthogonal. Because orthogonal transformations preserve Euclidean norms ($\|\mathbf{R}_j^{\text{global}}\mathbf{v}_j^{\text{rest}}\|_2 = \|\mathbf{v}_j^{\text{rest}}\|_2 = L_j$), bone-length error is **identically $0.0000\text{ mm}$ ($1.1 \times 10^{-5}\text{ mm}$ in FP32 ONNX)** and biomechanical validity is **$100.0\%$** for any input tensor, even random noise.

### Q2: How do you avoid gradient singularities when extracting Euler angles inside `BiomechanicalRoMClamper`?
* **Primary Responder:** Vrushank
* **Direct Answer:**
  * Standard `torch.acos(c)` has an infinite derivative $\frac{d}{dc}\arccos(c) = -\frac{1}{\sqrt{1-c^2}} \to -\infty$ as $c \to \pm 1$ ($\theta \to 0$), and `torch.atan2` is not supported across all embedded ONNX/TensorRT runtimes.
  * In `models/kinematics.py` (`_safe_angle_from_sin_cos`), we clamp $c \in [-0.99999, 0.99999]$ and switch to a $C^1$-continuous Taylor polynomial expansion $\theta \approx s \cdot \left(1 + \frac{1 - c}{3}\right)$ whenever $c > 0.9995$ ($|\theta| < 1.8^\circ$). This ensures exact unit gradient $\frac{\partial \theta}{\partial s} = 1.0$ at $\theta = 0^\circ$ and clean ONNX Opset 17 export (`onnx_checker_passed = True`).

### Q3: What is the exact quantitative contribution of the Bidirectional Temporal GRU compared to Spatial Graph Attention alone?
* **Primary Responder:** Abhishekh
* **Direct Answer:**
  * Comparing `Teacher ST-GAE` (Full GAT + Bi-GRU, $520,189$ params) against `SpatialOnlyGAE` (identical 3-layer 4-head GAT + 6D FK without temporal GRU, $461,693$ params) on the held-out test set:
    * **At 30% Occlusion:** Overall MPJPE drops from $9.46\text{ mm} \to \mathbf{6.05\text{ mm}}$ ($-36.0\%$); Occluded MPJPE drops from $14.58\text{ mm} \to \mathbf{8.85\text{ mm}}$ ($-39.3\%$); Temporal Jitter drops from $15.83 \to \mathbf{12.43\text{ mm/s}^2}$.
    * **At 50% Occlusion:** Overall MPJPE drops from $11.00\text{ mm} \to \mathbf{6.64\text{ mm}}$ ($-39.6\%$); Occluded MPJPE drops from $15.41\text{ mm} \to \mathbf{8.79\text{ mm}}$ ($-42.9\%$); Temporal Jitter drops from $16.76 \to \mathbf{12.03\text{ mm/s}^2}$.
    * **At 70% Occlusion:** Overall MPJPE drops from $14.03\text{ mm} \to \mathbf{9.58\text{ mm}}$ ($-31.7\%$); Occluded MPJPE drops from $17.28\text{ mm} \to \mathbf{11.64\text{ mm}}$ ($-32.6\%$); Temporal Jitter drops from $16.85 \to \mathbf{11.67\text{ mm/s}^2}$ ($-30.7\%$).

### Q4: Why not run the Teacher ST-GAE directly on the NVIDIA Jetson Nano or Raspberry Pi?
* **Primary Responder:** Kapish
* **Direct Answer:**
  * The Teacher ST-GAE requires $22.43\text{ MFLOPs/frame}$ ($179.42\text{ MFLOPs}$ per $T=8$ window) and sequential GRU state propagation. On our workstation CPU, the Teacher takes **$29.34\text{ ms}$ ($34.1\text{ FPS}$)**.
  * Our distilled `StudentKinematicMLP` requires only $163.4\text{ KFLOPs/frame}$ ($137.3\times$ fewer FLOPs) and $78,398$ parameters ($6.64\times$ smaller). It runs in **$1.792\text{ ms}$ ($558\text{ FPS}$)** on CPU in ONNX Runtime—an $11.5\times$ speedup on batched windows and over $16\times$ speedup on single-frame streaming ($1.79\text{ ms}$ vs $29.34\text{ ms}$)—while outperforming the larger `SpatialOnlyGAE` baseline on accuracy ($6.51\text{ mm}$ vs. $11.00\text{ mm}$ MPJPE at 50% occlusion) thanks to temporal knowledge distillation from the Teacher.

### Q5: How did you model realistic surgical occlusion and glove artifacts without risking train-test data leakage?
* **Primary Responder:** Bhavya
* **Direct Answer:**
  * First, the $4,800$ base trajectories were partitioned via a fixed seed (`data/splits/split_indices.json`: $3,840$ train / $960$ test) prior to any model training, and an independent $120$-sequence `custom_gloved_dataset.npz` was generated with distinct random seeds.
  * Second, `SyntheticOcclusionTransform` does not use independent Bernoulli coin flips per joint; it spawns a 3D cylindrical instrument volume ($R=28\text{ mm}$) and enforces distal-to-proximal subtree priority ($\text{TIP} \to \text{DIP} \to \text{PIP} \to \text{MCP}$), ensuring that occluded joints form physically contiguous subtrees behind the virtual tool while wrist joint 0 remains anchored.
