# Canva Presentation Flow & Visual Redesign Blueprint
**Target Defense:** Foundations of Data Science (FDS) Final Evaluation (`15 Marks` Total: Introduction `1M`, Objectives `1M`, Block Diagram `3M`, Results `5M`, Live Demo `5M`)  
**Team:** Abhishekh Verma, Bhavya, Vrushank, Kapish

---

## 1. Flow & Choreography Critique: Stop "Microphone Ping-Pong"

### The Problem in the Current Script
Your original speaker script switches speakers **14+ times across 11 slides** (e.g., all 4 teammates passing the mic on Slide 1 in 90 seconds, then switching twice per slide on Slides 2, 3, 8, 9, 10, 11). Passing a microphone 14 times wastes **90–120 seconds of dead air** and breaks narrative momentum.

### The Fix: 4 Contiguous Speaker Blocks (Only 3 Handoffs Total!)
Give each teammate one continuous, authoritative block:

| Time Window | Speaker | Slides Covered | Narrative Focus & Rubric Weight |
| :--- | :--- | :--- | :--- |
| **0:00 – 3:00** | **Abhishekh Verma** | **Slides 1, 2, 3** | **Hook, Literature Gap & Block Diagram (`5 Marks`: Intro `1M` + Obj `1M` + Block Diagram `3M`):** Open the talk, introduce the team in 15 seconds (teammates wave—no mic passing on Slide 1!), contrast soft-penalty prior work vs. hard FK on Slide 2, and walk through the 4-stage Block Diagram (`fig0`) on Slide 3. |
| **3:00 – 5:15** | **Bhavya** | **Slide 4 (+ Slide 9 Gloved)** | **Hybrid FreiHAND+MANO Data & Glove/Tool Physics:** Detail the $32,560$-pose FreiHAND stream, 80/20 split, `SyntheticGloveTransform` ($12\%$ stiffness + AR(1) spikes), and `SyntheticOcclusionTransform` (3D cylinder). |
| **5:15 – 7:45** | **Vrushank** | **Slide 5 (+ Slide 8 Proof)** | **6D $\text{SO}(3)$ Manifold, $\tanh$ RoM Clamper & Zero-Bone-Error Proof:** Present Gram-Schmidt $\mathbb{R}^6 \to \text{SO}(3)$, $C^1$ Euler clamping, and Theorem 1 ($\|\mathbf{P}_j - \mathbf{P}_{p(j)}\|_2 \equiv L_j \implies 0.0000\text{ mm}$). |
| **7:45 – 10:00** | **Abhishekh Verma** | **Slides 6, 8, 9** | **Teacher ST-GAE & Experimental Results (`5 Marks`):** Present visibility-gated GAT ($-\gamma_h(1-m_{t,j})$) + Bi-GRU ($T=8$), then immediately show the payoff in Slide 8 (30%/50%/70% ablation) and Slide 9 (Phalanx hierarchy & gesture utility). |
| **10:00 – 13:30** | **Kapish** | **Slides 7, 10, 11 + Live Demo** | **Knowledge Distillation, Edge Hardware & Live HUD (`5 Marks`):** Present the $6.64\times$ param / $137.3\times$ FLOPs Student compression (Slide 7), MacBook Air CoreML ($805\text{ FPS}$) & PC ($558\text{ FPS}$) benchmarks (Slide 10), and launch `run_presentation_demo.py` (Slide 11). |

---

## 2. Four Visual Bottlenecks to Fix When Moving to Canva

1. **Microscopic Font Sizes (`7.7pt` Tables & `9.4pt` Body):** On a classroom projector from 5 meters away, anything below `14pt` blurs together. Keep Canva body text at **`15pt–17pt`** and table text at **`13.5pt–15pt`**.
2. **Paragraph Walls (~320 Words per Slide):** Cut on-slide word count by **65%** ($\le 75$ words per slide). Put the full spoken sentences into **Canva Speaker Notes** and keep only punchy keywords + metrics on the slide.
3. **The 135-Cell Table on Slide 8:** Slide 8 stacks a $15 \times 9$ table (135 cells!) right above `fig1` and `fig2`, which *already plot those exact numbers*. Replace the 15-row table with the **5-row summary table** below so `fig1` and `fig2` can be twice as tall.
4. **10 Subplots Squeezed onto Slide 9:** Slide 9 crams 6 bullets + a table + `fig3` (3 panels) + `fig4` (4 3D hands) + `fig7` (3 panels) into one tiny strip. Use the **2x2 Visual Dashboard Grid** below instead.

---

## 3. Slide-by-Slide Canva Redesign Blueprint (Copy-Paste Ready)

### SLIDE 1: Title, Clinical Motivation & Hero Impact (`Rubric: Introduction — 1 Mark`)
* **Canva Layout:**
  * **Left Column (48%):** 2 Red "Failure Mode" cards (`#FFF1F2` fill, `#E11D48` border) + 1 Emerald "Core Novelty" card (`#ECFDF5` fill, `#10B981` border).
  * **Right Column (52%):** Embed **`results/figures/live_demo_telemetry.gif`** (Canva loops GIFs natively!) + **3 Hero Stat Badges** underneath.
  * **Bottom Strip (12%):** 4 clean Team Role Pills (Name + 4-word role, no `.py` file paths).
* **Copy-Paste On-Slide Text:**
  * **Failure Mode 1 — Surgical Tool Occlusion:** Scalpels, drills & probes block **30%–70%** of joints $\to$ direct $\mathbb{R}^{21\times 3}$ regressors collapse fingers & stretch bones by **15–27 mm**.
  * **Failure Mode 2 — Latex Glove Degradation:** Medical gloves add **12% elastic stiffness** + specular glare ($\sigma = 4.5\text{ mm}$ AR(1) jitter, **12 mm** tip spikes).
  * **Our Solution — Constrained Graph Kinematics:** Distill a **520K-param Teacher ST-GAE** into a **78.4K-param Edge Student MLP** predicting continuous **6D $\text{SO}(3)$ rotations** + $\tanh$ RoM clamping + exact **Differentiable FK**.
  * **3 Hero Stat Badges:** `0.0000 mm` Bone Error (`100%` Valid) • `7.04 mm` Student MPJPE @ 50% Occ • `805 FPS` (`1.24 ms`) MacBook Air CoreML

---

### SLIDE 2: Project Objectives & Literature Gap (`Rubric: Objectives — 1 Mark`)
* **Canva Layout:**
  * **Top Row (32%):** **5 Numbered Objective Cards** in a 5-column horizontal row.
  * **Bottom Area (68%):** **Simplified $7 \times 6$ Literature Comparison Table** (`15pt` font, highlight bottom 2 rows in `#ECFDF5` mint green).
* **Copy-Paste On-Slide Text:**
  * **Obj 1 (Hybrid Data):** $32,560$ real FreiHAND poses + 6 clinical gestures (30%/50%/70% 3D tool masking, 80/20 split).
  * **Obj 2 (Hard 6D FK):** Gram-Schmidt $\text{SO}(3)$ + $C^1$ $\tanh$ RoM clamping $\to$ **$0.0000\text{ mm}$** bone error.
  * **Obj 3 (Teacher ST-GAE):** 3-Layer 4-Head visibility-gated GAT + Bi-GRU ($T=8$) vs. `SpatialOnlyGAE`.
  * **Obj 4 (Edge Distillation):** $6.64\times$ smaller, $137.3\times$ fewer FLOPs (`78.4K` params) via 5-term geodesic + FK loss.
  * **Obj 5 (Live Telemetry):** $433\text{ KB}$ ONNX model running at **$805\text{ FPS}$** (CoreML) & **$558\text{ FPS}$** (PC) with 1/2-hand toggle.

| Method (Venue) | Output Space | MPJPE $\downarrow$ | Bone Error $\downarrow$ | Bio-Valid $\uparrow$ | Size / Speed $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HOPE-Net** (CVPR'20) | Adaptive GCN ($\mathbb{R}^{21\times 3}$) | $26.8\text{ mm}$ | $4.80\text{ mm}$ | $71.0\%$ | $96\text{ MB}$ ($60\text{ FPS}$) |
| **Spurr et al.** (ECCV'20) | CNN + Soft Hinge Loss | $22.4\text{ mm}$ | $1.90\text{ mm}$ | $91.4\%$ | $90\text{ MB}$ ($66\text{ FPS}$) |
| **HandOccNet** (CVPR'22) | Transformer + MANO | $24.0\text{ mm}$ | $0.80\text{ mm}$ | $94.2\%$ | $142\text{ MB}$ ($30\text{ FPS}$) |
| **MediaPipe** (CVPRW'20) | Mobile CNN ($\mathbb{R}^{21\times 3}$) | $19.5\text{–}28.4$ | $6.20\text{ mm}$ | $64.8\%$ | $5.5\text{ MB}$ ($220\text{ FPS}$) |
| **`UnconstrainedMLP`** (Abl.) | Direct $\mathbb{R}^{21\times 3}$ (No FK) | $9.95\text{ mm}$ | $2.31\text{ mm}$ | $29.95\%$ | $0.30\text{ MB}$ ($550\text{ FPS}$) |
| **`Teacher ST-GAE` (Ours)** | **GAT + Bi-GRU + Hard FK** | **$6.51\text{ mm}$** | **$0.0000\text{ mm}$** | **$100.0\%$** | **$1.99\text{ MB}$ ($166\text{ FPS}$)** |
| **`StudentKinematicMLP` (Ours)** | **Distilled MLP + Edge FK** | **$7.04\text{ mm}$** | **$0.0000\text{ mm}$** | **$100.0\%$** | **$0.30\text{ MB}$ ($805\text{ FPS}$)** |

---

### SLIDE 3: System Block Diagram (`Rubric: Block Diagram — 3 Marks`)
* **Canva Layout:**
  * **Delete all top bullet text** (because `fig0_block_diagram.png` already has all details inside its boxes!).
  * **Main Area (82% height):** Stretch **`results/figures/fig0_block_diagram.png`** edge-to-edge (`1760 px` wide).
  * **Bottom Strip (10% height):** 4 color-coded summary pills aligned under Stages 1, 2, 3, and 4.
* **Copy-Paste Bottom 4 Pills:**
  * **Stage 1 (Blue):** $(B, T, 21, 4)$ Input `[x, y, z, m]` • $32.5\text{K}$ FreiHAND + MANO
  * **Stage 2 (Purple):** Offline Teacher `SpatioTemporalGAE` • $520\text{K}$ Params (`6.51 mm`)
  * **Stage 3 (Teal):** Distilled `StudentKinematicMLP` • $78.4\text{K}$ Params (`137x` Less Compute)
  * **Stage 4 (Green):** Shared $\text{SO}(3)$ + $\tanh$ RoM + `EdgeFK` • **$0.0000\text{ mm}$ Bone Error @ $805\text{ FPS}$**

---

### SLIDE 4: Hybrid FreiHAND + MANO Data Pipeline & 3D Domain Adaptation
* **Canva Layout:** **3 Vertical Cards** side-by-side (Col 1: Hybrid Corpus; Col 2: `SyntheticGloveTransform`; Col 3: `SyntheticOcclusionTransform` + cropped 3D hand visual from `fig4`).
* **Copy-Paste On-Slide Text:**
  * **Card 1 — 50/50 Hybrid Real + Clinical Corpus:**
    * **$32,560$ Real 3D Poses** (`FreiHAND_pub_v2`) $\to$ Rodrigues IK (`positions_to_rot6d`) + canonical bone harmonization.
    * **6 Clinical Gestures ($T=8$):** Scalpel Pinch, Needle Driver, Power Drill, Flexion Waves, Precision Opposition, Flat Palpation.
    * **Strict 80/20 Split:** $960$ Train / $240$ Test sequences per tier (`split_indices.json`) + $120$ Custom Gloved sequences.
  * **Card 2 — `SyntheticGloveTransform` (Latex Physics):**
    * **12% Elastic Stiffness** damping toward neutral rest vector $\hat{\mathbf{u}}_j^{\text{rest}}$:
      $$\tilde{\mathbf{d}}_{t,j} = \text{normalize}\!\left(0.88\,\hat{\mathbf{d}}_{t,j} + 0.12\,\hat{\mathbf{u}}_j^{\text{rest}}\right)$$
    * **Stationary AR(1) Drift + Glare Spikes:**
      $$\mathbf{P}_{t,j}^{\text{gloved}} = \tilde{\mathbf{P}}_{t,j} + \sigma_j\boldsymbol{\epsilon}_{t,j} + \mathbb{I}(j \in \mathcal{V}_{\text{tip}})\,b_{t,j}\boldsymbol{\xi}_{t,j}$$
      ($\rho = 0.65$ with $\sqrt{1-\rho^2}$ stationary variance, $\sigma_j = 4.5\text{ mm}$, $20\%$ Bernoulli fingertip spikes $\sigma_{\text{spike}} = 12.0\text{ mm}$).
  * **Card 3 — `SyntheticOcclusionTransform` (3D Tool Cylinder):**
    * 3D instrument cylinder ($R_{\text{cyl}} = 28.0\text{ mm}$) across phalanges.
    * **Contiguous Subtree Masking:** $\text{TIP} \to \text{DIP} \to \text{PIP} \to \text{MCP}$ at **30% ($6/21$)**, **50% ($10/21$)**, and **70% ($15/21$)** while anchoring wrist $j=0$.

---

### SLIDE 5: 6D Continuous Rotations & Hard Edge Forward Kinematics
* **Canva Layout:**
  * **Left Column (52%):** 3 stacked equation cards (1. Gram-Schmidt $\to$ 2. $\tanh$ Clamper $\to$ 3. Theorem 1 Proof in bold green box).
  * **Right Column (48%):** Anatomical RoM Bounds Table + **`EdgeFK` $13.79\text{ kHz}$ ($0.0725\text{ ms}$) Callout Badge**.
* **Copy-Paste On-Slide Text:**
  * **1. Singularity-Free 6D $\to \text{SO}(3)$ (`rot6d_to_matrix`):**
    $$\mathbf{b}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|_2}, \quad \mathbf{b}_2 = \text{normalize}\!\left(\mathbf{a}_2 - (\mathbf{b}_1\cdot\mathbf{a}_2)\mathbf{b}_1\right), \quad \mathbf{R}_j^{\text{raw}} = [\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_1 \times \mathbf{b}_2] \in \text{SO}(3)$$
  * **2. $C^1$-Smooth Anatomical Clamping (`BiomechanicalRoMClamper`):**
    $$\boldsymbol{\theta}_j^{\text{clamp}} = \boldsymbol{\mu}_j + \boldsymbol{\Delta}_j \odot \tanh\!\left(\frac{\boldsymbol{\theta}_j - \boldsymbol{\mu}_j}{\boldsymbol{\Delta}_j \cdot \tau + \epsilon}\right) \in [\boldsymbol{\theta}_j^{\min}, \boldsymbol{\theta}_j^{\max}]$$
  * **3. Theorem 1 (Exact Bone-Length Invariance):**
    $$\mathbf{P}_j = \mathbf{P}_{p(j)} + \mathbf{R}_j^{\text{global}}\mathbf{v}_j^{\text{rest}} \implies \|\mathbf{P}_j - \mathbf{P}_{p(j)}\|_2 = \|\mathbf{v}_j^{\text{rest}}\|_2 \equiv L_j \quad (\varepsilon_{\text{bone}} \equiv \mathbf{0.0000\text{ mm}})$$

| Joint Group | Indices | Flexion ($r_x$) | Axial Twist ($r_y$) | Abduction ($r_z$) | Bone Lengths $L_j$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thumb Chain** | $1,2,3,4$ | $[-15^\circ, +80^\circ]$ | $[-15^\circ, +35^\circ]$ | $[-45^\circ, +25^\circ]$ | $34, 46, 30, 20\text{ mm}$ |
| **Finger MCPs** | $6,10,14,18$ | $[-15^\circ, +90^\circ]$ | $[-5^\circ, +5^\circ]$ | $[-20^\circ, +20^\circ]$ | $40, 42, 39, 31\text{ mm}$ |
| **Finger PIPs** | $7,11,15,19$ | $[0^\circ, +110^\circ]$ *(No hyperext.)* | $[-2^\circ, +2^\circ]$ | $[-2^\circ, +2^\circ]$ | $24, 27, 25, 19\text{ mm}$ |
| **Finger DIPs** | $8,12,16,20$ | $[0^\circ, +80^\circ]$ *(No hyperext.)* | $[-2^\circ, +2^\circ]$ | $[-2^\circ, +2^\circ]$ | $17, 18, 17, 16\text{ mm}$ |

---

### SLIDE 6: Spatio-Temporal Graph Autoencoder Teacher (`SpatioTemporalGAE`)
* **Canva Layout:**
  * **Left Column (58%):** 3 stacked architecture cards (7D Imputation $\to$ 69-Edge Mask-Aware GAT $\to$ Bi-GRU $T=8$).
  * **Right Column (42%):** **Temporal Ablation Payoff Card (`SpatioTemporalGAE` vs. `SpatialOnlyGAE` @ 50% Occlusion)** with 3 large delta bars.
* **Copy-Paste On-Slide Text:**
  * **1. Visibility-Imputed 7D Node Features:**
    $$\mathbf{x}_{t,j}^{\text{imp}} = m_{t,j}\tilde{\mathbf{x}}_{t,j} + (1 - m_{t,j})\bar{\mathbf{x}}_j^{\text{rest}}, \qquad \mathbf{f}_{t,j}^{\text{in}} = \left[\mathbf{x}_{t,j}^{\text{imp}} \;\|\; m_{t,j} \;\|\; m_{t,j}\Delta\tilde{\mathbf{x}}_{t,j}\right] \in \mathbb{R}^7$$
  * **2. 3-Layer 4-Head Mask-Aware GAT ($21$ Self + $40$ Bone + $8$ Palmar $= 69$ Directed Edges):**
    $$e_{t,i,j}^{(h)} = \frac{\mathbf{q}_{t,i}^{(h)} \cdot \mathbf{k}_{t,j}^{(h)}}{\sqrt{d_h}} + \text{GATv2}_{i,j}^{(h)} + B_{h,i,j} - \mathbf{\gamma_h(1 - m_{t,j})}$$
    * **Key Gate $-\gamma_h(1-m_{t,j})$:** Mutes outgoing messages from occluded keys ($m_{t,j}=0$) while letting occluded queries listen to visible knuckles.
  * **3. Temporal Core ($T=8$) Payoff vs. `SpatialOnlyGAE`:**
    * **Overall MPJPE (50% Occ):** $8.41\text{ mm} \to \mathbf{6.51\text{ mm}}$ (**$-22.6\%$ Error**)
    * **Occluded-Joint MPJPE:** $12.45\text{ mm} \to \mathbf{9.09\text{ mm}}$ (**$-27.0\%$ Error**)
    * **Acceleration Jitter:** $13.26 \to \mathbf{7.81\text{ mm/frame}^2}$ (**$-41.1\%$ Jitter**)

---

### SLIDE 7: Physics-Informed Knowledge Distillation & Edge Student Compression
* **Canva Layout:**
  * **Top Half:** Student Pipeline diagram + 5-Term `DistillationLoss` card.
  * **Bottom Half:** **3 Big Stat Cards** (`6.64x Smaller`, `137.3x Fewer FLOPs`, `433 KB ONNX`) + 4-column table.
* **Copy-Paste On-Slide Text:**
  * **Student Pipeline:** `(B,T,84) -> Linear(128) -> TemporalMixingBlock1D(k=3) -> ResidualMLP -> RoM Clamper + DifferentiableFK` ($78,398$ params).
  * **5-Term Kinematic Distillation Loss:**
    $$\mathcal{L}_{\text{KD}} = \mathbf{35.0}\,\mathcal{L}_{6\text{D}}(\hat{\mathbf{R}}_{\text{S}}, \mathbf{R}_{\text{T}}) + \mathbf{0.45}\,\mathcal{L}_{\text{MPJPE}}^w(\hat{\mathbf{P}}_{\text{S}}, \mathbf{P}_{\text{T}}) + \mathbf{0.55}\,\mathcal{L}_{\text{MPJPE}}^w(\hat{\mathbf{P}}_{\text{S}}, \mathbf{P}_{\text{GT}}) + \mathbf{25.0}\,\mathcal{L}_{6\text{D}}(\hat{\mathbf{R}}_{\text{S}}, \mathbf{R}_{\text{GT}}) + \mathbf{0.35}\,\mathcal{L}_{\text{smooth}}$$
    *(combines 6D MSE + $\text{SO}(3)$ geodesic angle $\arccos((\text{Tr}(\hat{\mathbf{R}}^\top\mathbf{R})-1)/2)$ + chordal distance, with $w_{\text{occ}}=1.5$).*

| Complexity Metric | Teacher `SpatioTemporalGAE` | Direct-XYZ `UnconstrainedMLP` | **Distilled `StudentKinematicMLP`** | **Compression Gain** |
| :--- | :---: | :---: | :---: | :---: |
| **Trainable Parameters** | $520,189$ | $78,527$ | **$78,398$** | **$6.64\times$ Smaller** |
| **Weights Footprint (FP32 / FP16)** | $1.99\text{ MB}$ / $1.00\text{ MB}$ | $0.30\text{ MB}$ / $0.15\text{ MB}$ | **$0.30\text{ MB}$ / $0.15\text{ MB}$** | **Fits Edge L2 Cache** |
| **FLOPs per Frame ($T=1$)** | $22.43\text{ MFLOPs}$ | $0.158\text{ MFLOPs}$ | **$0.163\text{ MFLOPs}$** | **$137.3\times$ Fewer FLOPs** |
| **ONNX Opset 17 Size & Parity** | — | $316.4\text{ KB}$ ($1.5\times 10^{-5}\text{ mm}$) | **$433.2\text{ KB}$ ($2.3\times 10^{-4}\text{ mm}$)** | **Exact ORT Parity** |

---

### SLIDE 8: Experimental Results I — Multi-Rate Occlusion Ablation (`Rubric: Results — 5 Marks`)
* **Canva Layout:**
  * **Top 58%:** Place **`fig1_occlusion_robustness.png`** (Left) and **`fig2_bone_length_and_biovalidity.png`** (Right) side-by-side at large scale (crop top Matplotlib titles by 8%).
  * **Bottom 42%:** **Focused 5-Row Summary Table** (`15pt` font, highlight bottom 2 rows in `#ECFDF5`).

| Model / Pipeline ($N_{\text{test}}=240$/tier) | 30% Occ MPJPE $\downarrow$ | **50% Occ MPJPE $\downarrow$** | **50% Occ Occluded $\downarrow$** | 70% Occ MPJPE $\downarrow$ | **Bone-Length Error $\downarrow$** | **Biomech. Validity $\uparrow$** | **Accel. Jitter ($\text{mm/frame}^2$) $\downarrow$** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corrupted Input (Gloved + Masked)** | $35.55$ | $52.08\text{ mm}$ | $100.29\text{ mm}$ | $70.38$ | $19.96\text{ mm}$ | $0.00\%$ | $25.67$ |
| **`UnconstrainedStudentMLP` (Direct XYZ)** | $8.23$ | $9.95\text{ mm}$ | $14.27\text{ mm}$ | $13.69$ | $2.31\text{ mm}$ | $29.95\%$ | $11.51$ |
| **`SpatialOnlyGAE` (GAT + FK, No GRU)** | $6.77$ | $8.41\text{ mm}$ | $12.45\text{ mm}$ | $12.58$ | **$0.0000\text{ mm}$** | **$100.0\%$** | $13.26$ |
| **`Teacher ST-GAE` (Full GAT + Bi-GRU)** | **$5.56$** | **$6.51\text{ mm}$** | **$9.09\text{ mm}$** | **$10.32$** | **$0.0000\text{ mm}$** | **$100.0\%$** | **$7.81$** |
| **`StudentKinematicMLP` (Ours Edge)** | **$5.93$** | **$7.04\text{ mm}$** | **$10.02\text{ mm}$** | **$11.15$** | **$0.0000\text{ mm}$** | **$100.0\%$** | **$8.27$** |

---

### SLIDE 9: Experimental Results II — 3D Hand Repair, Jitter & Granular FDS Analysis (`Rubric: Results — 5 Marks`)
* **Canva Layout (2x2 Visual Dashboard — Delete the text wall & let the 3 figures breathe!):**
  * **Top Half (Full Width, 48% height):** **`fig4_qualitative_3d_hand_repair.png`** stretched wide across the top (`1750 px`).
  * **Bottom Left (50% width):** **`fig7_fds_granular_and_gesture_utility.png`**.
  * **Bottom Right (50% width):** **`fig3_temporal_jitter_comparison.png`** + 1 callout pill for `custom_gloved_dataset.npz`.
* **Copy-Paste 3 Callout Badges:**
  * **Badge 1 (Top):** **3D Reconstruction @ 70% Occlusion:** Unconstrained stretches bones ($2.82\text{ mm}$ drift) vs. **Our 6D-FK (`0.0000 mm` Bone Error, `100%` Valid)**.
  * **Badge 2 (Bottom-Left):** **Distal Fingertip Error Cut by $-28.9\%$** ($16.90 \to 12.02\text{ mm}$) & **70%-Occ Gesture Retrieval Boosted $26.7\% \to 46.7\%$**.
  * **Badge 3 (Bottom-Right):** **Latex Glare Jitter Cut by $-69.6\%$** ($25.67 \to 7.81\text{ mm/frame}^2$) • **Zero-Shot Custom Gloved ($N=120$):** **$7.01\text{ mm}$** Student MPJPE.

---

### SLIDE 10: Multi-Hardware Edge Profiling & Live Demo HUD (`Rubric: Demo — 5 Marks`)
* **Canva Layout:**
  * **Top Row (35% height):** **3-Row Measured Hardware Table** + 1 compact footnote badge for Jetson Nano / Pi 5 compute envelope ($163.4\text{ KFLOPs} = 2,000\times$ lighter than MobileNetV2).
  * **Bottom Left (46% width):** **`fig5_edge_latency_pareto.png`**.
  * **Bottom Right (54% width):** **`fig6_live_demo_dashboard.png`** (or `live_demo_telemetry.gif`).

| Measured Hardware Platform | Execution Provider | Teacher ($T=8$) | **Student Latency ($T=1$ / $T=8$)** | **Student FPS** | **Bare-Metal `EdgeFK`** | **60 Hz Budget Used** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Apple MacBook Air (M-Series)** | ONNX `CoreMLExecutionProvider` | $6.03\text{ ms}$ ($166\text{ FPS}$) | **$1.242\text{ ms}$ / $1.935\text{ ms}$** | **$804.9\text{ – }4,134\text{ FPS}$** | **$0.0725\text{ ms}$ ($13.8\text{ kHz}$)** | **$7.5\%$ ($13.4\times$ Headroom)** |
| **Windows Workstation (24T x86_64)** | ONNX `CPUExecutionProvider` | $29.34\text{ ms}$ ($34\text{ FPS}$) | **$1.792\text{ ms}$ / $2.546\text{ ms}$** | **$557.9\text{ – }3,142\text{ FPS}$** | $0.199\text{ ms}$ ($5.02\text{ kHz}$) | **$10.7\%$ ($9.3\times$ Headroom)** |
| **iPhone 60 Hz ARKit UDP Stream** | `ARKitUDPStreamer` (`:5005`) + ORT | — | **$1.322\text{ ms}$ End-to-End** | **$756.4\text{ Hz}$ Max Rate** | $0.0725\text{ ms}$ | **Zero Frame Drop** |

---

### SLIDE 11: Live Demo Launcher, Team Contributions & Conclusion (`Rubric: Demo — 5 Marks`)
* **Canva Layout:**
  * **Left Column (42% — Dark Terminal Card `#0F172A`):** 1 Primary Command + interactive hotkeys (`[H]` 1/2 Hands, `[T]` Tool Box, `[O]` Occlusion Cycle).
  * **Right Column (58% — 2x2 Grid of Team Contribution Cards):** 1 card per teammate with bold quantitative impact.
* **Copy-Paste Left Terminal Card:**
  ```powershell
  # Launch Live 3-Panel Telemetry HUD
  python run_presentation_demo.py --mode replay --occlusion 0.50
  python run_presentation_demo.py --mode webcam --two-hands

  # Live HUD Hotkeys:
  [H] Toggle 1 / 2 Hands   [T] Virtual Surgical Tool Box
  [O] Cycle 0/30/50/70%    [S] Save PNG Snapshot
  ```

---

## 4. Exact Hex Colors & Canva Fast-Track Tips

* **Canvas Background:** `#FFFFFF` (Pure White) so `fig0`–`fig7` blend seamlessly with zero box seams.
* **Primary Ink:** `#0F172A` (Deep Navy Slate) for headers; `#334155` for body bullets.
* **Model Brand Colors (Matches `fig1`–`fig7`):**
  * **Student 6D-FK (Ours):** Emerald **`#10B981`** (Table row fill: `#ECFDF5`, bold text: `#047857`)
  * **Teacher ST-GAE (Ours):** Cerulean **`#0284C7`** (Table row fill: `#F0F9FF`)
  * **Unconstrained Baseline:** Warning Orange **`#F97316`** (Fill: `#FFF7ED`)
  * **Corrupted Input:** Crimson **`#E11D48`** (Fill: `#FFF1F2`)
* **Pro-Tip — Crop the Top 8% of `fig0`–`fig7` in Canva:** Every PNG in `results/figures/` has a baked-in Matplotlib title at the top. Cropping the top 8% removes the duplicate title and lets you scale every chart **20% larger**!
