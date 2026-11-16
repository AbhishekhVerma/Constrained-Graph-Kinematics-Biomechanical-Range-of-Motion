# Canva Presentation Flow & Visual Redesign Blueprint (16 Slides)
**Target Defense:** Foundations of Data Science (FDS) Final Evaluation (`15 Marks` Total: Introduction `1M`, Objectives `1M`, Block Diagram `3M`, Results `5M`, Live Demo `5M`)  
**Target Duration:** 16–18 Minutes (~4 Minutes per Speaker, 16 Slides Total)  
**Team Ownership & Speakers:**
* **Abhishekh Verma (Slides 1, 6, 7, 16):** Intro, 6D Kinematics & Forward Kinematics Proof, Live Demo & Conclusion
* **Bhavya (Slides 2, 3, 4, 5):** Objectives, Literature Gap, System Block Diagram, Hybrid Data Engineering
* **Vrushank (Slides 8, 9, 10, 11):** Teacher ST-GAE, Attention Mechanisms, Knowledge Distillation Loss, Student Compression
* **Kapish (Slides 12, 13, 14, 15):** Quantitative MPJPE Benchmarks, Bone Invariance, Jitter/Hand Repair, Multi-Hardware Edge

---

## 1. Clean Presentation Choreography (4 Clean Blocks, Zero Dead Air)

Passing the microphone randomly between slides wastes precious presentation time and confuses the audience. We use 4 clean, contiguous speaker segments totaling ~17 minutes:

```
[0:00 - 1:05] Abhishekh (S1: Intro, Problem & Team)
    └── Handover 1: "I hand over to Bhavya to present our objectives and data pipeline."
[1:05 - 5:20] Bhavya (S2: Objectives, S3: Literature Gap, S4: Block Diagram, S5: Data Pipeline)
    └── Handover 2: "I hand back to Abhishekh to explain the mathematical foundation of our kinematic manifold."
[5:20 - 7:35] Abhishekh (S6: 6D Rotations & RoM, S7: Forward Kinematics & Theorem 1 Proof)
    └── Handover 3: "I hand over to Vrushank to explain our Spatio-Temporal Graph Autoencoder Teacher."
[7:35 - 11:35] Vrushank (S8: Teacher ST-GAE, S9: Mask-Aware GAT, S10: Distillation Loss, S11: Student Architecture)
    └── Handover 4: "I hand over to Kapish to walk through our experimental evaluation and edge benchmarks."
[11:35 - 15:35] Kapish (S12: MPJPE Benchmark, S13: Bone Invariance, S14: Jitter/3D Repair, S15: Edge Hardware)
    └── Handover 5: "I hand back to Abhishekh for our interactive live demonstration and conclusion."
[15:35 - 17:05] Abhishekh (S16: Live 3-Panel HUD Demo, Key Takeaways & Defense Q&A)
```

---

## 2. Universal Canva Styling Rules (16:9 Widescreen)

* **Canvas Dimensions:** 1920 × 1080 px (16:9 Widescreen).
* **Color Palette:**
  * **Background:** Clean White (`#FFFFFF`) or Slate Dark (`#0F172A`).
  * **Card Fills:** Soft Off-White (`#F8FAFC`) with subtle border (`#E2E8F0`).
  * **Primary Accent (Trust / Tech):** Deep Indigo (`#4338CA`) or Electric Blue (`#2563EB`).
  * **Success Accent (Kinematics / Valid):** Emerald Mint (`#10B981` / `#ECFDF5`).
  * **Warning Accent (Unconstrained / Failure):** Amber Orange (`#F59E0B` / `#FEF3C7`).
  * **Error / Occlusion Accent:** Rose Crimson (`#E11D48` / `#FFF1F2`).
* **Typography Hierarchy:**
  * **Slide Title:** `28pt–32pt` Bold (Sans-serif: Montserrat, Inter, or Segoe UI).
  * **Subtitle / Section Tag:** `14pt–16pt` Semibold Uppercase with tracking (+50).
  * **Card Headings:** `18pt–20pt` Bold.
  * **Body / Bullet Text:** `14pt–16pt` Regular (never drop below `14pt` on a projector!).
  * **Key Stat Callouts:** `36pt–48pt` Extra Bold with small `12pt` unit label underneath.

---

## 3. Slide-by-Slide Canva Layout & Asset Guide

### Slide 1: Introduction — Problem Context, Motivation & Team Roles (Abhishekh)
* **Visual Structure:** Split 2-Column Hero layout.
  * **Left Column (50%):** 2 Failure Mode cards (Rose `#FFF1F2` for tool occlusion, Amber `#FEF3C7` for latex glare) + 1 emerald solution pill.
  * **Right Column (50%):** Hero GIF/Video embed (`results/figures/live_demo_telemetry.gif`) with 3 stat badges underneath (`0.0000 mm` Bone Error, `805 FPS` CoreML, `100%` Valid).
  * **Bottom Footer:** 4 Team Role chips (Abhishekh, Bhavya, Vrushank, Kapish).

### Slide 2: Five Core Research Objectives (Bhavya)
* **Visual Structure:** 5-Box Horizontal / Staggered Cards (Numbered 1 to 5).
  * Each card has a colorful number badge (`01` to `05`), a bold title, and 2 concise bullet points.
  * Card 1: Hybrid Real Data (FreiHAND + MANO)
  * Card 2: Hard Forward Kinematics (0.0000 mm error)
  * Card 3: Spatio-Temporal Graph Autoencoder (Visibility GAT + Bi-GRU)
  * Card 4: Knowledge Distillation (78.4K Student, 137x fewer FLOPs)
  * Card 5: Real-Time Edge Telemetry (>500 FPS on Apple Silicon & PC)

### Slide 3: State-of-the-Art Literature Gap & Comparative Matrix (Bhavya)
* **Visual Structure:** Clean Comparison Table (7 rows × 7 columns).
  * Top competitors: HandGCN, HOPE-Net, Spurr et al., HandOccNet, Unconstrained Student.
  * Bottom 2 rows highlighted in Emerald Green (`#ECFDF5` fill, `#10B981` border): `Teacher ST-GAE` and `StudentKinematicMLP`.
  * Highlight the **Bone-Length Error** column (competitors at $0.80\text{–}4.80\text{ mm}$ vs. our **$0.0000\text{ mm}$**).

### Slide 4: System Architecture & End-to-End Block Diagram (Bhavya)
* **Visual Structure:** Full-slide High-Resolution Diagram Showcase.
  * Place `results/figures/fig0_block_diagram.png` in the center spanning 85% of slide width.
  * Add 4 clean corner annotations highlighting Stage 1 (Data), Stage 2 (Teacher), Stage 3 (Student Distillation), and Stage 4 (Hard Kinematic Manifold).

### Slide 5: Data Engineering: Hybrid FreiHAND+MANO Pipeline & Transforms (Bhavya)
* **Visual Structure:** 2-Column Engineering Pipeline layout.
  * **Left Column:** Hybrid Dataset Card ($32,560$ real FreiHAND poses + 6 clinical MANO gestures, zero-leakage 80/20 train/test split).
  * **Right Column:** 2 Physics Transform cards:
    * `SyntheticGloveTransform`: $12\%$ elastic resistance equation + AR(1) specular noise ($\rho=0.65$) + $12\text{ mm}$ glare spikes.
    * `SyntheticOcclusionTransform`: 3D cylindrical clipping ($R=28\text{ mm}$) with distal-to-proximal priority (TIP $\to$ DIP $\to$ PIP $\to$ MCP).

### Slide 6: 6D Continuous Rotations & Anatomical RoM Clamping (Abhishekh)
* **Visual Structure:** Mathematical Formulation + Anatomical Limits Table.
  * **Left Side:** Why 6D? (Euler Gimbal lock & Quaternion antipodal ambiguity) + Gram-Schmidt $\mathbb{R}^6 \to \text{SO}(3)$ formulas.
  * **Right Side:** $C^1$-smooth $\tanh$ RoM clamping equation + anatomical joint boundary table (PIP/DIP hinges clamped to $[0^\circ, 110^\circ]$ and $[0^\circ, 80^\circ]$ with zero hyperextension).

### Slide 7: Differentiable Forward Kinematics & Theorem 1 Proof (Abhishekh)
* **Visual Structure:** Kinematic Tree Diagram + Formal Theorem Card.
  * **Left Side:** Directed tree traversal diagram showing parent indices from wrist to fingertips.
  * **Right Side:** Theorem 1 proof card: $\|\mathbf{P}_j - \mathbf{P}_{p(j)}\|_2 = \|\mathbf{R}_j^{\text{global}} \mathbf{v}_j^{\text{rest}}\|_2 = \sqrt{(\mathbf{v}_j^{\text{rest}})^\top \mathbf{I}_3 \mathbf{v}_j^{\text{rest}}} = L_j \implies \varepsilon_{\text{bone}} \equiv 0.0000\text{ mm}$.
  * **Stat Callout:** Bare-metal NumPy `EdgeFK` runs in **$0.0725\text{ ms}$ ($13,790\text{ Hz}$)**.

### Slide 8: Teacher Architecture: Spatio-Temporal Graph Autoencoder (Vrushank)
* **Visual Structure:** 2-Column Deep Architecture breakdown.
  * **Left Side:** 7D Imputed Input formulation ($[\mathbf{x}^{\text{imp}} \,\|\, m \,\|\, m(\mathbf{x}-\bar{\mathbf{x}}^{\text{rest}})]$) + Joint Identity Embedding $\mathbf{e}_j^{\text{joint}}$.
  * **Right Side:** 69-Edge Anatomical Skeleton Graph: 21 self-loops + 20 bone edges + 4 inter-MCP palmar arch edges. Explain why palmar arch edges allow occluded fingers to borrow context from visible knuckles.

### Slide 9: Spatial Graph Attention & Temporal Dilated Convolutions (Vrushank)
* **Visual Structure:** Attention Mechanism Card + Temporal Recurrence Gain.
  * **Top / Left:** Visibility Key Penalty equation ($-\gamma_h(1-m_{t,j})$). Explain the "muted microphone" concept.
  * **Bottom / Right:** Temporal Bi-GRU ablation comparison (`SpatialOnlyGAE` $8.41\text{ mm}$ vs. `Teacher ST-GAE` **$6.51\text{ mm}$**, showing $22.6\%$ error reduction and $41.1\%$ jitter reduction).

### Slide 10: Physics-Informed Knowledge Distillation Strategy (Vrushank)
* **Visual Structure:** 5-Term Multi-Objective Loss Breakdown.
  * Master equation at top: $\mathcal{L}_{\text{KD}} = 35.0 \mathcal{L}_{6\text{D}}^{\text{Teacher}} + 0.45 \mathcal{L}_{\text{MPJPE}}^{\text{Teacher}} + 0.55 \mathcal{L}_{\text{MPJPE}}^{\text{GT}} + 25.0 \mathcal{L}_{6\text{D}}^{\text{GT}} + 0.35 \mathcal{L}_{\text{smooth}}$.
  * 3 Visual Pillars underneath:
    * Geodesic $\text{SO}(3)$ distance: $\arccos((\text{Tr}(\hat{\mathbf{R}}^\top\mathbf{R})-1)/2)$.
    * Occlusion-weighted spatial loss: $1.5\times$ penalty on occluded joints.
    * Temporal velocity & acceleration smoothness.

### Slide 11: Edge Student Architecture & Compression Analysis (Vrushank)
* **Visual Structure:** Student Architecture Flow + Compression Table.
  * **Left Side:** Lightweight pipeline: Flat 84D projection $\to$ 1D depthwise convolution $\to$ Residual MLP $\to$ Shared Forward Kinematics.
  * **Right Side:** Compression stat grid:
    * **$6.64\times$ smaller** ($520\text{K} \to 78.4\text{K}$ parameters)
    * **$137.3\times$ fewer FLOPs** ($22.43\text{M} \to 0.163\text{M}$)
    * **$155\text{ KB}$ FP16** (fits entirely in CPU L2 cache)
    * **$231\text{ nm}$ ONNX parity**

### Slide 12: Quantitative Occlusion Robustness (30%, 50%, 70% MPJPE) (Kapish)
* **Visual Structure:** Split Results Table & Chart.
  * **Top / Left:** 3-Tier Multi-Rate Occlusion Table showing Corrupted Input, Unconstrained Student, Spatial-Only GAE, Teacher ST-GAE, and Student Kinematic MLP.
  * **Bottom / Right:** High-resolution embed of `results/figures/fig1_occlusion_robustness.png` showing how the distilled Student closely hugs the Teacher curve.

### Slide 13: Anatomical Invariance & Zero Bone Length Violation (Kapish)
* **Visual Structure:** Chart Showcase & Failure Analysis.
  * Embed `results/figures/fig2_bone_length_and_biovalidity.png` spanning 55% width.
  * Side-by-side callout cards:
    * Unconstrained Regressor Failure: $2.31\text{ mm}$ mean bone error, up to $27.0\text{ mm}$ stretching, validity collapses to $24.95\%$.
    * Kinematic Guarantee: Exactly **$0.0000\text{ mm}$ bone error** and **$100.00\%$ validity** across all 720 test sequences.

### Slide 14: Temporal Jitter Suppression & Qualitative 3D Hand Repair (Kapish)
* **Visual Structure:** 2-Image Comparative Showcase.
  * **Left Side:** `results/figures/fig3_temporal_jitter_comparison.png` showing how Teacher and Student cut acceleration jitter by $\sim 68\%$ ($25.67 \to 8.27\text{ mm/frame}^2$).
  * **Right Side:** `results/figures/fig4_qualitative_3d_hand_repair.png` showing 3D skeleton reconstructions under 50% and 70% tool occlusion (unconstrained collapse vs. kinematic restoration).

### Slide 15: Downstream Gesture Utility & Multi-Hardware Edge Telemetry (Kapish)
* **Visual Structure:** FDS Analysis & Hardware Benchmark Grid.
  * **Left Side:** `results/figures/fig7_fds_granular_and_gesture_utility.png` (Phalanx depth error, $-28.9\%$ distal tip error reduction, and gesture classification utility).
  * **Right Side:** Multi-Hardware Benchmark Table & `results/figures/fig5_edge_latency_pareto.png` (MacBook Air $1.24\text{ ms}$ / $805\text{ FPS}$, PC $1.79\text{ ms}$ / $558\text{ FPS}$, Bare-metal FK $13.8\text{ kHz}$).

### Slide 16: Interactive Live Telemetry Demonstration & Project Conclusion (Abhishekh)
* **Visual Structure:** Live Demo HUD Showcase + Conclusion Highlights.
  * Embed `results/figures/fig6_live_demo_dashboard.png` in the center.
  * 3 Panel Callouts: Red (Corrupted), Orange (Unconstrained), Green (Kinematic Student).
  * Command Box: `.\.venv\Scripts\python.exe run_presentation_demo.py`
  * Keyboard Hotkeys: `[O]` cycle occlusion, `[H]` toggle 2-hand tracking, `[T]` toggle 3D trails.
  * Final takeaway bullet: Hard kinematic constraints eliminate disarticulation under extreme occlusion at 800+ FPS on edge hardware.
