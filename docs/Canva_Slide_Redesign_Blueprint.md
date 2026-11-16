# Canva Presentation Flow & Visual Redesign Blueprint (16 Slides)
**Target Defense:** Foundations of Data Science (FDS) Final Evaluation (`15 Marks` Total: Introduction `1M`, Objectives `1M`, Block Diagram `3M`, Results `5M`, Live Demo `5M`)  
**Target Duration:** 16–18 Minutes (~4 Minutes per Speaker, 16 Slides Total)  
**Team Ownership & Speakers:**
* **Abhishekh Verma (Slides 1, 6, 7, 16):** Intro, 6D Kinematics & Forward Kinematics Proof, Live Demo & Conclusion
* **Bhavya (Slides 2, 3, 4, 5):** Objectives, Literature Gap, System Block Diagram, Hybrid Data Engineering
* **Vrushank (Slides 8, 9, 10, 11):** Teacher ST-GAE, Attention Mechanisms, Knowledge Distillation Loss, Student Compression
* **Kapish (Slides 12, 13, 14, 15):** Quantitative MPJPE Benchmarks, Bone Invariance, Jitter/Hand Repair, Multi-Hardware Edge

---

## 1. Clean Presentation Choreography (4 Contiguous Blocks, Zero Dead Air)

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

## 2. Universal Canva Styling Rules (Executive Cinematic Dark Glass)

To achieve the exact Apple/OpenAI keynote visual language of our reference design:

* **Canvas Dimensions:** 1920 × 1080 px (16:9 Widescreen).
* **Base & Atmosphere:**
  * **Base Background:** Deep Pitch Black (`#000000` / `#030305`).
  * **Ambient Mesh / Radial Glows:**
    * **Electric Cobalt Blue Aura:** `#2563EB` / `#1E40AF` radiating across the lower-left to mid-left quadrant (opacity ~40–50%, blur ~120px).
    * **Subtle Cyan Accent:** `#38BDF8` / `#06B6D4` highlighting technical elements and kickers.
    * **Warm Golden Amber / Orange Arc:** `#F59E0B` / `#D97706` sweeping across the mid-right quadrant (opacity ~40–48%, blur ~140px).
    * **Top Vignette:** Pitch black `#000000` across the header region ensuring 100% pure contrast for slide titles.
* **Typography Hierarchy:**
  * **Font Family:** Modern geometric sans-serif (Plus Jakarta Sans, Inter, Montserrat, or Segoe UI).
  * **Slide Title:** `26pt–30pt` Extra Bold / Black (`#FFFFFF`), uppercase, letter-spacing `+0.05em`.
  * **Title Accent Bar:** Sleek warm gold/amber horizontal bar (`#F59E0B`, width `60–75px`, height `3.5–4px`) directly beneath the first word of the title.
  * **Slide Kicker:** `9pt–10pt` Bold Uppercase (`#38BDF8`), letter-spacing `+0.08em` (`FOUNDATIONS OF DATA SCIENCE • FINAL EVALUATION`).
  * **Slide Badge:** Top-right glass pill (`#0F172A` with `1px solid rgba(255, 255, 255, 0.15)`, text `#F1F5F9`).
  * **Section Subtitle:** `12pt–14pt` Medium (`#94A3B8`).
  * **Numbered Items:** Large modern numbers (`01`, `02`, `03`, `04`, `05`) in `30pt–36pt` Bold (`#FFFFFF`), paired with clean thin divider lines (`1px solid rgba(255, 255, 255, 0.18)`).
  * **Body / Bullet Text:** `12pt–14pt` Regular (`#CBD5E1`), strong text in `#FFFFFF`.
* **Dark Glass Cards & Containers:**
  * **Card Fills:** `rgba(15, 23, 42, 0.70)` with `backdrop-filter: blur(8px)`.
  * **Card Border:** `1px solid rgba(255, 255, 255, 0.12)`, corner radius `8px`.
  * **Card Shadow:** `0 10px 30px rgba(0, 0, 0, 0.5), 0 0 15px rgba(37, 99, 235, 0.10)`.
* **Tables in Sleek Dark Glass:**
  * Container in dark glass with rounded corners.
  * Header in deep navy (`rgba(10, 15, 30, 0.95)`), cyan text (`#38BDF8`), and a gold (`#F59E0B`) underline.
  * **Teacher ST-GAE** & **StudentKinematicMLP** rows highlighted in glowing mint green (`rgba(16, 185, 129, 0.18)` fill, `#10B981` border, `#34D399` bold text).
  * **Unconstrained baseline** row highlighted in amber warning (`rgba(245, 158, 11, 0.12)`).
* **Figures & Image Embeds:**
  * Centered in dark glass frames with cyan/blue rim glow (`box-shadow: 0 10px 30px rgba(0,0,0,0.6), 0 0 20px rgba(37, 99, 235, 0.12)`).
  * Captions in `9pt–10pt` Medium Gray (`#94A3B8`).

---

## 3. Slide-by-Slide Canva Layout & Asset Guide

### Slide 1: Introduction — Problem Context, Motivation & Team Roles (Abhishekh)
* **Visual Structure:** Problem Context + Solution Summary + Team Ownership Table.
  * **Top Section:** 2 Failure Mode callouts (Surgical/Tool Occlusion with $15\text{–}27\text{ mm}$ bone stretch, Latex Glove glare with $\sigma = 4.5\text{ mm}$ noise) + Our Kinematic Solution.
  * **Bottom Section:** 4-row Team Ownership Table in sleek dark glass (`Abhishekh Verma`, `Bhavya`, `Vrushank`, `Kapish`) with cyan headers and gold accent.

### Slide 2: Five Core Research Objectives (Bhavya) — Exact Match to Reference Image
* **Visual Structure:** 2-Column Numbered Objectives Grid.
  * **Left Column:** Objectives `01`, `03`, `05`.
  * **Right Column:** Objectives `02`, `04`.
  * Each item features:
    * Large bold white number (`32pt` Bold, `#FFFFFF`).
    * 2-line title in crisp off-white (`16pt` SemiBold, `#F8FAFC`).
    * 1-line concise technical description underneath (`10pt` Regular, `#94A3B8`).
    * Clean horizontal divider line (`1px solid rgba(255, 255, 255, 0.18)`) beneath each item.
  * **Item Text:**
    * **01:** Hybrid Real FreiHAND + Clinical / MANO Benchmark
    * **02:** Zero-Bone-Error Differentiable Forward / Kinematics
    * **03:** Mask-Aware Spatio-Temporal Graph / Autoencoder Teacher
    * **04:** Knowledge Distillation into a Lightweight Edge / Student
    * **05:** Real-Time Edge Deployment & Live / Telemetry Demo

### Slide 3: State-of-the-Art Literature Gap & Comparative Matrix (Bhavya)
* **Visual Structure:** Bullet Flaws + 7-Row Comparative Matrix.
  * Comparative Table: HandGCN, HOPE-Net, Spurr et al., HandOccNet, Unconstrained Student, Teacher ST-GAE, StudentKinematicMLP.
  * Highlight Teacher and Student rows in glowing emerald mint green (`rgba(16, 185, 129, 0.18)`).
  * Bold callout on **$0.0000\text{ mm}$ bone error** vs. competitors' $0.80\text{–}4.80\text{ mm}$.

### Slide 4: System Architecture & End-to-End Block Diagram (Bhavya)
* **Visual Structure:** Diagram Showcase + 4 Horizontal Stage Cards.
  * Center: Embed `results/figures/fig0_block_diagram.png` in a dark glass card spanning 86% slide width.
  * Bottom Strip: 4 horizontal cards (`01 Data Ingestion`, `02 Teacher ST-GAE`, `03 Student Distillation`, `04 Hard Kinematic Manifold`) with gold top borders.

### Slide 5: Data Engineering: Hybrid FreiHAND+MANO Pipeline & Transforms (Bhavya)
* **Visual Structure:** 2-Column Engineering Pipeline layout.
  * Left: Hybrid Dataset Card ($32,560$ real FreiHAND poses + 6 clinical gestures, 80/20 train/test split).
  * Right: 2 Physics Transform cards with styled math equations:
    * `SyntheticGloveTransform`: $12\%$ elastic resistance equation + AR(1) specular noise ($\rho=0.65$).
    * `SyntheticOcclusionTransform`: 3D cylindrical clipping ($R=28\text{ mm}$) with distal-to-proximal priority.

### Slide 6: 6D Continuous Rotations & Anatomical RoM Clamping (Abhishekh)
* **Visual Structure:** Math Formulation + Anatomical Limits Table.
  * Top: 6D Gram-Schmidt $\text{SO}(3)$ continuous mapping formulas and $\tanh$ RoM clamping equation in gold-bordered glass math cards.
  * Bottom: Anatomical joint boundary table (Thumb Ray, MCP, PIP, DIP) in dark glass. Highlight zero dorsal hyperextension on PIP and DIP hinges.

### Slide 7: Differentiable Forward Kinematics & Theorem 1 Proof (Abhishekh)
* **Visual Structure:** Hierarchical Kinematic Tree + Formal Theorem 1 Card.
  * Top: `PARENT_INDICES` tree traversal formula.
  * Center: Dedicated **Theorem 1 Proof Card** with warm gold left border:
    $$\|\mathbf{P}_j - \mathbf{P}_{p(j)}\|_2 = \|\mathbf{R}_j^{\text{global}} \mathbf{v}_j^{\text{rest}}\|_2 = \sqrt{(\mathbf{v}_j^{\text{rest}})^\top \mathbf{I}_3 \mathbf{v}_j^{\text{rest}}} = L_j \implies \varepsilon_{\text{bone}} \equiv 0.0000\text{ mm} \quad \blacksquare$$
  * Bottom: Bare-metal NumPy `EdgeFK` speed badge: **$0.0725\text{ ms}$ ($13,790\text{ Hz}$)**.

### Slide 8: Teacher Architecture: Spatio-Temporal Graph Autoencoder (Vrushank)
* **Visual Structure:** 2-Column Deep Architecture breakdown.
  * Left: 7D Imputed Input formulation ($[\mathbf{x}^{\text{imp}} \,\|\, m \,\|\, m(\mathbf{x}-\bar{\mathbf{x}}^{\text{rest}})]$) + learned joint-identity embedding $\mathbf{e}_j^{\text{joint}}$.
  * Right: 69-Edge Anatomical Skeleton Graph (21 self-loops + 20 bone edges + 4 inter-MCP palmar arch edges). Explain palmar arch feature borrowing.

### Slide 9: Spatial Graph Attention & Temporal Dilated Convolutions (Vrushank)
* **Visual Structure:** Attention Mechanism Card + Temporal Recurrence Gain.
  * Top: Visibility Key Penalty equation ($-\gamma_h(1-m_{t,j})$).
  * Bottom: Bi-GRU recurrence gain: cuts 50% occlusion error by $22.6\%$ ($8.41 \to 6.51\text{ mm}$), occluded phalanx error by $27.0\%$, and acceleration jitter by $41.1\%$.

### Slide 10: Physics-Informed Knowledge Distillation Strategy (Vrushank)
* **Visual Structure:** 5-Term Multi-Objective Loss Breakdown.
  * Master equation at top with underbraces:
    $$\mathcal{L}_{\text{KD}} = 35.0 \mathcal{L}_{6\text{D}}^{\text{Teacher}} + 0.45 \mathcal{L}_{\text{MPJPE}}^{\text{Teacher}} + 0.55 \mathcal{L}_{\text{MPJPE}}^{\text{GT}} + 25.0 \mathcal{L}_{6\text{D}}^{\text{GT}} + 0.35 \mathcal{L}_{\text{smooth}}$$
  * 3 Visual Pillars underneath: Geodesic $\text{SO}(3)$ distance, $1.5\times$ occlusion-weighted spatial loss, and temporal smoothness.

### Slide 11: Edge Student Architecture & Compression Analysis (Vrushank)
* **Visual Structure:** Student Architecture Flow + Complexity Table.
  * Left: Lightweight pipeline: Flat 84D projection $\to$ 1D depthwise convolution $\to$ Residual MLP $\to$ Shared Forward Kinematics.
  * Right: Compression comparison table showing $6.64\times$ parameter reduction ($520\text{K} \to 78.4\text{K}$), $137.3\times$ FLOPs compression, and $231\text{ nm}$ ONNX numerical parity.

### Slide 12: Quantitative Occlusion Robustness (30%, 50%, 70% MPJPE) (Kapish)
* **Visual Structure:** Side-by-Side Table & Chart Dashboard.
  * Left (57% width): 15-row Multi-Rate MPJPE Table in dark glass with mint green highlighted Student and Teacher rows.
  * Right (43% width): High-resolution embed of `results/figures/fig1_occlusion_robustness.png` in a dark glass card.

### Slide 13: Anatomical Invariance & Zero Bone Length Violation (Kapish)
* **Visual Structure:** Side-by-Side Chart & Empirical Findings.
  * Left (48% width): `results/figures/fig2_bone_length_and_biovalidity.png`.
  * Right (52% width): Empirical verification cards:
    * Unconstrained Regressor Failure: $2.31\text{ mm}$ mean bone error, $27.0\text{ mm}$ peak stretching, validity drops to $24.95\%$.
    * Kinematic Guarantee: Exactly **$0.0000\text{ mm}$ bone error** and **$100.00\%$ validity** across all 720 test sequences.

### Slide 14: Temporal Jitter Suppression & Qualitative 3D Hand Repair (Kapish)
* **Visual Structure:** Dual Image Comparative Showcase.
  * Left Card: `results/figures/fig3_temporal_jitter_comparison.png` (Teacher and Student cut acceleration jitter by $\sim 68\%$).
  * Right Card: `results/figures/fig4_qualitative_3d_hand_repair.png` (3D skeleton reconstruction under 50% and 70% tool occlusion).

### Slide 15: Downstream Gesture Utility & Multi-Hardware Edge Telemetry (Kapish)
* **Visual Structure:** Hardware Benchmark Table + Dual Research Figures.
  * Top: Hardware Benchmark Table in dark glass (MacBook Air $1.24\text{ ms}$ / $805\text{ FPS}$, PC $1.79\text{ ms}$ / $558\text{ FPS}$, iPhone 60 Hz zero-drop).
  * Bottom: `results/figures/fig7_fds_granular_and_gesture_utility.png` and `results/figures/fig5_edge_latency_pareto.png` side-by-side.

### Slide 16: Interactive Live Telemetry Demonstration & Project Conclusion (Abhishekh)
* **Visual Structure:** Live Demo HUD Showcase + Interactive Controls.
  * Left Side: `results/figures/fig6_live_demo_dashboard.png` in dark glass card + PowerShell launcher box:
    ```powershell
    .\.venv\Scripts\python.exe run_presentation_demo.py
    ```
  * Right Side: Interactive Keyboard Controls (`[O]` cycle occlusion, `[H]` toggle 2-hand tracking, `[T]` toggle 3D trails, `[M]` switch stream) + Key Project Conclusions.
