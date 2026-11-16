# Constrained Graph Kinematics — 1-Page Examiner Cheat Sheet
**FDS Final Project Defense — November 17, 2026**  
*Team: Abhishekh Verma, Bhavya, Vrushank, Kapish*

---

## 1. Key Metrics at a Glance (The Numbers to Quote)

| Metric | Unconstrained Baseline | Our Distilled Student (6D-FK) | Teacher ST-GAE | Factor / Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Parameters** | 78,527 (316 KB) | **78,398 (322 KB)** | 520,189 (2.08 MB) | **6.64× smaller than Teacher** |
| **ONNX File Size** | 316.5 KB | **433.2 KB (Opset 17)** | 1.98 MB | Self-contained, zero PyG extensions |
| **Compute FLOPs** | 163.5 KFLOPs/frame | **163.4 KFLOPs/frame** | 22.44 MFLOPs/frame | **137.3× compute reduction** |
| **MPJPE (30% Tool Occ.)**| 7.12 mm | **5.93 mm** | 5.56 mm | **16.7% error reduction** |
| **MPJPE (50% Tool Occ.)**| 9.95 mm | **7.04 mm** | 6.51 mm | **29.2% error reduction** |
| **MPJPE (70% Tool Occ.)**| 14.88 mm | **11.15 mm** | 10.32 mm | **25.1% error reduction** |
| **Bone-Length Error** | 2.31 mm (spikes > 6 mm) | **0.0000 mm (Exact)** | **0.0000 mm (Exact)** | **Theorem 1 Invariant ($\|P_j - P_p\| \equiv L_j$)** |
| **Biomechanical Validity**| 29.95% (collapses) | **100.0% (Guaranteed)** | **100.0% (Guaranteed)** | **tanh RoM Clamper** |
| **MacBook Air CoreML** | 1.18 ms (847 FPS) | **1.242 ms (804.9 FPS)** | 5.045 ms (198 FPS) | **13.4× headroom** over 60 Hz budget |
| **Windows PC x86 CPU** | 1.62 ms (617 FPS) | **1.792 ms (557.9 FPS)** | 14.82 ms (67 FPS) | Pure CPUExecutionProvider |
| **NumPy EdgeFK Engine** | N/A | **0.0725 ms (13.79 kHz)** | N/A | Bare-metal ARM64 vectorized |
| **Mode B UDP Stream** | N/A | **1.322 ms End-to-End** | N/A | Sustains **756.2 Hz** max stream (Port 5005) |

---

## 2. The 5 Core Mathematical Formulations

1. **Visibility Gated Soft Attention (Slide 6):**
   $$\mathbf{k}_{v,t} \leftarrow \mathbf{k}_{v,t} \cdot \gamma_h(m_{v,t}), \quad \alpha_{uv} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{q}_u \,\|\, \mathbf{k}_v]\right)\right)}{\sum_{k \in \mathcal{N}_u} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{q}_u \,\|\, \mathbf{k}_k]\right)\right)}$$
   *Purpose:* Prevents zeroed coordinates of occluded joints from corrupting visible joints in the softmax denominator.

2. **Gram-Schmidt 6D Continuous Rotations on $\text{SO}(3)$ (Slide 5):**
   $$\mathbf{R} = [\mathbf{c}_1, \mathbf{c}_2, \mathbf{c}_3], \quad \mathbf{c}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|}, \quad \mathbf{c}_2 = \frac{\mathbf{a}_2 - (\mathbf{c}_1^\top \mathbf{a}_2)\mathbf{c}_1}{\|\mathbf{a}_2 - (\mathbf{c}_1^\top \mathbf{a}_2)\mathbf{c}_1\|}, \quad \mathbf{c}_3 = \mathbf{c}_1 \times \mathbf{c}_2$$
   *Purpose:* Continuous topology on $\mathbb{R}^6$, zero gimbal lock, zero quaternion antipodal ambiguities.

3. **Hyperbolic Tangent Range-of-Motion (RoM) Clamper (Slide 5):**
   $$\theta_{\text{clamped}} = \frac{\theta_{\max} + \theta_{\min}}{2} + \frac{\theta_{\max} - \theta_{\min}}{2} \tanh\left(\frac{2\theta - (\theta_{\max} + \theta_{\min})}{\theta_{\max} - \theta_{\min}}\right)$$
   *Purpose:* $C^1$-smooth, non-zero gradients everywhere, strictly confines joint angles to anatomical limits.

4. **Analytical Forward Kinematics Invariant (Theorem 1, Slide 5):**
   $$\mathbf{P}_j = \mathbf{P}_{p(j)} + \mathbf{R}_j^{\text{global}} (L_j \hat{\mathbf{d}}_j) \implies \|\mathbf{P}_j - \mathbf{P}_{p(j)}\| = \|L_j \hat{\mathbf{d}}_j\| = L_j \implies \text{Error} \equiv 0.0000\text{ mm}$$

5. **Multi-Objective Distillation Loss (Slide 7):**
   $$\mathcal{L}_{\text{distill}} = \lambda_{\text{rot}} \|\mathbf{R}_{6\text{d}}^S - \mathbf{R}_{6\text{d}}^T\|_1 + \lambda_{\text{geo}} \mathcal{L}_{\text{geo}}(\mathbf{R}^S, \mathbf{R}^T) + \lambda_{\text{pos}} \|\mathbf{P}^S - \mathbf{P}^T\|_1 + \lambda_{\text{temp}} \mathcal{L}_{\text{temp}} + \lambda_{\text{reg}} \|\Theta\|_2^2$$

---

## 3. Dataset & Empirical Integrity
- **Real Markers:** $32,560$ real 3D markerless poses from University of Freiburg `FreiHAND_pub_v2` streamed via HTTP range headers (`data/raw/freihand_xyz.json`).
- **Clinical Synthetics:** Articulated MANO trajectories with 6 surgical gesture primitives (Surgical Pinch, Scalpel Grasp, Hemostat Clamp, Suture Pull, Finger Tap, Fist Flexion).
- **Split:** Strict 80/20 train/test isolation persisted in `data/splits/split_indices.json` ($960$ train / $240$ test sequences per occlusion tier, total $720$ multi-rate test sequences).
- **Glove Physics:** `SyntheticGloveTransform` models $12\%$ latex stiffness, $\rho=0.65$ AR(1) specular noise ($\sigma=4.5\text{ mm}$), and $12.0\text{ mm}$ distal spikes.

---

## 4. Live Demo 1-Click Operations & Hotkeys

- **Windows Launch:** Double-click `run_demo.bat` -> Press `1` for Instant Offline Replay (50% Occlusion).
- **macOS / Linux Launch:** `./run_demo.sh` -> Press `1`.
- **Interactive Hotkeys inside HUD Window:**
  - `T`: Toggle virtual surgical tool box obstacle across phalanges.
  - `O`: Cycle occlusion rate through **0% (Pure Camera) -> 30% -> 50% -> 70%**.
  - `S`: Save 300-DPI high-res snapshot to `results/figures/live_interactive_snapshot.png`.
  - `Q` / `ESC`: Exit demo cleanly.
- **Failover Plan:** If classroom Wi-Fi fails or room lighting causes camera drops, **Option 1 (Offline Replay)** runs fully offline with deterministic 60 FPS surgical telemetry.
