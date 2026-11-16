# Feynman Technique Master Guide (Zero-to-Hero): Every Term, Number, Concept & Equation
**Project:** Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry  
**Team:** Abhishekh Verma, Bhavya, Vrushank, Kapish  
**Course:** Foundations of Data Science (FDS) — Presentation Date: November 17, 2026

> **Goal of this guide:** Never use a big word or a formula without explaining *what it actually means in plain English*, *why we need it*, and *what every single number represents*. Read Part 0 first to master every core term, then walk through Slides 1 to 11.

---

## Part 0: Zero-to-Hero Glossary (Every Key Term & Metric Explained)

### 1. What is "Kinematics" and "Forward Kinematics (FK)" vs. "Inverse Kinematics (IK)"?
* **Kinematics** is simply the geometry of motion—tracking where things are and how joints rotate, without worrying about forces or muscles.
* **Forward Kinematics (FK):** Imagine a robot arm or a human hand puppet made of rigid metal rods (bones) connected by hinges (knuckles). If you **know the angle of every hinge** (e.g., "bend the index knuckle $45^\circ$, bend the middle joint $30^\circ$") and you know the fixed length of each rod, **Forward Kinematics** calculates the exact $(x, y, z)$ position of the fingertip by starting at the wrist and stepping bone-by-bone outward.
  * *Why we love FK:* Because the rods are rigid, **it is impossible for a bone to stretch or shrink**.
* **Inverse Kinematics (IK):** The reverse puzzle! If you **know where the $(x, y, z)$ joints are** in 3D space (for example, from the FreiHAND dataset), **Inverse Kinematics** works backward to figure out what rotation angles at each knuckle produced that pose. (We use Rodrigues IK in Stage 1 to turn FreiHAND 3D positions into ground-truth joint angles for training).

### 2. What is an "Ablation Study" (and Why Do We Have `SpatialOnlyGAE` and `UnconstrainedStudentMLP`)?
* **Plain English:** Suppose you build a race car with both a **turbocharger** and **racing tires**, and it goes 200 km/h. If you want to *prove* to a skeptic that the turbo actually helped, you take the turbo out, leave everything else identical, and test the car again. That is an **ablation**—removing one specific ingredient from a system to measure exactly how much that single piece contributes.
* **Our Two Ablations:**
  1. **`UnconstrainedStudentMLP` (Removes Kinematics):** Has the exact same neural network brain (`~78.5K` parameters) as our Student, *except* we rip out the 6D rotations, RoM clamper, and Forward Kinematics, forcing it to guess raw $(x, y, z)$ coordinates. Result: bone error jumps from $0.0000\text{ mm} \to 2.31\text{ mm}$, and validity crashes from $100\% \to 29.95\%$! This proves our kinematic layer is what makes hands anatomically valid.
  2. **`SpatialOnlyGAE` (Removes Time Memory):** Has the same Graph Attention layers as our Teacher, *except* we rip out the temporal Conv1D + Bi-GRU so it only sees 1 frozen frame at a time ($T=1$). Result: 50%-occlusion error rises from $6.51\text{ mm} \to 8.41\text{ mm}$, and jitter rises from $7.81 \to 13.26\text{ mm/frame}^2$. This proves temporal memory is essential.

### 3. What Do All the Evaluation Metrics Actually Mean?
* **MPJPE (Mean Per-Joint Position Error, in `mm`):** Take all 21 joints of the hand. For each joint, measure the straight-line 3D distance (in millimeters) between our AI's predicted joint and the true ground-truth joint, then average across all 21 joints. *Smaller is better.* (For scale: a human pinky fingernail is ~10 mm wide; our Teacher achieves $6.51\text{ mm}$ and Student $7.04\text{ mm}$ even when half the hand is hidden!).
  * **Visible MPJPE:** Average error *only* on joints the camera could see ($m=1$).
  * **Occluded MPJPE:** Average error *only* on joints hidden behind the surgical tool ($m=0$).
* **PA-MPJPE (Procrustes-Aligned MPJPE, in `mm`):** Imagine our AI predicted the *exact right finger curl*, but the whole hand was shifted 2 mm to the left or tilted $2^\circ$. **Procrustes Alignment** rigidly slides/rotates/scales the predicted hand onto the ground-truth hand first, then measures the remaining error. It isolates **pure finger-shape accuracy** ($5.00\text{ mm}$ Teacher, $5.50\text{ mm}$ Student).
* **Bone-Length Error (in `mm`):** Measure the length of each of the 20 bones in the predicted hand ($\|\mathbf{P}_{\text{child}} - \mathbf{P}_{\text{parent}}\|_2$) and compare it to the true anatomical bone length $L_j$. Average the absolute difference. Standard AI models stretch/shrink bones by $1.96\text{–}6.20\text{ mm}$; ours is **$0.0000\text{ mm}$**.
* **Biomechanical Validity (%):** A predicted hand frame only passes this strict check (gets a "1") if **all 20 bones** have $\le 5\text{ mm}$ length drift **AND** **all joints** stay inside human bending limits ($\le 125^\circ$ flexion, $\le 15^\circ$ backward hyperextension). If even *one* finger stretches or bends backward, the whole frame fails ("0").
* **Temporal Jitter (in `mm/frame^2`):** Measures how violently the hand shakes from frame to frame. Take three consecutive frames $t-1, t, t+1$: compute $\mathbf{P}_{t+1} - 2\mathbf{P}_t + \mathbf{P}_{t-1}$. Notice that $(\mathbf{P}_{t+1} - \mathbf{P}_t) - (\mathbf{P}_t - \mathbf{P}_{t-1})$ is simply **change in velocity per frame** (discrete acceleration in millimeters per $\text{frame}^2$). If a hand moves at steady speed, jitter is $0$; if glare makes a fingertip jump back and forth, jitter spikes ($25.67\text{ mm/frame}^2$ raw $\to 7.81\text{ mm/frame}^2$ Teacher).

### 4. What is "Variance" and Why Does Stationary Variance Matter in Our Math?
* **What Variance Means:** Variance measures how spread out random numbers are around zero. Standard deviation $\sigma = \sqrt{\text{Variance}}$ tells you the typical size of the random wiggle (e.g., $\sigma = 4.5\text{ mm}$ means tracking dots typically wiggle by about $\pm 4.5\text{ mm}$).
* **Why Stationary Variance Matters on Slide 4 (AR(1) Noise):** Suppose you simulate camera shake over 100 frames by carrying over $65\%$ of the previous frame's error ($\rho = 0.65$) plus some fresh random noise $\mathbf{z}_t$:
  * If you added *full* fresh noise ($1.0 \cdot \mathbf{z}_t$) on top of $0.65 \cdot \boldsymbol{\epsilon}_{t-1}$, the total variance would grow bigger than $1$, so the simulated shake would blow up beyond $4.5\text{ mm}$!
  * If you added too little fresh noise, the shake would shrink toward $0$ over time!
  * By multiplying fresh noise by $\sqrt{1 - \rho^2} = \sqrt{1 - 0.65^2} \approx 0.76$, the variance at every single frame stays **pinned at exactly $1.0$** ($\rho^2 + (1-\rho^2) = 1$). That guarantees the camera noise stays steady and physically realistic at every frame—never exploding, never fading away.

### 5. What Do the Compute & Deployment Numbers Mean?
* **Parameters (`520,189` vs. `78,398`):** The number of learnable weights (knobs) inside the neural network. Fewer parameters = smaller memory footprint (`0.30 MB` FP32 / `0.15 MB` FP16) and faster loading.
* **FLOPs (`22.43 M` vs. `163,398`):** **Floating-Point Operations**—the exact count of additions and multiplications needed to process 1 frame. Cutting FLOPs by **$137.3\times$** means the Student does 137 times less arithmetic per frame!
* **ONNX (Opset 17, `433.21 KB`, `0.000231 mm` parity):** ONNX is a universal, portable model file format (like a PDF for neural networks) that runs on Windows, Mac, iPhone, and Jetson without needing PyTorch. "Parity error `0.000231 mm`" means the exported ONNX file produces the exact same 3D hand as PyTorch within 0.23 micrometers!
* **FPS, Latency (`ms`), and the 60 Hz Budget:** Video cameras often run at **60 frames per second (60 Hz)**, meaning each frame gets $\frac{1000\text{ ms}}{60} = 16.67\text{ ms}$ total to finish. Our Student finishes in **$1.242\text{ ms}$** on MacBook Air CoreML ($\frac{1000}{1.242} = 804.9\text{ FPS}$), using only $\frac{1.242}{16.67} = 7.5\%$ of the available time budget ($13.4\times$ headroom).

---

## Slide 1: Introduction — Motivation, Novelty & Team Roles

* **12-Year-Old Explanation:**  
  Optical hand trackers place 21 separate $(x,y,z)$ dots on a hand. In surgery, metal tools block half the fingers (**Occlusion**) and shiny latex gloves create bright reflections (**Specularity**) while stiffening the fingers (**12% stiffness**). Without visual clues, standard AI guesses wild positions for the 21 dots, stretching bones like rubber and bending knuckles backward. We replace loose dots with a **rigid mechanical hand puppet** controlled by joint rotations.
* **What the Numbers Mean:**
  * `21` joints (`0` wrist + `4` joints per finger $\times 5$ fingers = `21` joints, `20` bones).
  * `30%–70%` occlusion: hiding 6, 10, or 15 joints out of 21.
  * `32,560` poses: total real 3D hand captures from `FreiHAND_pub_v2`.
  * `804.9 FPS (1.242 ms)` on Apple Silicon MacBook Air; `557.9 FPS (1.792 ms)` on x86_64 PC.

---

## Slide 2: Project Objectives & Literature Gap

* **Five Objectives (In Plain English):**
  1. Build a clean, zero-leakage hybrid dataset of real hands + surgical gestures with simulated gloves and tool occlusion.
  2. Build a Forward Kinematics layer that makes $0.0000\text{ mm}$ bone error and $100\%$ anatomical validity mathematically guaranteed.
  3. Build a smart Teacher (`SpatioTemporalGAE`) using graphs + time memory, and prove time memory helps via an ablation (`SpatialOnlyGAE`).
  4. Compress that Teacher into a tiny Student (`StudentKinematicMLP`) and prove it beats an identical-size `UnconstrainedStudentMLP`.
  5. Export to ONNX and run $>500\text{ FPS}$ on real hardware + live demo.
* **Why Competitors Fall Short (Table):**
  * **HandGCN / HOPE-Net:** Use graphs inside, *still* output raw $(x,y,z)$ at the end $\implies 3.4\text{–}4.8\text{ mm}$ bone distortion.
  * **Spurr et al.:** Uses *soft* hinge penalties during training $\implies$ at test time under occlusion, bones still break ($1.90\text{ mm}$ error, $91.4\%$ valid).
  * **HandOccNet:** Heavy dual Transformer ($37.2\text{M}$ params, $142\text{ MB}$, $30\text{ FPS}$)—way too big for lightweight edge deployment.
  * **MediaPipe Hands:** Fast ($220\text{ FPS}$), but unconstrained ($6.20\text{ mm}$ bone error, $64.8\%$ valid under occlusion).

---

## Slide 3: System Block Diagram & Core Algorithms (Figure 0)

* **Walking Through the 4 Stages:**
  1. **Stage 1 (Data + Normalization):** Subtract the wrist position so wrist = $(0,0,0)$, and divide all coordinates by $S_{\text{palm}} = 100\text{ mm}$ (roughly the width/length of a palm). Why? Neural networks train stably when inputs are small numbers around $[-1, +1]$ rather than $[-150, +150]$. Add a 4th channel $m \in \{0, 1\}$ (`1` = visible, `0` = occluded). Tensor shape: $(B, T, 21, 4)$.
  2. **Stage 2 (Teacher `SpatioTemporalGAE`):** Replaces zeros at occluded nodes with a neutral rest hand, runs 3 layers of 4-head Mask-Aware Graph Attention, then runs Depthwise Conv1D + Bidirectional GRU over $T=8$ frames.
  3. **Stage 3 (Student `StudentKinematicMLP`):** Flattens $(21 \times 4 = 84)$ inputs per frame, mixes neighboring frames with one Conv1D (`k=3`), runs a tiny MLP, and learns from the Teacher via `DistillationLoss`.
  4. **Stage 4 (Shared Hard Kinematic Manifold):** Gram-Schmidt $\mathbb{R}^6 \to \text{SO}(3)$ $\to$ $\tanh$ RoM Clamper $\to$ Recursive FK (`DifferentiableFK` in PyTorch / `EdgeFK` in NumPy).

---

## Slide 4: Hybrid FreiHAND + MANO Data Pipeline & 3D Domain Adaptation

* **Why 80/20 Split (`960` Train / `240` Test) Matters:**
  If you accidentally test a model on sequences it already memorized during training (**data leakage**), your score looks artificially amazing but fails in the real world. `split_indices.json` locks 240 completely unseen test sequences away from training. (Multi-rate windowing creates $960 \times 3 = 2,880$ train windows and $240 \times 3 = 720$ test windows across speed strides).
* **Every Symbol in `SyntheticGloveTransform`:**
  $$\tilde{\mathbf{d}}_{t,j} = \frac{(1 - \alpha_{\text{stiff}})\hat{\mathbf{d}}_{t,j} + \alpha_{\text{stiff}}\hat{\mathbf{u}}_j^{\text{rest}}}{\|(1 - \alpha_{\text{stiff}})\hat{\mathbf{d}}_{t,j} + \alpha_{\text{stiff}}\hat{\mathbf{u}}_j^{\text{rest}}\|_2}$$
  * $\hat{\mathbf{d}}_{t,j}$: unit direction arrow (length 1) of bone $j$ at frame $t$.
  * $\hat{\mathbf{u}}_j^{\text{rest}}$: unit direction arrow of bone $j$ in a relaxed neutral hand.
  * $\alpha_{\text{stiff}} = 0.12$: latex glove pulls every bone $12\%$ toward rest; dividing by $\|\cdot\|_2$ re-normalizes length to $1$ so bone lengths never change.
  $$\mathbf{P}_{t,j}^{\text{gloved}} = \tilde{\mathbf{P}}_{t,j} + \sigma_j \boldsymbol{\epsilon}_{t,j} + \mathbb{I}(j \in \mathcal{V}_{\text{tip}})\,b_{t,j}\,\boldsymbol{\xi}_{t,j}$$
  * $\boldsymbol{\epsilon}_{t,j} = 0.65\boldsymbol{\epsilon}_{t-1,j} + \sqrt{1-0.65^2}\mathbf{z}_{t,j}$: stationary AR(1) temporal drift ($\text{Var} = 1$).
  * $\sigma_j$: $4.5\text{ mm}$ baseline jitter ($0\text{ mm}$ at wrist, $5.625\text{ mm}$ on DIP).
  * $\mathbb{I}(j \in \mathcal{V}_{\text{tip}})$: only turns on at the 5 fingertips. $b_{t,j} \sim \text{Bernoulli}(0.20)$ triggers a $12.0\text{ mm}$ glare spike on $20\%$ of frames.
* **`SyntheticOcclusionTransform`:**
  Spawns a $28\text{ mm}$-radius cylinder across the hand. Enforces **distal-to-proximal priority** ($\text{TIP} \to \text{DIP} \to \text{PIP} \to \text{MCP}$): if a knuckle is hidden, everything past it is hidden too. `round(0.30 * 21) = 6`, `round(0.50 * 21) = 10`, `round(0.70 * 21) = 15` occluded joints, while wrist `0` stays visible.

---

## Slide 5: 6D Continuous Rotations & Hard Edge Forward Kinematics

### 1. What is $\text{SO}(3)$ and Why 6D Rotations?
* **$\text{SO}(3)$ ("Special Orthogonal Group in 3D"):** Just a formal name for the set of all valid 3D rotation matrices $\mathbf{R}$ ($3\times 3$) that:
  1. Keep lengths and angles rigid ($\mathbf{R}^\top \mathbf{R} = \mathbf{I}_3$, meaning columns are perpendicular unit vectors), and
  2. Never mirror-flip left into right ($\det(\mathbf{R}) = +1$).
* **Why 6D instead of 3 Euler angles or 4 Quaternions?**
  * *Euler angles* hit Gimbal lock (two axes collapse at $90^\circ$) and jump at $\pm 180^\circ$.
  * *Quaternions* have a double-cover ($q$ and $-q$ are the same rotation), causing discontinuities when a neural network interpolates across the sphere.
  * *6D vectors* $[\mathbf{a}_1, \mathbf{a}_2] \in \mathbb{R}^6$ (Zhou et al., CVPR 2019) have no wrap-around jumps anywhere!
* **Gram-Schmidt Equations Decoded:**
  $$\mathbf{b}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|_2}, \quad \mathbf{b}_2 = \frac{\mathbf{a}_2 - (\mathbf{b}_1 \cdot \mathbf{a}_2)\mathbf{b}_1}{\|\mathbf{a}_2 - (\mathbf{b}_1 \cdot \mathbf{a}_2)\mathbf{b}_1\|_2}, \quad \mathbf{b}_3 = \mathbf{b}_1 \times \mathbf{b}_2$$
  1. Divide $\mathbf{a}_1$ by its length $\to$ unit vector $\mathbf{b}_1$.
  2. Subtract the shadow of $\mathbf{a}_2$ along $\mathbf{b}_1$ (`(b1 · a2) * b1`) so the remainder is strictly $90^\circ$ perpendicular to $\mathbf{b}_1$, then normalize $\to \mathbf{b}_2$.
  3. Cross product $\mathbf{b}_1 \times \mathbf{b}_2 \to \mathbf{b}_3$, guaranteed perpendicular to both and right-handed ($\det = +1$).

### 2. Smooth $\tanh$ Range-of-Motion Clamping (`BiomechanicalRoMClamper`)
$$\boldsymbol{\theta}_j^{\text{clamp}} = \boldsymbol{\mu}_j + \boldsymbol{\Delta}_j \odot \tanh\!\left(\frac{\boldsymbol{\theta}_j - \boldsymbol{\mu}_j}{\boldsymbol{\Delta}_j \cdot \tau + \epsilon}\right)$$
* $\boldsymbol{\mu}_j = \frac{\boldsymbol{\theta}_j^{\min} + \boldsymbol{\theta}_j^{\max}}{2}$ (midpoint) and $\boldsymbol{\Delta}_j = \frac{\boldsymbol{\theta}_j^{\max} - \boldsymbol{\theta}_j^{\min}}{2}$ (half-width).
* Because $-1 < \tanh(x) < +1$ for *any* real number $x$, $\boldsymbol{\theta}_j^{\text{clamp}}$ is strictly trapped inside $(\boldsymbol{\theta}_j^{\min}, \boldsymbol{\theta}_j^{\max})$!
* Why $\tanh$ instead of hard `min/max`? Hard `clamp` has zero derivative outside the bounds, killing backpropagation gradients; $\tanh$ is smooth ($C^\infty$) everywhere. And `_safe_angle_from_sin_cos` avoids the infinite slope of $\arccos(u)$ at $u=1$ ($\theta=0$).

### 3. Theorem 1 Proof (Exact Bone-Length Invariance)
$$\mathbf{P}_j = \mathbf{P}_{p(j)} + \mathbf{R}_j^{\text{global}} \mathbf{v}_j^{\text{rest}}$$
* Subtract $\mathbf{P}_{p(j)}$ from both sides and take the Euclidean length $\|\cdot\|_2$:
  $$\|\mathbf{P}_j - \mathbf{P}_{p(j)}\|_2 = \|\mathbf{R}_j^{\text{global}} \mathbf{v}_j^{\text{rest}}\|_2 = \sqrt{(\mathbf{v}_j^{\text{rest}})^\top \underbrace{(\mathbf{R}_j^{\text{global}})^\top \mathbf{R}_j^{\text{global}}}_{\mathbf{I}_3} \mathbf{v}_j^{\text{rest}}} = \|\mathbf{v}_j^{\text{rest}}\|_2 = L_j$$
* Because $\mathbf{R}_j^{\text{global}}$ is orthogonal ($\mathbf{R}^\top\mathbf{R} = \mathbf{I}_3$), it drops right out of the dot product! Bone length error is **identically $0.0000\text{ mm}$**.

---

## Slide 6: Spatio-Temporal Graph Autoencoder Teacher

* **7D Imputed Input Vector:**
  $$\mathbf{x}_{t,j}^{\text{imp}} = m_{t,j}\tilde{\mathbf{x}}_{t,j} + (1 - m_{t,j})\bar{\mathbf{x}}_j^{\text{rest}}, \quad \mathbf{f}_{t,j}^{\text{in}} = \left[\mathbf{x}_{t,j}^{\text{imp}} \;\|\; m_{t,j} \;\|\; m_{t,j}(\tilde{\mathbf{x}}_{t,j} - \bar{\mathbf{x}}_j^{\text{rest}})\right] \in \mathbb{R}^7$$
  * Prevents occluded joints ($m=0$, coords $= 0$) from looking like they collapsed onto the wrist; instead places them at neutral rest $\bar{\mathbf{x}}_j^{\text{rest}}$ with $m=0$ flag and $0$ displacement.
* **69 Directed Edges:**
  21 self-loops + 20 bidirectional bone edges ($40$ directed) + 4 bidirectional palmar arch edges connecting neighboring knuckles ($8$ directed) $= 21 + 40 + 8 = 69$.
* **Mask-Aware Attention Equation Decoded:**
  $$e_{t,i,j}^{(h)} = \frac{\mathbf{q}_{t,i}^{(h)} \cdot \mathbf{k}_{t,j}^{(h)}}{\sqrt{d_h}} + 0.25\,\text{LeakyReLU}\!\left(\mathbf{a}_{\text{src}}^{(h)}\cdot\mathbf{q}_{t,i}^{(h)} + \mathbf{a}_{\text{dst}}^{(h)}\cdot\mathbf{k}_{t,j}^{(h)}\right) + B_{h,i,j} - \gamma_h(1 - m_{t,j})$$
  * **Query ($i$):** the joint *asking* neighbors for help.
  * **Key ($j$):** the neighbor *offering* information.
  * **$-\gamma_h(1 - m_{t,j})$:** if neighbor $j$ is hidden ($m_{t,j}=0$), subtract $\gamma_h \approx 1.5$ before `softmax`, shrinking its outgoing weight by $e^{-1.5} \approx 0.22\times$. Hidden joints can still *listen* (as Queries) to visible knuckles without *polluting* them (as Keys)!
* **Bi-GRU ($T=8$):** Reads forward and backward across the 8-frame window, cutting 50%-occlusion MPJPE from $8.41\text{ mm}$ (`SpatialOnlyGAE`) down to $6.51\text{ mm}$ ($-22.6\%$) and jitter from $13.26 \to 7.81\text{ mm/frame}^2$ ($-41.1\%$).

---

## Slide 7: Physics-Informed Knowledge Distillation & Edge Student Compression

* **How the Student Achieves $137.3\times$ Fewer FLOPs:**
  Replaces $O(N^2)$ 4-head Graph Attention and sequential Bi-GRU loops with a flat linear layer ($84 \to 128$), one depthwise 1D convolution (`TemporalMixingBlock1D`, $k=3$), and one residual MLP block before the exact same `BiomechanicalRoMClamper` + `DifferentiableFK`.
* **Why `DistillationLoss` Has 5 Terms:**
  * $\alpha_{\text{kd}}^{\text{rot}} = 35.0$ & $\alpha_{\text{kd}}^{\text{pos}} = 0.45$: Match the frozen Teacher's 6D rotations and 3D FK joint positions.
  * $\alpha_{\text{gt}}^{\text{pos}} = 0.55$ & $\alpha_{\text{gt}}^{\text{rot}} = 25.0$: Stay anchored to true Ground Truth, weighting occluded joints by $w_{\text{occ}} = 1.5\times$.
  * $\alpha_{\text{temp}} = 0.35$: Penalize velocity and acceleration jumps so the Student moves smoothly.
* **Rotation Loss $\mathcal{L}_{6\text{D}}$ Decoded:**
  $$\mathcal{L}_{6\text{D}}(\hat{\mathbf{R}}, \mathbf{R}) = \|\hat{\mathbf{d}}_{6\text{D}} - \mathbf{d}_{6\text{D}}\|_2^2 + 0.25\,\arccos\!\left(\frac{\text{Tr}(\hat{\mathbf{R}}^\top\mathbf{R})-1}{2}\right) + 0.25\!\left(1 - \frac{\text{Tr}(\hat{\mathbf{R}}^\top\mathbf{R})-1}{2}\right)$$
  * $\text{Tr}(\hat{\mathbf{R}}^\top\mathbf{R}) = 1 + 2\cos\theta$, where $\theta$ is the true 3D rotation angle error between predicted $\hat{\mathbf{R}}$ and target $\mathbf{R}$.
  * $\arccos(\dots)$ = **Geodesic distance** (exact angle $\theta$ in radians on $\text{SO}(3)$).
  * $1 - \cos\theta$ = **Chordal distance** (smooth gradient when $\theta \approx 0$ where $\arccos$ gets steep).

---

## Slides 8 & 9: Experimental Results & Granular FDS Analysis

* **Key Pattern in Slide 8 Table (50% Occlusion):**
  * Raw Corrupted Input: $52.08\text{ mm}$ MPJPE, $19.96\text{ mm}$ bone error, $0.00\%$ valid.
  * `UnconstrainedStudentMLP` (no FK): $9.95\text{ mm}$ MPJPE, **$2.3052\text{ mm}$ bone error**, **only $29.95\%$ valid** (70% broken!).
  * `SpatialOnlyGAE` (no Bi-GRU): $8.41\text{ mm}$ MPJPE, $0.0000\text{ mm}$ bone error, $100.00\%$ valid.
  * `Teacher ST-GAE`: **$6.51\text{ mm}$ MPJPE** ($9.09\text{ mm}$ on occluded joints), $0.0000\text{ mm}$ bone error, $100.00\%$ valid, $7.81\text{ mm/frame}^2$ jitter.
  * `StudentKinematicMLP`: **$7.04\text{ mm}$ MPJPE** ($10.02\text{ mm}$ occluded), $0.0000\text{ mm}$ bone error, $100.00\%$ valid, $8.27\text{ mm/frame}^2$ jitter—beating the $5.89\times$ larger `SpatialOnlyGAE` because the Student distilled temporal motion priors!
* **Slide 9 Insights:**
  * **Jitter Suppression:** Cuts raw $25.67\text{ mm/frame}^2$ glare jitter by **$69.6\%$ (Teacher)** and **$67.8\%$ (Student)**.
  * **Kinematic Lever Effect (Figure 7a):** Just like a fishing rod amplifies a small handle tilt at the tip, joint error scales from Palm ($2.61\text{ mm}$) $\to$ MCP ($5.38\text{ mm}$) $\to$ PIP ($8.35\text{ mm}$) $\to$ Distal Tip ($10.98\text{ mm}$ Teacher / $12.02\text{ mm}$ Student, $-28.9\%$ better than unconstrained $16.90\text{ mm}$).
  * **Zero-Shot Custom Gloved (`120` seqs):** $6.37\text{ mm}$ Teacher / $7.01\text{ mm}$ Student at 50% occlusion without retraining.
  * **Downstream Gesture Utility:** Boosts 70%-occlusion trajectory retrieval from $26.7\%$ (unconstrained) to $39.2\%\text{–}46.7\%$.

---

## Slides 10 & 11: Edge Profiling & Live Telemetry Demo

* **Physical Benchmarks Decoded:**
  * **Apple MacBook Air (Bhavya, CoreML):** $1.242\text{ ms}$ ($804.9\text{ FPS}$) single-frame; $1.935\text{ ms}$ for $T=8$ ($0.242\text{ ms/frame} = 4,134\text{ FPS}$). Uses $7.5\%$ of a 60 FPS frame budget. Bare-metal NumPy `EdgeFK` runs in $0.0725\text{ ms}$ ($13.79\text{ kHz}$).
  * **Windows PC (Abhishekh, x86_64 CPU):** $1.792\text{ ms}$ ($557.9\text{ FPS}$) vs. Teacher's $29.34\text{ ms}$ ($34.1\text{ FPS}$)—$16.4\times$ speedup.
  * **iPhone ARKit UDP:** $0.08\text{ ms}$ UDP parse + $1.242\text{ ms}$ inference $= 1.322\text{ ms}$ ($1000/1.322 = 756.4\text{ Hz}$).
  * **Jetson Nano / Raspberry Pi:** Marked *Pending physical run*; $163.4\text{ KFLOPs}$ is $<0.1\%$ of Jetson Nano's $472\text{ GFLOPs}$ Maxwell GPU ($\sim 2,000\times$ lighter than MobileNetV2).
* **Live Demo Commands:**
  ```powershell
  # Primary offline replay (zero Wi-Fi/lighting failure risk):
  .\.venv\Scripts\python.exe run_presentation_demo.py --mode replay --occlusion 0.50

  # Live webcam mode ([T] = toggle surgical tool box, [O] = cycle 0%/30%/50%/70%, [Q] = quit):
  .\.venv\Scripts\python.exe run_presentation_demo.py --mode webcam --camera 0 --occlusion 0.50
  ```
