# Slide-by-Slide Presentation & Defense Master Guide (16 Slides)
**Project:** Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry  
**Presentation Date:** November 17, 2026  
**Course:** Foundations of Data Science (FDS) — Final Evaluation (15 Marks Presentation + 5 Marks Report)  
**Total Presentation Time:** 16–18 Minutes (~4 Minutes per Speaker, 16 Slides Total)  
**Presenters & Team Ownership:**
* **Abhishekh Verma:** Slides 1, 6, 7, 16 (~4:50 min) — Introduction, 6D Rotations & Forward Kinematics, Live Demo & Conclusion
* **Bhavya:** Slides 2, 3, 4, 5 (~4:15 min) — Objectives, Literature Gap, System Pipeline Block Diagram, Hybrid Data Engineering
* **Vrushank:** Slides 8, 9, 10, 11 (~4:00 min) — Teacher ST-GAE, Mask-Aware GAT, Distillation Loss, Student Compression
* **Kapish:** Slides 12, 13, 14, 15 (~4:00 min) — Multi-Rate Occlusion Benchmarks, Bone Invariance, Jitter/Hand Repair, Multi-Hardware Edge

---

## Master Presentation Timing & Handover Schedule (17:05 Total)

| Slide # | Slide Title | Speaker | Allocated Time | Cumulative Time | Handover Cue / Transition |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **S1** | Intro: Problem Context, Motivation & Team Roles | **Abhishekh** | 1:05 | 0:00 – 1:05 | *"I will now hand over to Bhavya to walk us through our core research objectives and literature gap."* |
| **S2** | Five Core Research Objectives (Rubric 1M) | **Bhavya** | 1:00 | 1:05 – 2:05 | Continues directly to Slide 3. |
| **S3** | SOTA Literature Gap & Comparative Matrix | **Bhavya** | 1:00 | 2:05 – 3:05 | Continues directly to Slide 4. |
| **S4** | System Architecture Block Diagram (Rubric 3M) | **Bhavya** | 1:00 | 3:05 – 4:05 | Continues directly to Slide 5. |
| **S5** | Data Engineering: Hybrid Pipeline & Transforms | **Bhavya** | 1:15 | 4:05 – 5:20 | *"I will now hand back to Abhishekh to explain the mathematical foundation of our 6D kinematic manifold."* |
| **S6** | 6D Continuous Rotations & Anatomical RoM Clamping | **Abhishekh** | 1:05 | 5:20 – 6:25 | Continues directly to Slide 7. |
| **S7** | Differentiable Forward Kinematics & Theorem 1 Proof | **Abhishekh** | 1:10 | 6:25 – 7:35 | *"I will now hand over to Vrushank to explain our Spatio-Temporal Graph Autoencoder Teacher."* |
| **S8** | Teacher Architecture: Spatio-Temporal GAE | **Vrushank** | 1:00 | 7:35 – 8:35 | Continues directly to Slide 9. |
| **S9** | Spatial Graph Attention & Temporal Dilated Convolutions | **Vrushank** | 1:00 | 8:35 – 9:35 | Continues directly to Slide 10. |
| **S10** | Physics-Informed Knowledge Distillation Strategy | **Vrushank** | 1:00 | 9:35 – 10:35 | Continues directly to Slide 11. |
| **S11** | Edge Student Architecture & Compression Analysis | **Vrushank** | 1:00 | 10:35 – 11:35 | *"I will now hand over to Kapish to walk through our experimental results and multi-hardware benchmarks."* |
| **S12** | Quantitative Occlusion Robustness (30%, 50%, 70%) | **Kapish** | 1:00 | 11:35 – 12:35 | Continues directly to Slide 13. |
| **S13** | Anatomical Invariance & Zero Bone Length Violation | **Kapish** | 1:00 | 12:35 – 13:35 | Continues directly to Slide 14. |
| **S14** | Temporal Jitter Suppression & Qualitative 3D Hand Repair | **Kapish** | 1:00 | 13:35 – 14:35 | Continues directly to Slide 15. |
| **S15** | Downstream Gesture Utility & Multi-Hardware Edge Telemetry | **Kapish** | 1:00 | 14:35 – 15:35 | *"I will now hand back to Abhishekh for our interactive live demonstration and conclusion."* |
| **S16** | Interactive Live Telemetry Demo HUD & Conclusion (Rubric 5M) | **Abhishekh** | 1:30 | 15:35 – 17:05 | Concludes presentation and opens floor to Professor Q&A while demo runs live. |

---

## Detailed Slide-by-Slide Script, Layman Intuition & Committee Defense

---

### Slide 1: Introduction — Problem Context, Motivation & Team Ownership
* **Speaker:** Abhishekh Verma
* **Target Time:** 1:05 (0:00 – 1:05)
* **Rubric Weight:** Introduction (1 Mark)

#### 1. Plain English Layman Analogy
Imagine a surgeon performing laparoscopic surgery with robotic arms, or a worker controlling industrial machinery via VR glasses. A vision camera tracks 21 points on their hand. Two severe problems destroy conventional AI tracking:
1. **Tool Occlusion:** A scalpel, drill, or forceps blocks 30% to 70% of the fingers from the camera's sight.
2. **Medical Latex Gloves:** Smooth rubber gloves erase skin texture and knuckle wrinkles, reflect harsh surgical lights, and make fingers slightly stiffer.
When fingers vanish behind tools, standard neural networks (like Google MediaPipe) guess floating 3D points in empty space. The result? Virtual fingers **stretch like elastic rubber bands** by 2 to 3 centimeters, or bend backward like broken bones. Our system replaces floating dots with an **anatomical digital puppet**: bones are physically rigid metal rods, and knuckles have mechanical stoppers that stop fingers from ever bending backward.

#### 2. Authoritative Spoken Script
> "Good morning professors and colleagues. Today our team—Bhavya, Vrushank, Kapish, and I—present *Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry*.
> 
> In safety-critical teleoperation and surgical robotics, camera-based hand trackers fail whenever medical instruments block line-of-sight to the fingers or when protective latex gloves introduce specular reflection. Traditional deep networks formulate hand pose estimation as unconstrained Cartesian regression, predicting 63 independent coordinates in $\mathbb{R}^{21 \times 3}$. Under severe occlusion, lacking visual gradients, these networks hallucinate points in free space—stretching bones by up to 27 millimeters and producing anatomically impossible hyperextensions.
> 
> To resolve this, we present a physics-informed graph kinematic framework that guarantees exact human bone lengths and valid biological joint limits by construction. Our pipeline runs at over 800 frames per second on mobile edge hardware. As shown in our team distribution table, each of us spearheaded a dedicated module of this pipeline. I will now hand over to Bhavya to walk us through our core research objectives and literature gap."

#### 3. Expected Professor Q&A on Slide 1
* **Professor:** *"Why can't you just smooth the MediaPipe coordinates with a low-pass Kalman filter?"*
* **Abhishekh:** *"A temporal Kalman filter only smooths motion over time; it cannot restore missing spatial anatomy. When a scalpel occludes a finger for 2 seconds, the Kalman filter smoothly preserves the wrong, disarticulated pose. Our model leverages spatial graph attention across visible knuckles to actively infer the missing finger's anatomical joint angles."*

---

### Slide 2: Five Core Research Objectives
* **Speaker:** Bhavya
* **Target Time:** 1:00 (1:05 – 2:05)
* **Rubric Weight:** Objectives (1 Mark)

#### 1. Plain English Layman Analogy
When building a safety-critical system, you need a strict flight checklist. We set five non-negotiable milestones:
1. Test on real human hands, not just clean 3D video game models.
2. Guarantee mathematically that bones cannot stretch by even a fraction of a millimeter.
3. Build a smart "Teacher" brain that can look at visible knuckles and deduce what hidden fingers are doing.
4. Compress that heavy Teacher into a tiny "Student" chip that fits inside a smartwatch or phone.
5. Prove it works live on real laptops and phones at over 60 frames per second.

#### 2. Authoritative Spoken Script
> "Thank you Abhishekh. In Slide 2, we formalize the five core research objectives of our project, directly aligned with our course grading rubric.
> 
> First, in Objective 1, we establish a zero-leakage benchmark combining 32,560 real markerless 3D poses from FreiHAND with clinical MANO trajectories, evaluated under realistic 30%, 50%, and 70% surgical tool occlusion and latex glove physics.
> 
> Second, in Objective 2, we formulate a differentiable forward kinematics layer that mathematically guarantees exactly 0.0000 mm bone-length error and 100% biomechanical validity.
> 
> Third, in Objective 3, we build a mask-aware Spatio-Temporal Graph Autoencoder Teacher that combines visibility-gated graph attention with bidirectional temporal recurrence.
> 
> Fourth, in Objective 4, we distill this teacher into a lightweight student model with fewer than 80,000 parameters—reducing compute by 137 times.
> 
> And fifth, in Objective 5, we deploy this architecture across Apple Silicon, PC, and iPhone hardware, sustaining sub-2 millisecond latency for live interactive telemetry."

#### 3. Expected Professor Q&A on Slide 2
* **Professor:** *"How do these objectives align with Foundations of Data Science (FDS)?"*
* **Bhavya:** *"Our project addresses the core data science lifecycle: data engineering with real-world distribution shifts, non-Euclidean graph modeling, knowledge distillation, hypothesis-driven ablation testing on held-out test splits, and hardware edge deployment."*

---

### Slide 3: State-of-the-Art Literature Gap & Comparative Matrix
* **Speaker:** Bhavya
* **Target Time:** 1:00 (2:05 – 3:05)

#### 1. Plain English Layman Analogy
Why haven't existing computer vision papers solved this?
* **Graph Models (HandGCN, HOPE-Net):** They connect joints in the middle, but their final output guesses free-floating $X, Y, Z$ numbers. When fingers are hidden, bones stretch by 3 to 5 millimeters.
* **"Soft Loss Penalties" (Spurr et al., ECCV 2020):** During training, they penalize the AI if it stretches a bone. But at test time, soft penalties are like speed limit signs—when heavy occlusion hits, the model speeds right through the sign, resulting in 8.6% broken poses.
* **Massive Transformers (HandOccNet, CVPR 2022):** They predict realistic hands, but the model has 37 million parameters and takes 33 milliseconds per frame on an expensive GPU. You cannot put that in an edge telemetry box.
* **Our Edge Model:** We embed hard mechanical stops inside the network itself: 0.0000 mm bone error, 100% validity, and runs in 1.24 milliseconds (805 FPS)—over 470 times smaller than HandOccNet!

#### 2. Authoritative Spoken Script
> "Slide 3 compares our approach against state-of-the-art vision literature. Prior works fall into two broad failure modes.
> 
> The first failure mode is unconstrained coordinate regression. While spectral graph networks like HandGCN and HOPE-Net use graph convolutions in hidden layers, their final layer regresses unconstrained Cartesian coordinates. Under occlusion, their bone-length error degrades to between 3.4 and 4.8 millimeters.
> 
> The second failure mode is reliance on soft loss penalties. Spurr et al. at ECCV 2020 penalize bone length violations during training. However, because soft penalties do not constrain the forward graph at inference time, out-of-distribution occlusion causes 8.6% of predicted poses to be anatomically invalid. In our own `UnconstrainedStudent` ablation at 50% occlusion, biomechanical validity drops to just 29.95%.
> 
> Meanwhile, transformer mesh regressors like HandOccNet achieve high accuracy but require 37 million parameters and 33 milliseconds per frame. Our distilled Student achieves 7.04 mm error while guaranteeing 0.0000 mm bone error and 100% biological validity, executing in just 1.24 milliseconds."

#### 3. Expected Professor Q&A on Slide 3
* **Professor:** *"Why is bone length error so critical? Isn't MPJPE enough?"*
* **Bhavya:** *"MPJPE measures average Euclidean distance across 21 joints, which can conceal catastrophic local disarticulations. If an index fingertip stretches by 25 millimeters while the remaining 20 joints are accurate, average MPJPE might appear acceptable, but robotic teleoperation would drive a surgical scalpel directly through patient tissue. Bone invariance ensures the physical integrity of the tool-hand interface."*

---

### Slide 4: System Architecture & End-to-End Block Diagram
* **Speaker:** Bhavya
* **Target Time:** 1:00 (3:05 – 4:05)
* **Rubric Weight:** Block Diagram (3 Marks)

#### 1. Plain English Layman Analogy
Look at **Figure 0** as a 4-station assembly line:
1. **Station 1 (Top Left — The Reality Simulator):** Takes real hand trajectories, normalizes hand size to 100 mm, adds latex glove stiffness, and drops a 3D cylindrical surgical tool that hides up to 70% of joints with a binary visibility flag ($m=1$ visible, $m=0$ hidden).
2. **Station 2 (Bottom Left — The Teacher):** Replaces missing joints with a relaxed hand pose, connects knuckles via a 21-joint graph, uses Graph Attention to let visible knuckles talk to hidden knuckles, and uses a Bidirectional GRU across 8 frames to smooth motion.
3. **Station 3 (Bottom Center — The Student):** Replaces heavy graph operations with a fast 1D depthwise convolution to run in real time.
4. **Station 4 (Right — The Kinematic Puppet Engine):** Turns raw outputs into Gram-Schmidt 6D rotations, clamps angles with $\tanh$, and walks out the bones using Forward Kinematics.

#### 2. Authoritative Spoken Script
> "Slide 4 presents our end-to-end system architecture block diagram, corresponding to Section 3 of our report and Algorithms 1 and 2.
> 
> In Stage 1 on the top left, the pipeline takes streaming hand poses, normalizes them relative to the wrist with a canonical palm scale of 100 mm, and applies our glove and occlusion transforms. This produces an input tensor $\mathbf{X}$ containing 3D coordinates and a binary visibility mask $m_{t,j}$.
> 
> In Stage 2 on the bottom left, our 520K-parameter Teacher imputes occluded nodes with canonical rest priors, runs 3 layers of 4-head Graph Attention with a visibility gating penalty, and aggregates temporal context across 8 frames via a Bidirectional GRU.
> 
> In Stage 3 on the bottom center, we distill the Teacher's knowledge into our 78.4K-parameter Student MLP using a 5-term loss supervising both joint angles and 3D positions.
> 
> In Stage 4 on the right, both models feed into our shared kinematic manifold: Gram-Schmidt orthogonalization onto $\text{SO}(3)$, hyperbolic-tangent range-of-motion clamping, and forward kinematics, compiled into an ultra-compact ONNX graph and a bare-metal NumPy engine."

#### 3. Expected Professor Q&A on Slide 4
* **Professor:** *"Where does the 100 mm palm scale come from, and does it distort real hand measurements?"*
* **Bhavya:** *"100 mm is an internal normalization factor used inside the neural network so that coordinate gradients remain numerically stable around unit variance. After Forward Kinematics computes the unit-scale pose, outputs are multiplied back into physical millimeters matching canonical anthropometric bone lengths."*

---

### Slide 5: Data Engineering: Hybrid FreiHAND+MANO Pipeline & Transforms
* **Speaker:** Bhavya
* **Target Time:** 1:15 (4:05 – 5:20)

#### 1. Plain English Layman Analogy
To train an AI model for surgery, you cannot just use clean synthetic hands from a computer game. We built a hybrid dataset:
* We ingested **32,560 real human hand poses** from the FreiHAND dataset captured with real cameras, and mixed them 50/50 with clinical MANO gestures like holding a scalpel, needle driver, or surgical drill.
* We created two domain adaptation engines:
  * **Glove Physics:** Rubber gloves pull fingers 12% toward a relaxed pose and reflect bright lights, creating 12 mm tracking spikes at the fingertips.
  * **Surgical Tool Occlusion:** A virtual 3D cylinder sweeps across the hand, hiding fingertips first, then middle joints, then knuckles—exactly how a real tool obscures fingers in an operating room.

#### 2. Authoritative Spoken Script
> "Slide 5 covers our data engineering and domain adaptation pipeline in `data/dataset_ingestion.py`.
> 
> To ensure our model generalizes beyond idealized synthetic animations, we ingested all 32,560 real 3D hand poses from the University of Freiburg FreiHAND dataset. We extracted local 6D rotations via Rodrigues inverse kinematics, harmonized them onto canonical bone lengths, and paired pose neighbors into smooth trajectories across 8 frames. We combined these 50/50 with clinical and industrial MANO gestures—including scalpel pinch, needle driver, and drill grasps—split strictly 80/20 with zero sequence leakage.
> 
> To simulate surgical environments, I engineered two physics-grounded transforms. First, `SyntheticGloveTransform` models latex elasticity by pulling bone unit vectors 12% toward the relaxed rest state while maintaining exact bone lengths $L_j$. It injects autoregressive AR(1) sensor noise with correlation $\rho = 0.65$ and 12-millimeter Bernoulli specular glare spikes at the fingertips.
> 
> Second, `SyntheticOcclusionTransform` projects a 3D cylinder of radius 28 millimeters across the phalanges, enforcing contiguous distal-to-proximal subtree priority so fingertips disappear before knuckles, mirroring real surgical instruments across 30%, 50%, and 70% occlusion. I will now hand back to Abhishekh to explain the mathematical foundation of our kinematic manifold."

#### 3. Expected Professor Q&A on Slide 5
* **Professor:** *"Why use contiguous subtree priority instead of randomly masking joints uniformly at random?"*
* **Bhavya:** *"In the physical world, an instrument does not occlude random disconnected joints—a scalpel shaft physically covers an entire finger segment from tip to knuckle. Uniform random masking would make the reconstruction problem artificially easy because an occluded middle joint would always have visible neighbors on both sides. Contiguous subtree masking correctly models complete distal blindness."*

---

### Slide 6: 6D Continuous Rotations & Anatomical RoM Clamping
* **Speaker:** Abhishekh Verma
* **Target Time:** 1:05 (5:20 – 6:25)

#### 1. Plain English Layman Analogy
How do you teach an AI to rotate a 3D joint?
* **Euler angles (Pitch, Yaw, Roll):** When two rotation axes line up, you get "Gimbal Lock"—the math breaks down and angles suddenly jump by 180 degrees.
* **Quaternions (4 numbers):** A rotation of $+q$ and $-q$ look identical in physical space, but to a neural network, they are opposite poles of a 4D sphere. The AI gets confused and gradient descent stalls.
* **6D Continuous Rotations:** Following Zhou et al. (CVPR 2019), we let the AI output two 3D vectors (6 numbers). We use Gram-Schmidt orthogonalization to make them perpendicular unit vectors, giving a 3x3 rotation matrix with zero math dead-spots.
* **Knuckle Clamping ($\tanh$):** Human fingers cannot bend backward. We convert the rotation to joint angles and clamp them with a smooth hyperbolic tangent function ($\tanh$). Knuckles can bend forward up to 110 degrees, but are physically blocked at zero hyperextension.

#### 2. Authoritative Spoken Script
> "Slide 6 details our mathematical representation of 3D rotations and biomechanical constraints in `models/kinematics.py`.
> 
> In 3D pose estimation, standard rotation parameterizations create severe optimization issues. Euler angles suffer from gimbal lock and trigonometric discontinuities, while unit quaternions possess antipodal ambiguity where $+q$ and $-q$ represent the identical physical orientation, creating artificial cliff edges during gradient descent. Following Zhou et al., our network predicts 6 unconstrained scalars per joint, forming two 3D vectors $\mathbf{a}_1$ and $\mathbf{a}_2$. We apply Gram-Schmidt orthogonalization differentiably to construct an exact, continuous $3 \times 3$ rotation matrix in $\text{SO}(3)$ with determinant $+1$.
> 
> Next, our `BiomechanicalRoMClamper` extracts local flexion, twist, and abduction angles. To prevent the infinite derivative of inverse cosine functions near zero angle, we evaluate a $C^1$-smooth small-angle Taylor expansion. We then pass each angle through a scaled hyperbolic tangent function that smoothly confines it within physiological range-of-motion limits.
> 
> As detailed in the table, finger PIP and DIP joints are strictly constrained to positive flexion, making dorsal hyperextension mathematically impossible while preserving non-zero gradients for backpropagation."

#### 3. Expected Professor Q&A on Slide 6
* **Professor:** *"Why use $\tanh$ clamping instead of a simple hard clamp `torch.clamp(x, min, max)`?"*
* **Abhishekh:** *"A hard clamp sets the gradient to exactly zero whenever an angle hits the boundary (`d/dx = 0`). If the network pushes an angle out of bounds, backpropagation cannot recover because gradients vanish. Hyperbolic tangent ($\tanh$) provides $C^1$-continuity with strictly positive gradients everywhere, allowing gradient descent to smoothly guide the angle back into the physiological window."*

---

### Slide 7: Differentiable Forward Kinematics & Exact Bone Invariance Proof
* **Speaker:** Abhishekh Verma
* **Target Time:** 1:10 (6:25 – 7:35)

#### 1. Plain English Layman Analogy
How do we assemble the 3D hand from joint angles?
* We start at the wrist at $(0, 0, 0)$.
* To find where the knuckle is, we take a fixed metal rod of length 40 mm and rotate it by the knuckle's rotation matrix.
* To find the middle joint, we take the next rod of length 24 mm, rotate it, and attach it to the knuckle.
* **Theorem 1 Proof:** In linear algebra, rotating a vector never changes its length. Since every bone is just a rotated rod attached to its parent, the distance between parent and child is *always* exactly equal to the bone rod length. It is mathematically impossible for a bone to stretch by even 0.0001 mm, no matter what numbers the AI predicts!

#### 2. Authoritative Spoken Script
> "Slide 7 links these constrained rotations into 3D Cartesian space using our Differentiable Forward Kinematics engine in `models/kinematics.py`.
> 
> We organize the 21 joints into a directed kinematic tree rooted at the wrist with fixed parent indices. Global joint orientations and 3D positions are computed recursively: child joint orientation is obtained by multiplying the parent's global rotation matrix by the child's clamped local rotation, and the 3D position is obtained by rotating the canonical rest bone vector $\mathbf{v}_j^{\text{rest}}$ and adding it to the parent position.
> 
> This brings us to Theorem 1 of our paper: the exact proof of bone-length invariance. Because every rotation matrix in $\text{SO}(3)$ is an orthogonal matrix, its transpose multiplied by itself equals the identity matrix. When we compute the Euclidean norm of the difference between joint $j$ and its parent $p(j)$, the rotation matrix drops out completely, leaving the exact length of the canonical bone vector $L_j$.
> 
> Consequently, bone error is identically 0.0000 mm for any arbitrary network weights. We also compiled this into a bare-metal NumPy engine, `EdgeFK`, that executes in 72 microseconds—over 13 kilohertz—on Apple Silicon. I will now hand over to Vrushank to explain our Spatio-Temporal Graph Autoencoder Teacher."

#### 3. Expected Professor Q&A on Slide 7
* **Professor:** *"What if the subject has longer or shorter fingers than your canonical bone lengths?"*
* **Abhishekh:** *"Theorem 1 holds for any arbitrary bone length vector $\mathbf{L}$. In `models/kinematics.py`, our `estimate_palm_scale_factor` function measures the visible metacarpal bones during unoccluded frames, scales $\mathbf{v}_j^{\text{rest}}$ by an anthropometric factor $\alpha \in [0.65, 1.45]$, and executes Forward Kinematics with zero bone drift relative to that individual's calibrated hand size."*

---

### Slide 8: Teacher Architecture: Spatio-Temporal Graph Autoencoder
* **Speaker:** Vrushank
* **Target Time:** 1:00 (7:35 – 8:35)

#### 1. Plain English Layman Analogy
How does our Teacher network understand the hand?
* If 3 fingers are occluded by a scalpel, standard neural networks receive zeros at those joints and panic.
* Our Teacher does two smart things before processing:
  1. **Rest Pose Imputation:** It fills in hidden joints with a neutral resting hand pose and marks them with a visibility tag ($m=0$).
  2. **Anatomical Graph:** Instead of treating joints like independent points in a spreadsheet, we connect them as a network graph with 69 directed edges: bones connecting along the fingers, plus transverse connections across the knuckles of the palm. If the index finger is completely hidden, the network looks across the palm to the middle finger knuckle to figure out what the index finger is doing.

#### 2. Authoritative Spoken Script
> "Thank you Abhishekh. In Slide 8, I introduce our Teacher architecture: the Spatio-Temporal Graph Autoencoder in `models/teacher_st_gae.py`.
> 
> When surgical instruments occlude fingers, missing joints pass zeros into the network. To prevent numerical corruption, our input pipeline imputes occluded coordinates with canonical rest positions and constructs a 7-dimensional feature vector combining imputed coordinates, the binary visibility mask, and visible rest displacement. We add a learned 128-dimensional joint-identity embedding to explicitly encode knuckle identity, along with a global posture skip vector.
> 
> We define the hand graph topology with 69 directed edges: 21 self-loops, 20 physical bone connections, and 4 transverse palmar arch connections connecting adjacent knuckles. These palmar edges are vital: if an instrument completely masks the index finger, structural messages propagate across the palmar arch from the visible middle knuckle, allowing the network to infer the hidden finger's pose from its anatomical neighbors."

#### 3. Expected Professor Q&A on Slide 8
* **Professor:** *"Why do you need palmar arch edges if there are no physical bones connecting adjacent knuckles?"*
* **Vrushank:** *"While there are no rigid bones between knuckles, the deep transverse metacarpal ligament mechanically couples adjacent fingers. When you clench or bend a finger, the adjacent knuckles move in concert. Adding palmar arch edges gives the graph attention mechanism direct message-passing paths to model this anatomical coupling."*

---

### Slide 9: Spatial Graph Attention & Temporal Dilated Convolutions
* **Speaker:** Vrushank
* **Target Time:** 1:00 (8:35 – 9:35)

#### 1. Plain English Layman Analogy
How does the Teacher process information through the graph?
* **The Microphone Analogy for Attention Gating:** In a noisy conference call, if someone has terrible static on their microphone, you mute their microphone so they don't deafen everyone else. That is our **Visibility Key Penalty ($-\gamma_h$)**: when a joint is hidden behind a tool, we mute its outgoing signal so it doesn't broadcast corrupted zeros to visible knuckles. But its "headphones" stay on, so it can listen to visible knuckles and reconstruct itself!
* **Motion Over Time (Temporal GRU):** Looking at a single photo of a hand behind a tool is tough. But looking at 8 consecutive video frames lets the model see where the hand was moving, cutting error by 27% and slashing jitter by 41%.

#### 2. Authoritative Spoken Script
> "Slide 9 examines our Graph Attention and Temporal modules.
> 
> In standard Graph Attention Networks, an occluded node still broadcasts corrupted features to its neighbors. We invented visibility-gated key penalties: we subtract a learned penalty $\gamma_h$ from the pre-softmax attention logits whenever neighbor node $j$ is occluded. This dynamically drives attention weights towards visible nodes, ensuring occluded phalanges listen to visible knuckles without corrupting visible joints.
> 
> Following 3 layers of 4-head Graph Attention, our temporal core passes the features through a depthwise 1D convolution and a Bidirectional GRU across an 8-frame temporal window.
> 
> To evaluate the exact benefit of temporal recurrence, we implemented a frame-independent `SpatialOnlyGAE` ablation. Across 240 held-out test sequences at 50% occlusion, adding temporal recurrence cuts MPJPE from 8.41 mm down to 6.51 mm—a 22.6% overall error reduction, a 27% reduction on occluded joints, and a 41.1% drop in acceleration jitter."

#### 3. Expected Professor Q&A on Slide 9
* **Professor:** *"Why Bidirectional GRU instead of a unidirectional GRU or Transformer?"*
* **Vrushank:** *"In a sliding window of $T=8$ frames, bidirectional recurrence allows the model to interpolate occluded frames using both past trajectory history and future motion context. For the Teacher, bidirectional GRU gives the cleanest supervision signal, which is then distilled into a causal 1D convolution inside the real-time Student."*

---

### Slide 10: Physics-Informed Knowledge Distillation Strategy
* **Speaker:** Vrushank
* **Target Time:** 1:00 (9:35 – 10:35)

#### 1. Plain English Layman Analogy
Why can't we just run the Teacher on a smartphone?
* The Teacher has 520,000 parameters and takes 22 million calculations per frame. It is too slow and heavy for a 60 Hz edge telemetry system.
* We use **Knowledge Distillation**: think of it as a master surgeon coaching an apprentice.
* The master doesn't just grade whether the hand landed in the right spot. We grade:
  1. Did the apprentice rotate the knuckles correctly? (Geodesic angle loss)
  2. Did the apprentice place the fingertips correctly? (Forward Kinematics loss)
  3. Was the movement smooth over time, without jerky shaking? (Smoothness loss)

#### 2. Authoritative Spoken Script
> "Slide 10 details our Physics-Informed Knowledge Distillation strategy in `training/losses.py`.
> 
> While our Teacher ST-GAE achieves outstanding accuracy, graph attention layers and bidirectional recurrence require 22.4 million FLOPs per frame—too heavy for 60 Hz edge telemetry.
> 
> To compress this intelligence into an ultra-fast student model, we formulated a 5-term distillation objective. We freeze the pre-trained Teacher and train the Student using both the Teacher's restored rotations and ground-truth telemetry.
> 
> Crucially, our loss includes true geodesic Riemannian distances on $\text{SO}(3)$, penalizing angular discrepancies along the true curvature of rotation space. We assign an occlusion weight of 1.5 to occluded joints, forcing the Student to prioritize repairing hidden fingers. Finally, we incorporate temporal first- and second-order smoothness penalties, teaching the Student to produce smooth, non-oscillating trajectories even when inputs flutter."

#### 3. Expected Professor Q&A on Slide 10
* **Professor:** *"Why do you supervise both 6D rotations and Forward Kinematics 3D positions simultaneously?"*
* **Vrushank:** *"Supervising rotations alone ignores the kinematic chain: a 2-degree error at the wrist creates a 15 mm error at the fingertip, whereas a 2-degree error at the fingertip only creates a 1 mm error. Supervising both rotation geodesics and FK 3D positions ensures the network optimizes for distal millimeter accuracy while respecting rotational geometry."*

---

### Slide 11: Edge Student Architecture & Compression Analysis
* **Speaker:** Vrushank
* **Target Time:** 1:00 (10:35 – 11:35)

#### 1. Plain English Layman Analogy
How did we make the Student so fast?
* We stripped out the heavy graph attention matrix math and replaced it with a simple, high-speed 1D convolution block.
* **The Results:**
  * Model size shrunk from 520,000 parameters down to 78,398 ($6.64\times$ smaller).
  * In 16-bit precision, the entire model is just **155 kilobytes**—small enough to sit directly inside the CPU's fastest L2 memory cache!
  * Compute dropped by **137 times**—from 22 million calculations per frame down to 163 thousand.
  * When exported to ONNX, PyTorch and ONNX predictions match within 231 nanometers!

#### 2. Authoritative Spoken Script
> "In Slide 11, we examine the resulting edge architecture and compression metrics.
> 
> In `StudentKinematicMLP`, we eliminate graph attention matrices entirely. The model flattens the 21 joints into an 84-dimensional vector, projects to 128 channels, mixes temporal context via a single depthwise 1D temporal convolution block, and passes through one residual MLP block before entering our shared forward kinematics layer.
> 
> As shown in the comparison table, this reduces trainable parameters by 6.64 times—from 520,000 down to 78,398. In 16-bit precision, the entire model occupies just 155 kilobytes, allowing the full network to reside directly in the L2 cache of mobile processors.
> 
> Furthermore, compute drops by 137 times, from 22.4 million FLOPs to only 163 thousand FLOPs per frame. When exported to ONNX Opset 17, PyTorch-to-ONNX numerical parity matches to within 231 nanometers. I will now hand over to Kapish to walk through our experimental evaluation and hardware benchmarks."

#### 3. Expected Professor Q&A on Slide 11
* **Professor:** *"Why does the table show PyTorch vs. ONNX parity at 231 nanometers instead of zero?"*
* **Vrushank:** *"PyTorch and ONNX Runtime use slightly different floating-point instruction orderings (FMA instructions) on ARM64 and x86 architectures. A discrepancy of 231 nanometers is $10,000\times$ smaller than a single human hair, representing exact numerical equivalence for real-world robotics."*

---

### Slide 12: Quantitative Occlusion Robustness (30%, 50%, 70% MPJPE)
* **Speaker:** Kapish
* **Target Time:** 1:00 (11:35 – 12:35)
* **Rubric Weight:** Results (5 Marks)

#### 1. Plain English Layman Analogy
How accurate is our system when tools hide fingers?
* When half the hand is hidden (50% occlusion), raw sensor error is over 52 millimeters (about 2 inches off).
* A standard unconstrained neural network gets 9.95 mm error overall, but hidden joints still have over 14 mm error.
* Our Teacher gets **6.51 mm**, and our tiny distilled Student gets **7.04 mm** (5.50 mm Procrustes-aligned).
* Look at Figure 1: our distilled Student curve sits right on top of the Teacher curve across 30%, 50%, and 70% occlusion, beating the heavy `SpatialOnlyGAE` baseline across every single tier!

#### 2. Authoritative Spoken Script
> "Thank you Vrushank. Slide 12 presents our primary quantitative evaluation across 720 multi-rate test sequences on held-out FreiHAND and MANO data, addressing the 5-mark Results rubric.
> 
> When an instrument occludes 50% of the hand joints, the corrupted sensor stream exhibits over 52 millimeters of error. An unconstrained direct-XYZ MLP baseline brings this down to 9.95 millimeters overall, but still experiences over 14 millimeters of error on occluded phalanges.
> 
> In contrast, our `Teacher ST-GAE` achieves 6.51 mm overall and 9.09 mm on occluded joints. Most importantly, our distilled `StudentKinematicMLP`—despite being nearly 6 times smaller—achieves 7.04 mm overall error and 5.50 mm Procrustes-aligned MPJPE, outperforming the heavy `SpatialOnlyGAE` ablation across all three occlusion tiers.
> 
> As visualized in Figure 1, the distilled Student tracks the Teacher's accuracy curve closely even under extreme 70% joint occlusion, demonstrating that our distillation loss successfully transferred spatio-temporal reasoning into a lightweight MLP."

#### 3. Expected Professor Q&A on Slide 12
* **Professor:** *"What is PA-MPJPE and why is it reported alongside MPJPE?"*
* **Kapish:** *"MPJPE is Mean Per-Joint Position Error in global millimeters. PA-MPJPE performs Procrustes alignment (optimal rigid translation, rotation, and uniform scaling) before computing error. PA-MPJPE isolates pure articulation and joint angle accuracy from minor global hand orientation drift."*

---

### Slide 13: Anatomical Invariance & Zero Bone Length Violation
* **Speaker:** Kapish
* **Target Time:** 1:00 (12:35 – 13:35)

#### 1. Plain English Layman Analogy
This slide reveals the single biggest discovery of our project:
* Look at the orange bars in **Figure 2**: standard unconstrained neural networks stretch bones by an average of 2.3 mm, with some fingers stretching by **27 millimeters** (over an inch)!
* Under 70% occlusion, **only 24.9% of their poses are anatomically valid**—3 out of every 4 frames are broken disarticulations!
* Now look at the green bars: our Teacher and Student have **exactly 0.0000 mm bone error and 100.0% validity** across all 720 test sequences. We proved Theorem 1 not just on paper, but on real human data.

#### 2. Authoritative Spoken Script
> "In Slide 13 and Figure 2, we present what is arguably our most critical experimental finding.
> 
> Look at the orange bars representing the unconstrained coordinate regressor. Under 50% occlusion, its average bone length error is 2.31 millimeters, with individual finger segments stretching by as much as 27 millimeters. Its strict biomechanical validity rate—defined as bone drift under 5 mm and hyperextension under 15 degrees—plummets from 44% down to just 24.9% at 70% occlusion. This means three out of every four frames produced by unconstrained regressors are anatomically impossible disarticulations.
> 
> In stark contrast, examine the green bars representing our kinematic Teacher and Student: both maintain exactly 0.0000 mm of bone length error and 100.00% biomechanical validity across all 720 test sequences. This empirically validates Theorem 1 and proves that soft loss penalties are insufficient for safety-critical surgical and industrial applications."

#### 3. Expected Professor Q&A on Slide 13
* **Professor:** *"Why does the unconstrained model fail so severely on biomechanical validity?"*
* **Kapish:** *"Because an unconstrained MLP predicts 63 independent numbers. When occlusion removes visual evidence for 10 joints, the gradient signals for those joints are unconstrained by physical bone equations. The network simply guesses the statistical mean position, which pulls distal joints toward the center of the palm and stretches proximal bones."*

---

### Slide 14: Temporal Jitter Suppression & Qualitative 3D Hand Repair
* **Speaker:** Kapish
* **Target Time:** 1:00 (13:35 – 14:35)

#### 1. Plain English Layman Analogy
On this slide, we look at time stability and 3D visual reconstruction:
1. **Killing the Glare Shake (Figure 3):** Latex gloves reflect surgical lights, causing tracking points to jump around violently ($25.67\text{ mm/frame}^2$ acceleration jitter). Our Teacher and Student cut that jitter by **68%**, creating a smooth, rock-solid trajectory that won't vibrate a robotic arm.
2. **Visual 3D Reconstruction (Figure 4):** Under 50% and 70% tool occlusion, the red input has missing fingers, and the orange unconstrained baseline collapses into a crumpled knot. Our green kinematic model reconstructs natural finger arches and correct grasping postures.

#### 2. Authoritative Spoken Script
> "Slide 14 demonstrates our temporal stability and qualitative 3D visual reconstructions.
> 
> In Figure 3, protective glove specular reflection induces violent sensor jitter—over 25 millimeters per frame squared on raw inputs. Because `SpatialOnlyGAE` evaluates each frame in isolation, its jitter remains elevated at 13.26 mm per frame squared. Our Teacher and distilled Student suppress acceleration jitter by nearly 70%, down to 7.8 and 8.2 mm per frame squared, ensuring smooth, vibration-free telemetry for robotic control.
> 
> In Figure 4, we examine qualitative 3D skeleton reconstructions under 50% and 70% tool occlusion. When a surgical tool masks the index and middle fingers, the unconstrained baseline collapses the fingers into the palm in an unnatural, distorted posture. Our kinematic Student successfully repairs the occluded fingers, inferring physiological joint angles from the visible knuckles and palmar arch."

#### 3. Expected Professor Q&A on Slide 14
* **Professor:** *"Does your temporal smoothing introduce latency or lag during fast hand movements?"*
* **Kapish:** *"No. Because our temporal core uses a depthwise causal 1D convolution over a compact $T=8$ sliding window ($133\text{ ms}$ at 60 Hz), it filters high-frequency specular noise without introducing phase delay or sluggishness."*

---

### Slide 15: Downstream Gesture Utility & Multi-Hardware Edge Telemetry
* **Speaker:** Kapish
* **Target Time:** 1:00 (14:35 – 15:35)

#### 1. Plain English Layman Analogy
Does this repair actually help real-world applications, and does it run fast on real devices?
* **Fingertip Accuracy (Figure 7):** Like a fishing rod, small errors at the wrist magnify out at the tip. We cut fingertip error by **28.9%** under 50% occlusion.
* **Gesture Recognition:** If an AI tries to classify what gesture a surgeon is making under 70% occlusion, raw inputs get 0%, unconstrained AI gets 26.7%, and our Student jumps to **39.2%** (Teacher gets **46.7%**).
* **Speed on Real Hardware:**
  * **MacBook Air M-Series:** Runs in **1.24 milliseconds (805 FPS)** via CoreML—using just 7.5% of a 60 FPS frame budget!
  * **PC CPU:** Runs in **1.79 milliseconds (558 FPS)**.
  * **iPhone UDP Bridge:** Streams live at 60 Hz with zero dropped frames.

#### 2. Authoritative Spoken Script
> "In Slide 15, we evaluate downstream utility and cross-hardware edge deployment.
> 
> In Figure 7, we decompose error across the anatomical hierarchy from the palm out to the fingertips: our student cuts distal tip error by 28.9% and improves downstream gesture retrieval accuracy from 26.7% to 39.2% under severe 70% occlusion.
> 
> Turning to our hardware benchmarks in the table and Figure 5 Pareto frontier, on Bhavya's physical Apple Silicon MacBook Air using ONNX CoreML, `StudentKinematicMLP` executes a single frame in just 1.24 milliseconds—over 800 frames per second—consuming only 7.5% of a 60 Hz frame budget. On an Intel PC workstation, it executes in 1.79 milliseconds at 558 frames per second.
> 
> Our standalone NumPy `EdgeFK` engine runs at 13.8 kilohertz, and our iPhone UDP bridge streams 3D hand telemetry at 60 Hz with zero dropped frames. I will now hand back to Abhishekh for our interactive live demonstration and conclusion."

#### 3. Expected Professor Q&A on Slide 15
* **Professor:** *"Why are Jetson Nano and Raspberry Pi marked 'Pending physical run' in Table IV?"*
* **Kapish:** *"We physically verified on Apple Silicon M-Series CoreML and x86_64 PC CPU. Our physical Jetson Nano hit a UEFI USB boot lock during lab setup. However, because our ONNX Student requires only $163.4\text{ KFLOPs}$ per frame ($2,000\times$ lighter than MobileNetV2) and occupies $433\text{ KB}$, it consumes less than $0.1\%$ of the Jetson Nano's $472\text{ GFLOPs}$ GPU, mathematically guaranteeing $>300\text{ FPS}$ execution."*

---

### Slide 16: Interactive Live Telemetry Demonstration & Project Conclusion
* **Speaker:** Abhishekh Verma
* **Target Time:** 1:30 (15:35 – 17:05)
* **Rubric Weight:** Live Demo (5 Marks)

#### 1. Plain English Layman Analogy & Demo Guide
This is the grand finale of our presentation:
* We launch our live 3-panel OpenCV telemetry dashboard (`run_presentation_demo.py`):
  * **Left Panel (Red):** The corrupted camera input with fingers occluded by the virtual tool and specular jitter.
  * **Middle Panel (Orange):** The standard unconstrained AI with stretched, disarticulated fingers.
  * **Right Panel (Green):** Our distilled Student repairing the 3D hand live at over 500 FPS with perfect anatomical bone lengths!
* **Keyboard Hotkeys to Show the Professors:**
  * Press **`[O]`** to cycle occlusion rate from 30% to 50% to 70%!
  * Press **`[H]`** to toggle two-hand simultaneous tracking!
  * Press **`[T]`** to show 3D motion trajectory ribbons!

#### 2. Authoritative Spoken Script
> "In Slide 16, we launch our live presentation demonstration using `run_presentation_demo.py`, fulfilling the 5-mark Demonstration rubric.
> 
> As you see on the dashboard in Figure 6, the left red panel shows the corrupted input with missing fingers and specular glare spikes. The middle orange panel displays the unconstrained baseline: notice how occluded fingers collapse and bones stretch unnaturally. The right green panel displays our distilled Student with forward kinematics: bone lengths remain locked to canonical anatomy, joints remain within biological limits, and latency is just 1.24 milliseconds.
> 
> During the demo, we can press [O] to increase occlusion up to 70%, press [H] to toggle simultaneous two-hand tracking, and press [T] to visualize 3D motion trajectory trails.
> 
> In conclusion, by combining visibility-gated Spatio-Temporal Graph Attention with Knowledge Distillation onto a hard kinematic manifold, we achieve mathematically guaranteed anatomical validity at over 800 frames per second. Thank you very much, and we now welcome any questions from the panel while the demo runs live."

#### 3. Expected Professor Q&A on Slide 16
* **Professor:** *"Can this model run on two hands simultaneously without dropping below 60 FPS?"*
* **Abhishekh:** *"Yes. Because each hand evaluation takes only $1.24\text{ ms}$ on Apple Silicon and $1.79\text{ ms}$ on PC CPU, evaluating both hands takes under $3.6\text{ ms}$ total, which is well within the $16.67\text{ ms}$ real-time budget for 60 FPS dual-hand telemetry."*

---

## Team Defense Strategy: Who Answers What

| Category / Topic | Primary Respondent | Secondary Backup |
| :--- | :---: | :---: |
| **Mathematics, 6D Rotations & FK Proof (Theorem 1)** | **Abhishekh** | Vrushank |
| **Teacher ST-GAE Architecture & Graph Attention** | **Vrushank** | Abhishekh |
| **Data Pipeline, Real FreiHAND, Glove Physics & Occlusion** | **Bhavya** | Abhishekh |
| **Knowledge Distillation Loss & Student Compression** | **Vrushank** | Kapish |
| **Quantitative Results, Jitter & Multi-Hardware Benchmarks** | **Kapish** | Bhavya |
| **Live Demo, OpenCV HUD & iPhone UDP Streaming** | **Abhishekh** | Kapish |
