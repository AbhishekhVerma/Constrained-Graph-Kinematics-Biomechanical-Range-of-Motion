# FDS Final Presentation — Team Speaker Cue Sheet & Defense Playbook
**Date:** November 17, 2026 | **Total Time:** 12 Minutes Presentation + 5 Minutes Defense Q&A  
**Speakers:** Abhishekh Verma (Lead), Bhavya, Vrushank, Kapish  

---

## ⏱️ Master Presentation Timeline (12:00 Total)

```text
[0:00 - 2:00]   Abhishekh: Slide 1 (Motivation & Novelty) + Slide 2 (Objectives & Gap)
[2:00 - 4:15]   Bhavya:    Slide 3 (System Block Diagram) + Slide 4 (FreiHAND Data & Glove Physics)
[4:15 - 6:00]   Vrushank:  Slide 5 (6D Continuous SO(3), tanh RoM Clamper & Theorem 1 Proof)
[6:00 - 7:30]   Abhishekh: Slide 6 (Teacher ST-GAE & Visibility Graph Attention)
[7:30 - 9:00]   Kapish:    Slide 7 (Multi-Objective Distillation & Student Compression)
[9:00 - 10:15]  Abhishekh & Bhavya: Slide 8 (Occlusion Benchmark) + Slide 9 (Qualitative Fig 4 & 7)
[10:15 - 12:00] Kapish & Vrushank: Slide 10 (Hardware Profiling) + Slide 11 (LIVE DEMO)
[12:00 - 17:00] FULL TEAM: Defense Q&A with Examiners
```

---

## 🎙️ Individual Speaker Cards & Transitions

### 1. Abhishekh Verma (Slides 1, 2, 6, 8)
* **Slide 1 (0:00 - 1:00):**
  - *"In surgical teleoperation and smart gloved robotics, vision-based hand trackers fail when surgical tools occlude line-of-sight to the fingers or when protective latex gloves introduce specular glare."*
  - *"Unconstrained neural networks regress 63 independent X, Y, Z coordinates, producing stretched bones and hyperextended joints whenever visual evidence drops out."*
  - *"We solve this by replacing unconstrained regression with a 520K Teacher distilled into a 78K Student that predicts continuous 6D rotations on SO(3), decoded by an exact Differentiable Forward Kinematics engine."*
* **Slide 2 (1:00 - 2:00):**
  - *"State-of-the-art models like HandOccNet achieve 9.1 mm error on HO-3D, but require 37 million parameters and 33 ms on a desktop GPU. Meanwhile, soft loss penalties like Spurr et al. fail at test time, collapsing to 29.9% validity under 50% occlusion."*
  - *"Our model guarantees 0.0000 mm bone error and 100% validity in under 1.8 ms."*
  - **Handoff:** *"Now Bhavya will walk through our end-to-end architecture and hybrid data pipeline."*

* **Slide 6 (6:00 - 7:30):**
  - *"For our Teacher ST-GAE, we design a 3-layer 4-head Graph Attention Network with Soft Visibility Gating: k_v · γ_h(m_v). This prevents zeroed coordinates of occluded joints from polluting visible fingers."*
  - *"A Bidirectional GRU across T=8 frames reduces temporal jitter from 25.67 to 7.81 mm/s², outperforming our Spatial-Only ablation."*
  - **Handoff:** *"Now Kapish will explain how we compressed this 520K model into a sub-millisecond edge student."*

---

### 2. Bhavya (Slides 3, 4, 8b, 9)
* **Slide 3 & 4 (2:00 - 4:15):**
  - *"Our architecture operates in four stages: Data Ingestion, Teacher ST-GAE, Distillation, and Kinematic Manifold."*
  - *"We streamed all 32,560 real 3D hand poses from the University of Freiburg FreiHAND dataset, paired them with clinical surgical gestures, and enforced a zero-leakage 80/20 train/test split."*
  - *"Our SyntheticGloveTransform models 12% latex elasticity and AR(1) specular noise, while SyntheticOcclusionTransform enforces contiguous distal subtree priority at 30%, 50%, and 70% occlusion tiers."*
  - **Handoff:** *"Now Vrushank will detail the mathematical formulation of our 6D kinematic manifold."*

* **Slide 9 (9:30 - 10:15):**
  - *"Looking at Figure 4 and Figure 7, under 50% tool occlusion, the unconstrained baseline compresses fingers by 2.3 mm with severe hyperextension. Our model strictly preserves the anatomical phalanx hierarchy and achieves 93.8% gesture utility."*
  - **Handoff:** *"Now Kapish will present our edge hardware benchmarks and launch our live telemetry demo."*

---

### 3. Vrushank (Slides 5, 11b)
* **Slide 5 (4:15 - 6:00):**
  - *"To avoid gimbal lock in Euler angles and antipodal sign discontinuities in quaternions, we predict continuous 6D rotations. Gram-Schmidt orthogonalization guarantees an orthonormal SO(3) matrix with determinant +1."*
  - *"Our BiomechanicalRoMClamper maps raw angles through a C¹-smooth tanh function into physiological flexion and abduction limits."*
  - *"In Theorem 1, we prove that because Forward Kinematics computes joint positions via recursive rotation of rigid rest vectors, the distance between any joint and its parent is identically equal to the canonical bone length: ||P_j - P_p|| ≡ L_j, giving exactly 0.0000 mm error."*
  - **Handoff:** *"Now Abhishekh will present the Teacher ST-GAE architecture."*

---

### 4. Kapish (Slides 7, 10, 11)
* **Slide 7 (7:30 - 9:00):**
  - *"To deploy on edge microcontrollers and laptops, we distill the Teacher into StudentKinematicMLP (78,398 parameters, 433 KB ONNX) using a 5-term loss combining 6D angular L1, geodesic SO(3) distance, FK position loss, and temporal smoothness."*
  - *"The Student achieves a 6.64x parameter reduction and 137x FLOPs reduction while staying within 0.53 mm of the Teacher's accuracy."*
  - **Handoff:** *"Now Abhishekh and Bhavya will present our multi-rate benchmark results."*

* **Slide 10 & 11 (10:15 - 12:00):**
  - *"On Apple Silicon MacBook Air CoreML, our model executes in 1.24 ms (805 FPS)—13.4x faster than 60 FPS real-time requirements. On an x86 PC CPU, it runs in 1.79 ms (558 FPS)."*
  - *"Now we will launch our live 3-panel surgical telemetry demo (`run_presentation_demo.py`)."*
  - **[LAUNCH DEMO]:**
    - Show **Left (Red):** Corrupted input ($m=0$) behind the surgical tool.
    - Show **Center (Orange):** Unconstrained MLP collapsing and stretching bones.
    - Show **Right (Green):** Our 6D-FK Student reconstructing natural kinematics with $0.0000\text{ mm}$ bone error.
  - *"Thank you. We are now open for questions from the examination committee."*

---

## 🛡️ Q&A Defense Assignment Guide

| Question Topic | Designated Primary Speaker | Backup / Technical Support | Key Equation / Number to Cite |
| :--- | :--- | :--- | :--- |
| **Q1: Scale Variation & Different Hand Sizes** | **Vrushank** | Bhavya | Dynamic Palm-Scale Calibration ($s = \|\mathbf{x}_{\text{middle}} - \mathbf{x}_{\text{wrist}}\| / 100\text{ mm}$), verified bone error $7.63 \times 10^{-6}\text{ mm}$. |
| **Q2: Synthetic Glove vs Real Surgical Suits** | **Bhavya** | Abhishekh | Telemetry layer abstraction ($m=0$ tool bounding box); calibrated $12\%$ latex stiffness, $4.5\text{ mm}$ noise, $12\text{ mm}$ distal spike. |
| **Q3: Teacher vs Student Trade-Off** | **Kapish** | Vrushank | $6.64\times$ smaller ($78\text{K}$ vs $520\text{K}$ params), $1.24\text{ ms}$ latency vs $\sim 15\text{ ms}$ graph runtime; within $0.53\text{ mm}$ MPJPE of Teacher. |
| **Q4: Unconstrained Baseline Validity Collapse** | **Bhavya** | Vrushank | $29.95\%$ validity, $2.31\text{ mm}$ average stretch (spikes $>6\text{ mm}$); lack of kinematic tree constraints. |
| **Q5: Visibility Attention Key Gating** | **Abhishekh** | Kapish | $\mathbf{k}_{v,t} \cdot \gamma_h(m_{v,t})$ prevents zeroed coordinates from corrupting visible joints in the softmax denominator. |
| **Q6: Hardware Benchmarks & Jetson Status** | **Kapish** | Abhishekh | $163\text{ KFLOPs}$ budget mathematically fits within Jetson Nano 5W envelope ($1.4\text{ ms}$ projected); Apple CoreML & PC CPU physically measured. |
