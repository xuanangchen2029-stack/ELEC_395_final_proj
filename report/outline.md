# 2-page report outline

## 1. Introduction
- Soft pneumatic actuators and why force/displacement calibration matters
- FEM cost and black-box model limitations
- PINN inverse identification as a middle ground

## 2. Reduced-order continuum model
- Inextensible planar centerline
- x_s = cos(theta), y_s = sin(theta)
- EI theta_ss + k_p p = 0
- clamped-free BCs

## 3. PINN inverse formulation
- network outputs x, y, theta
- trainable EI_eff and k_p
- physics loss + BC loss + data loss

## 4. Data generation
- synthetic BVP solutions
- optional FEM-generated data

## 5. Results
- identified parameters
- shape prediction
- sparse-data comparison with MLP
- optional blocked-force augmentation

## 6. Discussion
- benefits, limitations, and next step toward FEM/experimental validation
