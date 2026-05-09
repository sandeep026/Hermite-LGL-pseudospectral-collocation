# Hermite-Legendre-Gauss-Lobatto (HLGL) Direct Transcription

This repository contains a Python implementation of the **Hermite-Legendre-Gauss-Lobatto (HLGL)** direct transcription method for trajectory optimization, as described in the paper:

> Williams, P., "Hermite-Legendre-Gauss-Lobatto Direct Transcription in Trajectory Optimization," Journal of Guidance, Control, and Dynamics, Vol. 32, No. 4, 2009. 
> 
> 

## Overview

HLGL is a high-order direct transcription method that approximates optimal control problems by expanding state trajectories using local **Hermite interpolating polynomials**. The method leverages **Legendre-Gauss-Lobatto (LGL)** points for both collocation and quadrature, providing high accuracy for smooth problems while maintaining the sparsity benefits typically associated with Hermite-Simpson methods. 

### Key Features

* **Arbitrary Higher-Order:** Supports odd polynomial degrees $n \ge 3$ (e.g., $n=3$ corresponds to Hermite-Simpson). 


* **Unified Framework:** Integrates the computation of constraint residual equations (defects) and the performance index using Gauss-Lobatto quadrature. 


* **Efficient Implementation:** Uses matrix-vector operations with precomputed constant coefficients for faster constraint evaluation. 

---

## Mathematical Formulation

### 1. Optimal Control Problem

The algorithm solves a general Bolza-form problem: 

$$\text{Minimize: } \mathcal{J} = \mathcal{E}[x(t_0), x(t_f), t_0, t_f] + \int_{t_0}^{t_f} \mathcal{L}[x(t), u(t), t] dt$$

Subject to:

$$\dot{x}(t) = f(x(t), u(t), t)$$

And various path, box, and endpoint constraints. 

### 2. Discretization

The time interval is divided into $m$ subintervals. In each interval, the state is approximated by an $n^{th}$ degree polynomial: 

$$x(\tau) \approx a_0 + a_1\tau + a_2\tau^2 + \dots + a_n\tau^n, \quad \tau \in [-1, 1]$$



* **Nodes ($\tau_j$):** Used to form the interpolating polynomial. These are located at the LGL points $\xi_{2j-1}$. 


* **Collocation Points ($\zeta_j$):** Used to formulate residual equations. These are located at the LGL points $\xi_{2j}$. 



### 3. Residual Equations (Defects)

The NLP drives the difference between the polynomial derivative and the system dynamics to zero at the collocation points: 

$$\Delta = \frac{dx(\zeta_j)}{d\tau} - \frac{h_i}{2}f(x(\zeta_j), u(\zeta_j), t(\zeta_j)) = 0$$



---

## Implementation Details

The provided Python script implements the HLGL algorithm using **CasADi** and **Ipopt**.

### Control Parameterization

While the HLGL framework supports various control representations, this implementation assumes **piecewise constant control** ($u$) across each interval $m$.

### Code Structure

1. **LGL Point Generation:** Uses a helper class `LGL(N)` to compute nodes, collocation points, and quadrature weights.
2. **Polynomial Coefficients:** Solves for coefficients $a$ in terms of states and derivatives at the nodes using matrix inversion ($a = [A]^{-1}b$). 


3. **NLP Assembly:** 
  * **Decision Variables:** States at nodes (`x_seg_n1`) and piecewise constant controls (`u_c`).
  * **Continuity Constraints:** Ensures $x$ is continuous across interval boundaries.
  * **Defect Constraints:** Enforces dynamics at the mid-segment collocation points.
  * **Quadrature:** Approximates the cost function using $w_j$ weights. 





---

## Usage

### Prerequisites

* Python 3.x
* CasADi
* NumPy
* Matplotlib
* lgltools

[lgltools](https://github.com/sandeep026/lgltools) and can be installed via pip, poetry or other package manager.

### Example: Double Integrator (Toy Car)

The script solves a minimum-effort problem for a double integrator system:


$$\dot{x}_1 = x_2, \quad \dot{x}_2 = -x_2 + u$$

$$\text{Cost: } \int u^2 dt$$

To run the optimization:

```python
python hlgl_transcription.py

```

### Adjusting Order

To change the accuracy, modify the polynomial order $N$. It must be an **odd integer**:

* `N = 3`: Classic Hermite-Simpson. 
* `N = 5`: Fifth-order transcription. 

---

## References

[1] Williams, P. (2009). Hermite-Legendre-Gauss-Lobatto Direct Transcription in Trajectory Optimization. *Journal of Guidance, Control, and Dynamics*. 

[2] Herman, A. L., and Conway, B. A. (1996). Direct Optimization Using Collocation Based on High-Order Gauss-Lobatto Quadrature Rules. *Journal of Guidance, Control, and Dynamics*.
