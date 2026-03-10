This is an excellent and classic problem in matrix optimization. It sits at the intersection of **Matrix Completion** and **Robust Principal Component Analysis (RPCA)**. Since you are tasked with deriving a solution and eventually coding it, let's break down the exact mathematical nature of the problem, the core challenges, and the best mathematical framework to solve it.

Here is a detailed, step-by-step breakdown of how you should approach your derivation, algorithm design, and reporting.

---

### 1. Deconstructing the Problem
You are asked to solve:
$$ \min_{X \in \mathbb{R}^{m \times n}} \| Y - R \odot X \|_0 + \lambda \|X\|_* $$

*   **The First Term ($\| \cdot \|_0$)**: The $\ell_0$ "norm" simply counts the number of non-zero entries. Because $Y$ is $\pm 1$ and $R$ is a binary mask, $Y - R \odot X$ represents the mismatch between the observed entries in $Y$ and our predicted matrix $X$. The $\ell_0$ norm essentially applies a constant penalty (1) for every observed entry where $X_{ij}$ doesn't *exactly* match $Y_{ij}$.
*   **The Second Term ($\|X\|_*$)**: The nuclear norm (sum of singular values) is the standard convex relaxation for the rank of a matrix. It ensures the latent preference matrix $X$ remains low-rank.

#### The Core Challenge
1.  **Non-smoothness:** The nuclear norm is convex but non-smooth (non-differentiable), meaning you cannot use simple Gradient Descent.
2.  **Non-convexity & NP-Hardness:** The $\ell_0$ norm is non-convex, combinatorial, and NP-hard to optimize directly. 

Because of this, the prompt gives you permission to use a **"clearly justified approximation"**. 

---

### 2. The Best Approach: ADMM Framework
The absolute best framework to solve this is the **Alternating Direction Method of Multipliers (ADMM)**. ADMM is specifically designed to split complex objectives with multiple non-smooth terms into smaller, easy-to-solve subproblems.

You have two highly valid paths you can take for your derivation. **Path A** is the most mathematically standard "approximation" (convex relaxation), while **Path B** attempts to solve the $\ell_0$ problem directly (non-convex but often empirically stronger). I highly recommend writing your report on **Path A** as it is easier to justify globally, but Path B is also acceptable.

#### Path A: The $\ell_1$ Convex Relaxation (Highly Recommended)
In compressed sensing and RPCA, the standard, strictly justified approximation for the $\ell_0$ norm is the $\ell_1$ norm. By relaxing $\ell_0$ to $\ell_1$, your problem becomes fully convex:
$$ \min_{X \in \mathbb{R}^{m \times n}} \| R \odot (Y - X) \|_1 + \lambda \|X\|_* $$
*Justification to write in your report:* The $\ell_1$ norm is the tightest convex envelope of the $\ell_0$ norm on the unit $\ell_\infty$ ball. It converts an NP-hard exact matching problem into a computationally tractable robust regression problem, a well-known result from the foundational Robust PCA paper by Candès et al. (2011).

#### Path B: Direct $\ell_0$ Optimization (Hard Thresholding)
You can keep the $\ell_0$ norm exactly as written. While the overall problem remains non-convex, ADMM can still be applied using "Hard Thresholding." 
*Justification to write in your report:* While theoretical convergence guarantees for non-convex ADMM are weaker, keeping the exact $\ell_0$ penalty ensures we are strictly solving the objective provided, and empirical evidence in sparse optimization shows ADMM with hard thresholding performs excellently.

---

### 3. Deriving the ADMM Algorithm (The Math)
Let's formulate the ADMM updates. We will introduce an auxiliary variable $Z$ so that we can separate the nuclear norm from the observation loss.

**Rewrite the Optimization Problem:**
$$ \min_{X, Z} \quad \lambda \|X\|_* + \| R \odot (Y - Z) \|_p \quad \text{subject to} \quad X = Z $$
*(Where $p=1$ for relaxation, or $p=0$ for exact).*

**Formulate the Augmented Lagrangian:**
$$ \mathcal{L}_\rho(X, Z, \Lambda) = \lambda \|X\|_* + \|R \odot (Y - Z)\|_p + \langle \Lambda, X - Z \rangle + \frac{\rho}{2} \|X - Z\|_F^2 $$
Where $\Lambda$ is the dual variable matrix, $\rho > 0$ is the penalty parameter, and $\langle A, B \rangle = \text{Tr}(A^T B)$.

ADMM consists of three repeating updates until convergence:

#### Step 1: Update X (Singular Value Thresholding)
Isolate $X$ in the Lagrangian:
$$ X^{(k+1)} = \arg\min_X \left( \lambda \|X\|_* + \frac{\rho}{2} \left\| X - \left(Z^{(k)} - \frac{\Lambda^{(k)}}{\rho}\right) \right\|_F^2 \right) $$
**Solution:** This is the standard proximal operator for the nuclear norm.
$$ X^{(k+1)} = \text{SVT}_{\lambda/\rho}\left( Z^{(k)} - \frac{\Lambda^{(k)}}{\rho} \right) $$
*How SVT works:* Compute the SVD of the matrix $U \Sigma V^T$. Subtract $\lambda/\rho$ from the diagonal elements of $\Sigma$ (clipping negatives to 0). Reconstruct the matrix.

#### Step 2: Update Z (Element-wise Thresholding)
Isolate $Z$. Let $C = X^{(k+1)} + \frac{\Lambda^{(k)}}{\rho}$. The problem is completely independent for each matrix entry $(i,j)$:
$$ Z_{ij}^{(k+1)} = \arg\min_{Z_{ij}} \left( |R_{ij}(Y_{ij} - Z_{ij})|_p + \frac{\rho}{2} (Z_{ij} - C_{ij})^2 \right) $$

**If $R_{ij} = 0$ (Unobserved):**
The first term vanishes. The minimum is simply $Z_{ij}^{(k+1)} = C_{ij}$.

**If $R_{ij} = 1$ (Observed):**
*   *If you chose Path A ($\ell_1$ norm):* This evaluates to the Soft-Thresholding operator.
    $$ Z_{ij}^{(k+1)} = Y_{ij} - \text{SoftThresh}_{1/\rho}(Y_{ij} - C_{ij}) $$
    *(Where SoftThresh$_\tau(x) = \text{sign}(x) \cdot \max(|x|-\tau, 0)$)*
*   *If you chose Path B ($\ell_0$ norm):* This requires checking which is cheaper: setting $Z_{ij} = Y_{ij}$ (costing 0 for the $\ell_0$ term) or setting $Z_{ij} = C_{ij}$ (costing 1 for the $\ell_0$ term).
    $$ Z_{ij}^{(k+1)} = \begin{cases} Y_{ij} & \text{if } \frac{\rho}{2}(Y_{ij} - C_{ij})^2 \leq 1 \\ C_{ij} & \text{otherwise} \end{cases} $$

#### Step 3: Update $\Lambda$ (Dual Update)
$$ \Lambda^{(k+1)} = \Lambda^{(k)} + \rho(X^{(k+1)} - Z^{(k+1)}) $$

---

### 4. Synthetic Data Generation Strategy
When you transition to coding, here is how you should structure your data setup (as dictated by the prompt):
1.  **Ground Truth ($X^\star$):** Create two random matrices $U \in \mathbb{R}^{100 \times r}$ and $V \in \mathbb{R}^{100 \times r}$ where $r$ is the true rank (e.g., $r = 5$). Let $X^\star = U V^T$.
2.  **Observations ($Y$):** Apply the sign function to make it binary ratings: $Y^\text{true} = \text{sign}(X^\star)$. 
3.  **Mask ($R$):** Create a binary mask matrix of 100x100 where 1s are placed randomly with a certain probability (e.g., 30% observed).
4.  *(Optional but recommended)* **Noise:** Flip the signs of $Y$ on a small percentage of the observed entries (e.g., 5% noise) to show off the robust $\ell_0/\ell_1$ nature of your algorithm.

---

### 5. Plan for the Report
The prompt specifically outlines what must be in your short report. Structure your PDF exactly like this:
1.  **Formulation:** Briefly write out the objective and justify replacing/interpreting $\ell_0$ with your chosen relaxation (or state your intent to use hard thresholding).
2.  **Algorithm Derivation:** Lay out the ADMM steps exactly as derived above. Show the Augmented Lagrangian and the $X, Z, \Lambda$ update steps.
3.  **Objective Trend:** You will need a plot showing the objective function value (Equation 1) dropping over ADMM iterations.
4.  **Reconstruction Quality:** Don't just check MSE. The prompt says "sign accuracy on held-out observed entries". You need to mask some data initially, run your algorithm to get $X$, and report the percentage where $\text{sign}(X_{ij}) == Y^\text{true}_{ij}$ for the *unobserved* entries.
5.  **Runtime/Iterations:** State how many iterations ADMM took to converge (usually when $\|X - Z\|_F < \epsilon$) and the wall-clock time in seconds.

**Pro-tip for implementation later:** SVT can be computationally heavy. Because your matrix is $100 \times 100$, calculating the full SVD via `scipy.linalg.svd` every iteration will be fast enough, but usually $\lambda$ (regularization weight) and $\rho$ (ADMM learning rate) require a little trial and error tuning to get the rank to drop perfectly. Start with $\rho = 1.0$ and $\lambda = 5.0$ and tune from there.