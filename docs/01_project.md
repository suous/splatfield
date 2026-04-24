# Projection

Transforms each 3D Gaussian from world space to screen space.
The key challenge is preserving the Gaussian form under perspective projection,
which is non-linear. 3DGS solve this via **local affine approximation**
(first-order Taylor expansion) at each Gaussian's center, making the projected
distribution analytically tractable.

```bash
3D Gaussian (World)           Camera Space                  2D Gaussian (Screen)
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│  Σ_3D = RSSᵀRᵀ      │  ──►  │  Σ_cam = WΣWᵀ       │  ──►  │  Σ_2D = JΣ_camJᵀ    │
│                     │   W   │                     │   J   │                     │
│  Rotation + Scaling │       │  View Transform     │       │  Perspective Proj.  │
└─────────────────────┘       └─────────────────────┘       └─────────────────────┘
```

---

## 1. 3D Covariance

Guaranteed positive semi-definite via the decomposition
$\boldsymbol{\Sigma_{3D}} = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$.

Let $\mathbf{M} = \mathbf{R} \mathbf{S}$ where
$\mathbf{R}$ is the rotation from quaternion
$\mathbf{q} = (q_w, q_x, q_y, q_z)$:

$$
\mathbf{R} = \begin{pmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{pmatrix}
$$

and
$\mathbf{S} = \text{diag}(s_x, s_y, s_z)$ where
$s_i = e^{\sigma_i}$ (scales are stored as log-space values $\boldsymbol{\sigma}_{log}$
to ensure positivity):

$$
\mathbf{M} = \begin{pmatrix}
R_{00}s_x & R_{01}s_y & R_{02}s_z \\
R_{10}s_x & R_{11}s_y & R_{12}s_z \\
R_{20}s_x & R_{21}s_y & R_{22}s_z
\end{pmatrix}
$$

Then $\boldsymbol{\Sigma_{3D}} = \mathbf{M} \mathbf{M}^T$. Only the upper
triangle (6 values) is computed since $\boldsymbol{\Sigma_{3D}}$ is symmetric:

$$
\Sigma_{3D} = \begin{bmatrix}
\color{green}{\Sigma_{xx}} & \color{green}{\Sigma_{xy}} & \color{green}{\Sigma_{xz}} \\
\Sigma_{xy} & \color{green}{\Sigma_{yy}} & \color{green}{\Sigma_{yz}} \\
\Sigma_{xz} & \Sigma_{yz} & \color{green}{\Sigma_{zz}}
\end{bmatrix}
$$

---

## 2. 2D Covariance

The full projection from world-space covariance to screen space in a
single equation:

$$
\boldsymbol{\Sigma}_{2d} = \mathbf{J}\, \boldsymbol{\Sigma_{cam}}\, \mathbf{J}^T =
\mathbf{J}\, \mathbf{W}\, \boldsymbol{\Sigma_{3D}}\, \mathbf{W}^T \mathbf{J}^T
$$

where $\mathbf{W} = \mathbf{R}_{view}$ is the view rotation.
This factors into two stages:

### Camera Covariance

First rotate the 3D covariance into camera coordinates:

$$
\boldsymbol{\Sigma}_{cam} = \mathbf{W}\, \boldsymbol{\Sigma_{3D}}\, \mathbf{W}^T
$$

$$
\boldsymbol{\Sigma}_{cam} = \begin{bmatrix}
\color{green}{\sigma_{xx}} & \color{green}{\sigma_{xy}} & \color{green}{\sigma_{xz}} \\
\sigma_{xy} & \color{green}{\sigma_{yy}} & \color{green}{\sigma_{yz}} \\
\sigma_{xz} & \sigma_{yz} & \color{green}{\sigma_{zz}}
\end{bmatrix}
$$

### Jacobian of the Perspective Projection

The pinhole model maps a camera-space point $(x, y, z)$ to pixel
coordinates:

$$u = f_x \frac{x}{z} + c_x, \quad v = f_y \frac{y}{z} + c_y$$

Taking partial derivatives with respect to the camera-space coordinates:

$$
\mathbf{J} = \begin{pmatrix}
\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} & \frac{\partial u}{\partial z} \\[6pt]
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} & \frac{\partial v}{\partial z}
\end{pmatrix}
= \begin{pmatrix}
\frac{f_x}{z} & 0 & -\frac{f_x x}{z^2} \\[6pt]
0 & \frac{f_y}{z} & -\frac{f_y y}{z^2}
\end{pmatrix}
$$

The $z$-column captures how depth variation shifts the projected
position — steeper at close range (small $z$), flatter at distance.

### Project to 2D

Apply the Jacobian to $\boldsymbol{\Sigma}_{cam}$:

$$
\boldsymbol{\Sigma}_{2d} = \mathbf{J}\, \boldsymbol{\Sigma}_{cam}\, \mathbf{J}^T
= \begin{bmatrix} \color{green}{a} & \color{green}{b} \\ b & \color{green}{c} \end{bmatrix}
$$

---

## 3. Conic

The **conic** $\mathbf{C} = \boldsymbol{\Sigma}_{2d}^{-1}$ is the inverse 2D
covariance. It is precomputed per Gaussian so the rasterizer can evaluate the
Gaussian's contribution at any pixel without a per-pixel matrix inversion.

$$
\mathbf{C} = \boldsymbol{\Sigma}_{2d}^{-1} = \frac{1}{ac - b^2} \begin{pmatrix} c & -b \\ -b & a \end{pmatrix}
$$

### Usage in Rasterization

The 2D Gaussian PDF is:

$$
f(\mathbf{x}) = \frac{1}{2\pi\,|\boldsymbol{\Sigma}_{2d}|^{1/2}}
  \exp\!\left(-\frac{1}{2}\,\Delta^T\,\mathbf{C}\,\Delta\right)
$$

where $\Delta = (p_x - \mu'_x,\; p_y - \mu'_y)$ is the offset from the
projected mean. 3DGS drops the normalization constant and folds the peak
height into **opacity** instead:

$$
\alpha = \text{opacity} \cdot \exp\!\left(-\frac{1}{2}\,\Delta^T\,\mathbf{C}\,\Delta\right)
$$

Expanding the quadratic form:

$$
\begin{aligned}
\text{power} &= \frac{1}{2}\,\Delta^T\,\mathbf{C}\,\Delta \\[6pt]
             &= \frac{1}{2}\begin{pmatrix} \Delta_x & \Delta_y \end{pmatrix}
                \begin{pmatrix} a & b \\ b & c \end{pmatrix}
                \begin{pmatrix} \Delta_x \\ \Delta_y \end{pmatrix} \\[6pt]
             &= \frac{1}{2}(a\,\Delta_x^2 + c\,\Delta_y^2) + b\,\Delta_x\,\Delta_y
\end{aligned}
$$

- The **diagonal** terms $a\,\Delta_x^2$ and $c\,\Delta_y^2$ control the
  falloff along each axis — larger $a$ or $c$ means a tighter, sharper
  Gaussian in that direction.
- The **off-diagonal** term $b\,\Delta_x\,\Delta_y$ introduces the
  correlation between axes — this tilts/rotates the ellipse away from
  being axis-aligned. When $b = 0$ the ellipse axes are parallel to the
  pixel grid.

The conic makes the inner rasterization loop a cheap quadratic evaluation —
high near the center (power $\approx 0$), smoothly falling off toward the tails.
