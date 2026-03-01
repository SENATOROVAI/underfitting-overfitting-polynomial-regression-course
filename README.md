# stepik: https://stepik.org/a/250384

# Underfitting, Overfitting, Polynomial Regression — Data Science

**Keywords:** underfitting, overfitting, polynomial regression, bias-variance tradeoff, machine learning, regularization, cross-validation, data science, model complexity

---

## 📌 Overview

This repository explains the fundamental concepts of:

* **Underfitting**
* **Overfitting**
* **Polynomial Regression**
* **Bias–Variance Tradeoff**
* **Regularization (Ridge / Lasso)**
* **Model Complexity in Machine Learning**

All concepts are explained mathematically and geometrically, with formulas written in LaTeX using `$$` for proper GitHub rendering and Google indexing.

---

# 1️⃣ Underfitting

## Definition

**Underfitting** occurs when a model is too simple to capture the true structure of the data.

Mathematically, we try to approximate:

$$
y = f(x) + \varepsilon
$$

but instead use an overly simple model:

$$
\hat{y} = w_0 + w_1 x
$$

when the true function is nonlinear.

---

## Characteristics

* High bias
* Low variance
* High training error
* High test error

---

## Example

True function:

$$
f(x) = x^2
$$

Fitted model:

$$
\hat{y} = w_0 + w_1 x
$$

A linear model cannot represent quadratic curvature → underfitting.

---

# 2️⃣ Overfitting

## Definition

**Overfitting** occurs when a model is too complex and learns noise instead of the true signal.

Instead of approximating:

$$
y = f(x) + \varepsilon
$$

the model tries to approximate:

$$
\hat{y} \approx f(x) + \varepsilon
$$

including the noise term.

---

## Characteristics

* Low training error
* High test error
* Low bias
* High variance

---

## Example

Polynomial regression with very high degree:

$$
\hat{y} = w_0 + w_1 x + w_2 x^2 + \dots + w_{20} x^{20}
$$

The curve passes through all training points but generalizes poorly.

---

# 3️⃣ Polynomial Regression

Polynomial regression expands the feature space:

$$
x \rightarrow (x, x^2, x^3, \dots, x^d)
$$

Model:

$$
\hat{y} = \sum_{k=0}^{d} w_k x^k
$$

Matrix form:

$$
\hat{y} = X w
$$

where:

$$
X =
\begin{bmatrix}
1 & x_1 & x_1^2 & \dots & x_1^d \
1 & x_2 & x_2^2 & \dots & x_2^d \
\vdots & \vdots & \vdots & & \vdots
\end{bmatrix}
$$

---

## Closed-form solution (OLS)

Minimize Mean Squared Error:

$$
J(w) = \frac{1}{n} |Xw - y|^2
$$

Solution:

$$
w^* = (X^T X)^{-1} X^T y
$$

---

# 4️⃣ Bias–Variance Tradeoff

Expected test error:

$$
\mathbb{E}[(y - \hat{f}(x))^2] =
\text{Bias}^2 + \text{Variance} + \sigma^2
$$

Where:

* **Bias** increases when the model is too simple
* **Variance** increases when the model is too complex
* $$\sigma^2$$ is irreducible noise

---

## Graphically

* Low degree polynomial → high bias
* High degree polynomial → high variance
* Optimal degree → minimal test error

---

# 5️⃣ Regularization

To prevent overfitting, we penalize large weights.

## Ridge Regression (L2)

$$
J(w) = |Xw - y|^2 + \lambda |w|^2
$$

Solution:

$$
w^* = (X^T X + \lambda I)^{-1} X^T y
$$

---

## Lasso (L1)

$$
J(w) = |Xw - y|^2 + \lambda |w|_1
$$

Promotes sparsity.

---

# 6️⃣ Model Complexity vs Error

As polynomial degree increases:

* Training error ↓
* Test error ↓ then ↑

This produces the classic U-shaped test error curve.

---

# 7️⃣ Practical Recommendations

### ✅ Avoid Underfitting

* Increase model complexity
* Add polynomial features
* Reduce regularization

### ✅ Avoid Overfitting

* Reduce polynomial degree
* Add regularization
* Use cross-validation
* Increase dataset size

---

# 8️⃣ Implementation Example (Python)

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    PolynomialFeatures(degree=5),
    Ridge(alpha=1.0)
)

model.fit(X, y)
```

---

# 9️⃣ SEO Keywords (for GitHub & Google)

machine learning, underfitting vs overfitting, polynomial regression tutorial, bias variance tradeoff, ridge regression, lasso regression, regularization in machine learning, model complexity, data science course, supervised learning theory

---

# 🔎 Related Topics

* Linear Regression
* Gradient Descent
* Ridge vs Lasso
* Cross-Validation
* Feature Engineering
* Regularization Paths

---

# 🎯 Target Audience

* Data Science students
* Machine Learning engineers
* Researchers studying model generalization
* Anyone learning bias-variance tradeoff

---

If you are studying **Data Science fundamentals**, this repository provides a clear mathematical and intuitive explanation of underfitting and overfitting using polynomial regression.

---

⭐ If this repository helps you understand the bias–variance tradeoff, consider giving it a star.
