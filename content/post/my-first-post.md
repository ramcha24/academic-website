---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Schmidt et. al.'s  Adversarial Robust Generalization requires more data"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2019-10-31T14:17:42-04:00
lastmod: 2019-10-31T14:17:42-04:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---


Trial post to see if I can blog my notes

## Mathematical Setup


### Standard Classification Error
Let $\mathcal{P} : \mathbb{R}^d\times \{-1,+1\} \rightarrow \mathbb{R}$ be the **data distribution** over input data and output label,  $(x,y)$. 
Let $f:\mathbb{R}^d  \rightarrow \{-1,+1\}$ be a **classifier** that maps input data to output label. 

The **standard classification error**, $\beta(f)$ is defined to be,
$$
\beta(f) = \mathrm{Pr}_{(x,y) \sim \mathcal{P}} \left[f(x) \neq y \right]
$$
Here, $\beta(f)$ is the probability that an input data and output label sampled from $\mathcal{P}$ is misclassified by the classifier $f$.

### Robust Classification Error
Let $\mathcal{P} : \mathbb{R}^d\times \{-1,+1\} \rightarrow \mathbb{R}$ be the **data distribution** over input data and output label,  $(x,y)$. 
Let $f:\mathbb{R}^d  \rightarrow \{-1,+1\}$ be a **classifier** that maps input data to output label. 
Let $\mathcal{B}:\mathbb{R}^d \times \mathfrak{P}(\mathbb{R}^d)$ be the **perturbation set** that maps input data to a set of points _near_ it. 

The **$\mathcal{B}$-robust classification error**, $\beta(f)$ is defined to be,
$$
\beta(f) = \mathrm{Pr}_{(x,y) \sim \mathcal{P}} \left[\exists \; x' \in \mathcal{B}(x) \text{ such that } f(x') \neq y \right]
$$

Here, $\beta(f)$ is the probability that an input data and output label $(x,y)$ sampled from $\mathcal{P}$ is misclassified by the classifier $f$ on any point $x'$ near $x$.

**Question 1** : 
If we sample a pair $(x,y) \sim P$ where $y \;|\; x$ is very unlikely, wouldn't this automatically result in a high probability that there exists a $x' \in \mathcal{B}(x)$ such that $f(x') \neq y$. 
The current definition of robust error includes the probability of these events. 
Shouldn't we be looking at a different definition of robust classification error? One that will not take such a misclassification into account?
Perhaps we should weight the probability stronger to bias it towards not penalizing for misclassifying edge cases

**Question 2** : 
In the gaussian model for data, for all $x$ there is a non-zero probability that the label is $y_1$ or $y_2$.
so, no matter what $x,y$ we sample and what our classifier $f$ is, there exists $x'$ such that $f(x')\neq y$
What exactly is the probabilty being calculated? 
Is it, 

$$
\mathrm{Pr}_{(x,y) \sim \mathcal{P}, \; (x',y) \sim P, \; x' \in \mathcal{B}(x)} \left[f(x') \neq y \right]  \quad ?
$$


We specifically define the set of perturbations w.r.t to the norm $l\_{\infty}$ as, 
$$\mathcal{B}^{\epsilon}\_{\infty} (x) := \\{ x' \in \mathbb{R}^d \text{ such that } \|x'-x\|\_{\infty} \leq \epsilon \\}$$

We refer to $\mathcal{B}^{\epsilon}\_{\infty}$ robustness as simply $l_{\infty}$-robustness. 

### An example classifier
For a parameter vector $w$, the **linear classifier** $f_w : \mathbb{R}\times\{-1,+1\}$ is defined as 
$$f_w(x):= \text{sign}(\langle w,x\rangle)$$ 


### The Gaussian Model for data distribution
The model of data $(x,y)$ is two spherical Gaussians with one component per output class. 
Let $\theta^{\star} \in \mathbb{R}^d$ be the per-class mean vector. 
Let $\sigma > 0$ be the variance parameter for each each Gaussian.

We define the following distributions over data $(x,y) \in \mathbb{R}^d \times \{-1,+1\}$ to be the $(\theta^{\star},\sigma)-$ Gaussian Model : 
- First, draw a random label $y \in \{-1,+1\}$ uniformly at random. 
- Sample the data point $x\in \mathbb{R}^d$ according to $\mathcal{N}(y \cdot \theta^{\star},\sigma^2I)$

i.e. the points that are labelled positive have the distribution $\mathcal{N}(\theta^{\star},\sigma^2I)$. 
and, the points that are labelled negative have the distribution $\mathcal{N}(-\theta^{\star},\sigma^2I)$. 

We want a classifier that separates these two points and has low generalization error (standard and robust)

We assume here that $\|\theta^{\star}\|_2 \approx \sqrt{d}$. Therfore, if the variance $\sigma^2$ is  too high then there is more overlap between the two gaussians. 

### Theorem 4: 
**Standard generalizaiton of a Linear classifier under the Guassian Model (with single sample)**
Let $(x,y)$ be drawn from a $(\theta^{\star},\sigma)-$Gaussian model where for some constant $c$, 
$$\|\theta^{\star}\|_2 = \sqrt{d}, \quad \sigma \leq c\cdot d^{\frac{1}{4}}$$

Choose parameter vector $\hat{w} := y\cdot x$ for the classifier. 

Then with high probability, the linear classifier $f_{\hat{w}}$ has classification error $\leq 1\%$. 

### Theorem 5:
**Robust error of Linear classifier under the Gaussian Model (under n samples)**
Lets draw $n$ samples i.i.d from the $(\theta^{\star},\sigma)-$Gaussian model : $(x_1,y_1),\ldots,(x_n,y_n)$. 
Again the parameters of the guassian model are such that for some constant $c_1$,
$$\|\theta^{\star}\|_2 = \sqrt{d}, \quad \sigma \leq c_1\cdot d^{\frac{1}{4}}$$

Choose for the classifier, the parameter vector $\hat{w} := \frac{1}{n}\sum_{i=1}^n y_ix_i$ which is the class-weighted sample mean. 

Then with high probability the linear classifier $f\_{\hat{w}}$ has $l^{\epsilon}_{\infty}$-robust classification error $\leq 1\%$ , **if**,  
$$
n \geq 
\begin{cases}
1 &\quad \text{ for } \epsilon \leq \frac{1}{4}d^{\frac{-1}{4}}, \\\ 
c_2 \epsilon^2 \sqrt{d} &\quad \text{ for } \frac{1}{4}d^{\frac{-1}{4}} \leq \epsilon \leq \frac{1}{4}. 
\end{cases}
$$ 

Therefore if the $\epsilon$ is small enough it is possible to learn a $l^{\epsilon}_{\infty}$-robust classification error given large enough number of samples $n$. 

### Theorem 6:
**Robust error of ANY classifier under the Gaussian Model (under n samples)**
Lets draw $n$ samples i.i.d from the $(\theta,\sigma)-$Gaussian model : $(x_1,y_1),\ldots,(x_n,y_n)$. 
Again the parameters of the guassian model are such that for some constant $c_1$,
$$\theta \sim \mathcal{N}(0,I), \quad \sigma = c_1\cdot d^{\frac{1}{4}}$$

Let $g_n$ be ANY learning algorithm that gives a binary classifier $f_n$. 

Then, the expected $l^{\epsilon}_{\infty}$-robust classification error of $f_n$ is at least $(1-\frac{1}{d})\frac{1}{2}$ , **if** 
$$n \leq c_2 \frac{\epsilon^2 \sqrt{d}}{\log(d)}.  
$$


Therefore if the sample size is smaller than the quantity given, then the **expected** robust classification accuracy of **ANY classifier** is **lower bounded** by $(1-\frac{1}{d})\frac{1}{2}$. 

To do better than that we necessarily need more samples (that is larger $n$). 

Note : A classifier that predicts either class every time will have robust error of $\frac{1}{2}$. Thus this lower bound is tight in that it says for a small enough $n$, ANY classifier is going to have (**in expectation**) error in the interval $\{\frac{1}{2}-\frac{1}{2d},1\}$, while trivial classifier already achieves expected error of $\frac{1}{2}$. 
This lower -bound becomes worse when we are dealing with data of larger dimensions!. This statement also holds for ANY $\epsilon >0$. 

### Bernoulli Model

The model of the data $(x,y)$ is defined on the hypercube $\\{-1,+1\\}^d$ with the two classes being opposite vertices. 

Let $\theta^* \in \\{-1,+1\\}^d$ be the per-class mean vector.
Let $\tau > 0$ be the class bias parameter. 

We define the following distributions over data $(x,y) \in \\{-1,+1\\}^d \times \\{-1,+1\\}$ to be the $(\theta^{\star},\tau)-$ Bernoulli Model : 
- First, draw a label $y \in \\{-1,+1\\}$ uniformly at random. 
- Sample the data point $x\in \\{-1,+1\\}^d$ by sampling each co-ordinate $x_i$ from the distribution, 

$$x_i = \begin{cases} 
y\cdot \theta_i^* &, \quad \text{with probability } \frac{1}{2}+\tau \\\ 
-y\cdot \theta_i^* &, \quad \text{with probability } \frac{1}{2}-\tau 
\end{cases}
$$

Note that, $\mathbb{E}[x_i] = 2\tau \cdot y\cdot \theta^*\_i $. 

The points that are labelled positive have the bernoulli distribution with expected value $2\tau \cdot \theta^*$. 

The points that are labelled negative have the bernoulli distribution with expected value $-2\tau \cdot \theta^*$. 

We want a classifier that separates these two points and has low generalization error (standard and robust)

As before, $\tau$, the bias dictates the amount of overlap between the two classes. If $\tau = \frac{1}{2}$, then zero overlap. If $\tau = 0$, then maximum overlap (no bias). 

### Theorem 8: 
**Standard generalizaiton of a Linear classifier under the Bernoulli Model (with single sample)**
Let $(x,y)$ be drawn from a $(\theta^{\star},\tau)$-Bernoulli model where for some constant $c$ such that , 
$$\tau \geq c\cdot d^{-\frac{1}{4}}$$

Choose parameter vector $\hat{w} := y\cdot x$ for the classifier. 

Then with high probability, the linear classifier $f_{\hat{w}}$ 
has classification error $\leq 1\%$. 

### Theorem 9:
**Robust error of ANY linear classifier under the Bernoulli Model (under n samples)**
Lets draw $n$ samples i.i.d from the $(\theta,\sigma)-$Gaussian model : $(x_1,y_1),\ldots,(x_n,y_n)$. 
Again the parameters of the bernoulli model are such that for some constant $c_1$,
$$\theta^* \sim \text{Uniform}\{-1,+1\}^d, \quad \tau = c_1\cdot d^{-\frac{1}{4}}$$

Let $g_n$ be ANY learning algorithm that gives a linear binary classifier $f_n$. 

Let $\epsilon < 3 \tau$ and $0 \leq \gamma \leq \frac{1}{2}$. 

Then, the expected $l^{\epsilon}_{\infty}$-robust classification error of $f_n$ is at least $\frac{1}{2}-\gamma$ , **if** 
$$
n \leq c_2 \frac{\epsilon^2 \gamma^2d}{\log(\frac{d}{\gamma})}.  
$$


Therefore if the sample size is smaller than the quantity given, then the **expected** robust classification accuracy of **ANY classifier** is **lower bounded** by $(1-\frac{1}{d})\frac{1}{2}$. 

To do better than that we necessarily need more samples (that is larger $n$). 

Note : A classifier that predicts either class every time will have robust error of $\frac{1}{2}$. Thus this lower bound is tight in that it says for a small enough $n$, ANY classifier is going to have (**in expectation**) error in the interval $\{\frac{1}{2}-\frac{1}{2d},1\}$, while trivial classifier already achieves expected error of $\frac{1}{2}$. 
This lower -bound becomes worse when we are dealing with data of larger dimensions!. This statement also holds for ANY $\epsilon >0$. 




