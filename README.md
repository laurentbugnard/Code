# Power Law Correlations in Elastoplastic Models: Implications for Critical Phenomena

The behavior of complex amorphous materials under stress is a topic of great interest in many fields, including mechanics, materials science, and geophysics. In particular, earthquakes often display critical phenomena, where the system exhibits power law statistics and is characterized by scale-invariance and self-organized criticality. In this study, we analyze an elastoplastic model under quasistatic loading, with a focus on the effects of inhomogeneities in the yield stress field on the emergence of power law behaviors. We consider a field with power law correlations and explore its effects on the material response. This study can be seen as a followup to the paper "Elastoplastic description of sudden failure in athermal amorphous materials during quasistatic loading" by Popović et al. (2018).

## Generating the yield stress field $\sigma^Y$

### Power law correlations

We would like to generate an inhomogeneous 2D-map $s(\vec{x}), \vec{x} \in \mathbb{R}^2$, which fulfills two requirements:

- The correlations between two sites at a distance $r$ should follow a power law (scale invariant patch sizes).
- It should exhibit some randomness (random patch locations).

The intuitive idea, which will be formalized below, is to start with a random (e.g. Gaussian) field $u(x) \sim N(0,1), \forall x$ that is completely uncorrelated, i.e. $\langle u(x) u(x-r) \rangle = \delta_{r,0}$ and to "blur" it by convoluting it with a correlator $C(x)$. Hence, our $s$ will be decomposed as

$$ s(\vec{x}) = (C \ast u)(x) $$

Our aim is to find which form $C(x)$ should take in order to obtain the desired behavior of $s(x)$. To this end, we will first have to formalize what we mean by power law correlations. Then, the use of the Fourier Transform (FT) will naturally arise because of its close relation to the concepts of convolution and autocorrelation. Once we are in Fourier space, operations will simplify greatly and we will finally be able to find an expression for the FT of the correlator $\tilde{C}$, or at least be able to impose the desired correlations numerically.

#### Correlation function

We start by defining the correlation function $\Gamma (r)$ between two sites separated by $\vec{r}$. We want it to follow a power law distribution, i.e.

$$    \Gamma (r) := \frac{\langle s(\vec{x})s(\vec{x}-\vec{r})\rangle}{\langle s^2 \rangle} \sim |\vec{r}|^{-\alpha}, \alpha > 0. $$

>**Note:** Here, we implicitely assumed $\langle s(\vec{x}) \rangle = 0$ everywhere, thus avoiding the subtraction in the covariance term (numerator). Also we consider a system which is invariant under translation and isotropic, s.t. $\Gamma$ is only a function of $r = |\vec{r}|$ and $\langle s^2 \rangle$ is well-defined. In the following, to simplify notation, we will assume $\langle s^2 \rangle = 1$, since this is just a constant factor and we can effectively impose it. We will also omit vector notation: for example, $x-r$ must be understood as $\vec{x} - r\vec{e}$, where $\vec{e}$ is a unit vector in any direction.

To begin, we introduce the fact that the correlation function can be understood as a convolution, which will prove useful when we start working with its FT.
To this end, we suppose that the field $s(x)$ is large enough to assume that averaging over different realisations is equivalent to averaging over the whole system, i.e. we can write $\Gamma(r) =  \langle s(x)s(x-r)\rangle = \frac{1}{L^d} \int s(x) s(x-r) dx$, where $L$ is the length of the system and $d$ the spatial dimension. This simply traduces the fact that we want to look at correlations *within* the system, rather than correlations *between* different realisations of it (*WRITE A PARAGRAPH TO JUSTIFY --> GENERAL ARGUMENT WITH ANY QUANTITY*).
<!-- write paragraph with any quantity. Look at wiki: autocorrelation and when we can write it as an expectation. Wiener-Khinchin theorem -->
The above integral is known as the **autocorrelation** of the "signal" $s$ and can also be written in terms of the cross-correlation operator "$\star$":

$$\Gamma (r) = \frac{1}{L^d}(s\star s)(r).$$

>**Note** As claimed above, this is in fact equivalent to a convolution. Indeed, we can define $s'(x) := s(-x)$, which yields $\Gamma(r) = \frac{1}{L^d}(s\ast s')(r).$ In the following, we will stick to the autocorrelation point of view, but everything can be done in terms of convolutions.

The advantage of using this form of the correlation function is that we can simplify this operation by going to Fourier space. Remember that the convolution theorem states that the FT converts convolutions into multiplications ($f \ast g \stackrel{F}{\rightarrow} \tilde{f} \cdot \tilde{g}$). It is easy to prove an equivalent theorem for the cross-correlation, with the only difference that the left term is replaced by its complex conjugate ($f \star g \stackrel{F}{\rightarrow} \overline{\tilde{f}} \cdot \tilde{g}$). For the autocorrelation, we obtain the power spectrum: this result is called the Wiener–Khinchin theorem ($f \star f \stackrel{F}{\rightarrow} |\tilde{f}|^2$). In our case:

$$ \tilde{\Gamma}(q) = \frac{1}{L^d}|\tilde{s}|^2(q). $$

#### Introducing randomness

By now, we already have a pretty simple relationship between $\Gamma$ and $s$, with which we could already try to impose power law correlations. However, we don't want to end up with a field that is smoothly correlated, but rather want to see randomly located patches. As stated above, an idea to introduce randomness is to see split $s$ into two convoluted components, one for the randomness ($u$) and one for the correlations ($C$). In Fourier space, we have

$$ \tilde{s}(q) = \tilde{C}(q) \tilde{u}(q). $$

>**Note:** The only constraint on $u$ is that we require $\langle u \rangle_{spatial} = 0$, or equivalently, $\tilde{u}(q) = 0$. Otherwise, as one can deduce from the equation above, $\langle s \rangle_{spatial}$ can fluctuate greatly for different realisations of $\tilde{u}(0)$, which is an unwanted effect. This ie even more relevant considering that, in general, $\tilde{C}(0)$ can be very large.

We now show that the random component $\tilde{u}$ doesn't intervene in the correlation function, except for a multiplicative constant. Indeed, using the exact same reasoning as before, but backwards, we get

$$ \frac{1}{L^d}|\tilde{u}|^2 = \frac{1}{L^d}F[u \star u] = F[\langle u(x)u(x-r)\rangle] = F[\delta_{r,0}] = 1, $$

where we used the uncorrelatedness of the random (e.g. Gaussian) field. Thus, as long as we choose any uncorrelated random field, we have a very simple equation relating the correlation function with the correlator:

$$ \tilde{\Gamma} = |\tilde{C}|^2 \iff \Gamma = C \star C$$

#### Derivation of $C$

We can now impose any correlation function $\Gamma$ desired. Since only the complex norm of $\tilde{C}$ is involved, an infinite number of correlators can lead to the same result, where the relationship between its real and imaginary parts is the degree of freedom (in real space, we can see loosely see this as corresponding to the symmetry of $C$ around $x = 0$). For simplicity, we will opt for the natural choice $\tilde{C} \in \mathbb{R}$ with even symmetry: $\tilde{C}(q) = \tilde{C}(-q)$ [1]. In this convention,

$$ \tilde{C}(q) = \sqrt{\tilde{\Gamma}(q)},$$

which is our final expression for the correlator, provided we know $\tilde{\Gamma}.$
In practice, we can proceed in 2 differents ways:

1) $\alpha$ *method - Numerical:*
The first, and easiest method is to directly impose the power law: $\Gamma (r) \stackrel{!}{\sim} |r|^{-\alpha}$, and to proceed by numerically computing the expression for $\tilde{C}(q)$. In the code, we call this the "$\alpha$ method", as we can directly choose the exponent. However, in this case, we don't obtain an analytical expression for $\tilde{C}$.

2) $\beta$ *method - Analytical:*
We can try to bypass this restriction by analytically deriving $\tilde{\Gamma}(q) = F[\Gamma(r)] \sim F[|r|^{-\alpha}]$. Under certain conditions (*WHICH ONES?*), the FT of a power law remains a power law [2]. In this case, $\tilde{C}(q)$ will be a power law with exponent $\beta$, s.t.

$$
    \tilde{C}(q) \sim |q|^{-\beta} \iff \Gamma(r) \sim |r|^{2\beta - d},
$$
from which we finally derive a **relation between exponents**:

$$
    \alpha = d - 2\beta
$$

>[1]: This is also going to be valid for the correlator $C$ in real space, due to the nice property of the FT: $f \in \mathbb{R}^d$ even $\iff F[f] \in \mathbb{R}^d$ even.
\
\
>[2]: $F[|\vec{x}|^{-\gamma}] \propto |\vec{q}|^{\gamma -d}$ in $\mathbb{R}^d$
\
\
>**Note 1:**
Using scale-invariant correlations, we are cursed with ever-growing patch sizes as our system size $L$ grows. Thus, in practice, we will use a cutoff size $\xi$, such that $\xi \ll L$. In the $\alpha$ method case, we can directly add an exponential cutoff $\Gamma (r) \stackrel{!}{\sim} |r|^{-\alpha} e^{-r/\xi}$. In the $\beta$ method case, one can show (*CITE SOMETHING*) that using the correlator: $\tilde{C}(q) \sim \frac{1}{|q|^{\beta} + \xi^{-\beta}}$ will lead to similar (*OR SAME?*) results.
\
\
>**Note 2:**
Furthermore, as $\Gamma(r)$ is singular at $r=0$, we will impose $\Gamma(0) = k$ for some $k\geq 0$ (*FIND OUT WHICH VALUE SUITS BEST*). This corresponds to $\tilde{s}(0) = \tilde{C}(0) \tilde{u}(0) = 0$ which sets the spatial average of $s$ to 0. Without this, it would be subject to random and extreme fluctuations in different system realisations, since $\tilde{u}(0)$ is random.
\
\
**Note 3 (WIP):** Limitations
The procedure described in the two previous steps has limitations on the values of $\beta$:1) $\alpha = 2(1-\beta)$ implies that we have to restrict ourselves to  $\beta < 1$ in order to keep $\alpha > 0$. Indeed, as can be seen further below, choosing $\beta > 1$ leads to correlations that are not distributed as a power law.2) Choosing $\beta \ll 1$ also leads to a breakdown, as the "correlator width" gets comparable to the pixel size, and our correlations become insignificant (the field $s$ looks just like the initial gaussian field $u$).
<!-- TODO Make clear that this is only for method beta -->
<!-- TODO add limitation of continuous vs. discrete case -->

### Yield stress field $\sigma^Y$

We are now looking for a transformation $\sigma^Y$ = $f[s]$ which verifies the following conditions:

1. $\sigma^Y(x) > 0, \forall x$
2. $std(\sigma^Y) \ll \langle \sigma^Y \rangle$

For condition 1, the most natural choice of a positive function is the exponential. However, choosing $f[s] = e^s$ we could be significantly distorting our field $s$ by exploding bigger values, which would also conflict with condition 2. To avoid this, we use

$$f[s] = e^{ps}$$

instead, where we introduced a factor $p$ to scale down the range of $s$. Choosing $p \ll 1$, we can write $f[s] \approx 1 + ps$. This corresponds to rescaling and shifting $s$ to obtain

$$
\begin{align*}
\langle \sigma^Y \rangle &= 1, \\
std(\sigma^Y) &\approx p,
\end{align*}
$$

provided we initially normalized $s$ to $std(s) = 1$. (*WRITE NOTE ABOUT COMPLEX DISTRIBUTIONS*) This scale-shift also has the advantage to preserve power law correlations. Indeed the correlation functions of $s$ and $\sigma^Y$ differ only by a constant $\frac{p^2}{1+p^2}$:

$$
\begin{align*}
\Gamma_\sigma(r) &= \frac{\langle \sigma^Y(x) \sigma^Y(x-r)] \rangle - \langle \sigma^Y(x) \rangle \langle \sigma^Y(x-r) \rangle}{\langle (\sigma^Y)^2 \rangle} \\ &=\frac{\langle [1+ps(x)] [1+ps(x-r)] \rangle - \langle [1+ps(x)] \rangle \langle [1+ps(x-r)] \rangle}{\langle[1+ps]^2\rangle} \\ &= \frac{1 + p^2 \Gamma(r) - 1}{1 + p^2} = \frac{p^2}{1+p^2}\Gamma(r).
\end{align*}
$$

### Implementation (WIP, not up to date)

#### $\alpha$ vs. $\beta$ method

![0.2](images/avsb_b%3D0.2.png)
![0.4](images/avsb_b%3D0.4.png)
![0.6](images/avsb_b%3D0.6.png)
![0.8](images/avsb_b%3D0.8.png)
![1](images/avsb_b%3D1.png)
![1.2](images/avsb_b%3D1.2.png)

#### Examples for $\beta$ method

Below are two examples of the whole procedure,for $\beta = 0.8, \xi \to \infty$  and $L = 100$ and $1000$ respectively. In step 1, each of the 3 fields $u$, $C$ and $s$ are plotted along with their FT. In step 2, we show $s$ again along with $\sigma^Y$. It should look the same, since it just corresponds to a shift, but we also restricted the colorbar to a range of 1 std above and below the mean, to make differences more visible.

**L = 100**

|![gen1](images/gen1.png)|
|:---:|
|**Step 1: Generating power law correlations**|
|![gen1](images/fin1.png)|
|**Step 2: Yield stress field $\sigma^Y$**|

**L = 1000**

|![gen2](images/gen2.png)|
|:---:|
|**Step 1: Generating power law correlations**|
|![fin2](images/fin2.png)|
|**Step 2: Yield stress field $\sigma^Y$**|

#### Verification of power law correlations

To verify numerically that we indeed obtain the desired behavior, we could brute-force calculate correlation statistics on realisations of our system, which requires to compute the product $s(x)s(x-r)$ for each pair of pixels for a fixed distance $r$, and this $\forall r$. This computation is expensive and goes as $\mathcal{O}(N^3)$ However, there is an easier way: we already showed in equation 1.1 that $\Gamma (r)$ can be written as $\Gamma (r) = \frac{1}{L^d}  F^{-1}[\tilde{s}(q)\tilde{s}(-q)]$. As $s(x)$ is real, its FT is symmetric around 0 up to complex conjugation, which yields
$$
    \Gamma (r) = \frac{1}{L^d}  F^{-1}[|\tilde{s}|^2(q)].
$$
This is way more efficient computationally, as the Fast Fourier Transform algorithm has a $\mathcal{O}(N\log{N})$ complexity.

A few examples of numerical correlation measurements are shown in the figures below.

|![corr0.8](images/corr_beta=0.8.png)|
|:---:|
|**Power law behaviour for $\beta = 0.8$**|
|However, the measured exponent $\alpha_m = 0.64$ does not correspond to the predicted one $\alpha = 2(1-0.8) =0.4$. This is discussed further below.|

|![corr1.1](images/corr_beta=1.1.png)|
|:---:|
|**Breakdown for $\beta > 1$**|
|In this case, our analytical results are not valid anymore.|

|![corr0.3](images/corr_beta=0.3.png)|
|:---:|
|**Breakdown for $\beta \ll 1$**|
|In this case, the correlator is too narrow and we are left with an almost uncorrelated gaussian field.|

To better understand the relationship between $\beta$ and $\alpha_m$, several simulations were made for a range of $\beta$'s and $L$'s. The plot below shows the results, along with the predicted behaviour $\alpha = 2(\beta -1)$.

|![alpha_m_scan](images/results/alpha_m_scan.png)|
|:---:|
|**$\alpha_{m}$ vs. predicted $\alpha$**|
|We see that we fail to predict correct values for small $\beta$'s, no matter the system size $L$. This is due to resolution issues: the correlator has a width comparable to the pixel size. This doesn't depend on $L$ in our procedure, so increasing it doesn't solve the problem. For intermediary $\beta$'s, the prediction is slightly off, which is still an issue. For $\beta$'s close to 1, we see that $\alpha_m$ tends towards a plateau.|

**Note:** As can be seen in these figures, the procedure doesn't give optimal results, we still have to figure out how to enhance it.

## Elastoplastic model (EPM)

Our implementation of the EPM is basically the same as the one used in "Elastoplastic description of sudden failure in athermal amorphous materials during quasistatic loading" by Popović et al. (2018). The code was simplified and adapted by T. de Geus and L. Bugnard. In the following, we provide a brief description of the conventions used.

### Brief description

We use an elastoplastic model which is:

- defined on a **2D square** system of **linear dimension L** (in pixels),
- under **pure shear** conditions,
- which allows us to reduce tensorial stress and strain to **scalar** values (this is a reasonable approximation *CITE*).
- A **strain-driven** protocol is used,
- and the system is driven by **quasistatic loading**. This means that at each timestep, the overall stress is increased by the smallest possible value such that a single cell (the weakest) yields.
- This induces a **plastic strain** at the location of failure,
- and the stress release is then propagated through the system using an **Eshelby-like propagator**.
- This can lead to **avalanches**, i.e. new instabilities in other cells. The system thus needs to **relax**,
- which is done by sequentially choosing a **random cell among the unstable ones** and letting it yield in the same way, until all cells are stabilized.
- The **yield stresses** are **power law correlated** like described above.
- Furthermore, each step corresponds to an **increase in time** of ...(*COMPLETE*)
<!-- Compléter le timestep. Bien différencier relaxation step ou spatial particle failure step -->

### Some clarifications

For completeness, we provide further detail about the system's initialization, the propagator and the updating of yield stresses:

- **Preparation:**
The initial stress is randomized, using ...(*DESCRIBE PROCEDURE*), which keeps its overall sum (and mean) to zero. Since it can not be ruled out that some cells are already unstable, the system is initially relaxed.
- **Propagator:**
We use the Eshelby elastic stress propagator $G^E(r) = \frac{cos(4\theta)}{\pi r^2}$, which is then discretized using the convention described by Rossi et al. (2022). It is adapted for a strain-driven protocol and allows to very easily evolve the system for each site $\sigma_j$ at each event-driven step using the single equation $\sigma_j \rightarrow \sigma_j + \Delta\sigma_{ext} + \sum_{i} G_{j,i} \Delta\sigma_i$, where $\Delta\sigma_{ext}$ is the applied quasistatic load, the $\Delta\sigma_i$ are the local failure stress drops and $G_{j,i}$ is the discretized propagator.
- **Updating of yield stresses:**
After a plastic event, we choose to sample a new yield stress for the location of failure from a normal distribution. The mean corresponds to the initial value, thus approximately preserving the initial yield map. The standard deviation is a parameter that can vary and quantifies to what extent the local resistance of the medium can vary.
