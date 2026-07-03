# ROT-RH Minimal Finite `N=8000` Mangoldt/Jacobi GUE Discovery Reproducer

This repository contains a focused numerical reproducer for the current working finite-dimensional result in the ROT-RH / GUE cracking project.

The claim tested here is deliberately narrow:

> At fixed finite size `N=8000`, a Mangoldt-weighted self-adjoint Jacobi operator passes a local GUE statistics gate and beats a matched control panel under the selected benchmark score.

This is **not** claimed to be a proof of the Riemann Hypothesis. It is a reproducible finite numerical benchmark showing a GUE-like spectral signature in one arithmetic Jacobi construction.

---

## Main Result From the Reference Run

Reference command:

```powershell
python .\rot_rh_N8000_minimal_gue_discovery.py `
  --controls 50 `
  --out-dir rot_gue_N8000_minimal_discovery
```

The script evaluates 50 controls for each of four matched control modes:

```text
Gaussian controls
permuted controls
sign-flip controls
random-support controls
```

Total controls:

```text
4 modes × 50 controls = 200 controls
```

Reference output:

```text
MANGOLDT verdict=PASS_FINITE_N8000_GUE_BENCHMARK
score=1.118734526821640e-01
KS_GUE=1.092593767024393e-01
r_mean=5.969859240202753e-01
NV_GUE=3.067249938136101e+00
NV_Poisson=3.878340788981460e+00
local_gate=1
rigidity_gate=1
hard_gate=1

CONTROLS count=200
controls_beating=0
best_score=0.11207140297582752
margin=0.0001979502936634847
best_mode=gaussian
best_rep=27
p_strict=0.004975124378109453
```

The finite benchmark passed because:

```text
KS_GUE = 0.109259 < 0.13
r_mean = 0.596986, close to the GUE adjacent-gap-ratio target ≈ 0.5996
NV_GUE = 3.06725 < NV_Poisson = 3.87834
0 / 200 matched controls beat the Mangoldt candidate
```

---

## Candidate Parameters

The fixed candidate tested by this reproducer is:

```text
N         = 8000
prime_max = 4000
s0        = 0.60
sigma     = -1.0
shape     = cos4
K0        = 0.046209986233064583
S_rec     = 278.6908968682551
eps       = -0.1795770094514358
```

The candidate is not fitted during the run. The script evaluates this fixed candidate and compares it against matched controls.

---

## Operator Being Tested

The numerical object is a finite self-adjoint Jacobi matrix

```math
J_N=
\begin{pmatrix}
a_0 & b_0 & 0 & \cdots & 0 \\
b_0 & a_1 & b_1 & \ddots & \vdots \\
0 & b_1 & a_2 & \ddots & 0 \\
\vdots & \ddots & \ddots & \ddots & b_{N-2} \\
0 & \cdots & 0 & b_{N-2} & a_{N-1}
\end{pmatrix},
\qquad b_n>0.
```

The arithmetic input is the Mangoldt function

```math
\Lambda(n)=
\begin{cases}
\log p, & n=p^k \text{ for a prime } p \text{ and integer } k\ge 1,\\
0, & \text{otherwise.}
\end{cases}
```

The benchmark studies the local statistics of the eigenvalues

```math
\lambda_1\le \lambda_2\le \cdots \le \lambda_N
```

of the finite matrix `J_N`.

---

## GUE Gates

The script tests three finite spectral gates.

### 1. Nearest-neighbor spacing gate

After unfolding the spectrum, nearest-neighbor spacings are compared against the GUE Wigner-surmise reference.

```math
s_i = x_{i+1}-x_i,
```

where `x_i` are unfolded eigenvalues.

The reference density used for visual comparison is

```math
p_{\mathrm{GUE}}(s)=\frac{32}{\pi^2}s^2\exp\left(-\frac{4s^2}{\pi}\right).
```

The benchmark requires

```math
KS_{\mathrm{GUE}} < 0.13.
```

The reference run gives

```math
KS_{\mathrm{GUE}} = 0.1092593767.
```

### 2. Unfolding-free adjacent gap-ratio gate

The adjacent gap ratio is computed without unfolding:

```math
r_i=\frac{\min(\lambda_{i+1}-\lambda_i,\lambda_{i+2}-\lambda_{i+1})}
{\max(\lambda_{i+1}-\lambda_i,\lambda_{i+2}-\lambda_{i+1})}.
```

For GUE, the expected mean adjacent gap ratio is approximately

```math
\langle r\rangle_{\mathrm{GUE}}\approx 0.5996.
```

The reference run gives

```math
\langle r\rangle = 0.5969859240.
```

### 3. Mesoscopic number-variance gate

The number variance measures rigidity of the unfolded spectrum. For a window length `L`, it is

```math
\Sigma^2(L)=\operatorname{Var}\left(N[x,x+L]\right),
```

where `N[x,x+L]` counts unfolded levels in the interval.

The finite benchmark requires the observed number variance to be closer to the GUE reference than to the Poisson reference under the selected RMSE score:

```math
\mathrm{RMSE}_{\mathrm{GUE}} < \mathrm{RMSE}_{\mathrm{Poisson}}.
```

The reference run gives

```math
\mathrm{RMSE}_{\mathrm{GUE}}=3.0672499381,
\qquad
\mathrm{RMSE}_{\mathrm{Poisson}}=3.8783407890.
```

---

## Matched Controls

The script compares the fixed Mangoldt candidate against four matched control families.

| Mode | Purpose |
|---|---|
| `gaussian` | Tests whether a generic random input can match the gate. |
| `permuted` | Tests whether the arithmetic ordering matters. |
| `signflip` | Tests sensitivity to arithmetic sign structure. |
| `random-support` | Tests whether support alone explains the effect. |

Reference control summary:

```text
MODE gaussian       count=50 min_score=1.120714029758275e-01 median_score=4.631270254502352e-01 margin=+1.979502936634847e-04 beating=0 best_rep=27
MODE permuted       count=50 min_score=1.354093672559822e-01 median_score=7.792444020569359e-01 margin=+2.353591457381821e-02 beating=0 best_rep=37
MODE random-support count=50 min_score=1.305799964657209e-01 median_score=5.550938524306397e-01 margin=+1.870654378355685e-02 beating=0 best_rep=41
MODE signflip       count=50 min_score=1.163916983672022e-01 median_score=4.630703301418594e-01 margin=+4.518245685038197e-03 beating=0 best_rep=40
```

The strongest control was a Gaussian control, but it did not beat the Mangoldt candidate:

```math
\mathrm{score}_{\mathrm{Mangoldt}}=0.111873452682164,
```

```math
\mathrm{score}_{\mathrm{best\ control}}=0.11207140297582752.
```

The margin was small but positive:

```math
\Delta = \mathrm{score}_{\mathrm{best\ control}}-\mathrm{score}_{\mathrm{Mangoldt}}
       = 0.0001979502936634847.
```

Since lower score is better, the Mangoldt candidate wins this finite matched-control panel.

---

## Expected Output Files

A successful run creates an output folder such as:

```text
rot_gue_N8000_minimal_discovery/
```

Important files:

```text
DISCOVERY_SUMMARY.md
DAVID_MINIMAL_NOTE.txt
summary.json
metadata.json
data/all_trials.csv
data/mangoldt_vs_controls.csv
data/control_mode_summary.csv
data/operator_coefficients_mangoldt.csv
data/spectrum_mangoldt.csv
data/unfolded_levels_mangoldt.csv
data/spacings_mangoldt.csv
data/gap_ratios_mangoldt.csv
data/number_variance_mangoldt.csv
plots/control_score_by_mode.png
plots/gap_ratio_histogram_mangoldt.png
plots/number_variance_mangoldt.png
plots/operator_coefficients_mangoldt.png
plots/spacing_histogram_mangoldt.png
```

---

## Recommended Figures

### 1. Mangoldt vs matched controls

![Mangoldt vs matched controls](plots/control_score_by_mode.png)

This is the main discovery plot. Lower score is better. In the reference run, the Mangoldt candidate beats the best matched control in every mode.

### 2. Gap-ratio histogram

![Gap-ratio histogram](plots/gap_ratio_histogram_mangoldt.png)

This is the most robust local-statistics check because adjacent gap ratios do not require unfolding.

### 3. Number variance

![Number variance](plots/number_variance_mangoldt.png)

The observed finite number variance is not identical to the GUE curve, but it is closer to GUE than to Poisson under the benchmark RMSE score.

### 4. Spacing distribution

![Spacing histogram](plots/spacing_histogram_mangoldt.png)

The unfolded nearest-neighbor spacing distribution is compared against the GUE Wigner-surmise reference.

### 5. Jacobi coefficients

![Operator coefficients](plots/operator_coefficients_mangoldt.png)

This plot shows the finite self-adjoint Jacobi coefficients used to construct the `N=8000` operator.

---

## How To Run

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib mpmath tqdm
```

Run the minimal reproducer:

```powershell
python .\rot_rh_N8000_minimal_gue_discovery.py `
  --controls 50 `
  --out-dir rot_gue_N8000_minimal_discovery
```

For a quicker smoke test:

```powershell
python .\rot_rh_N8000_minimal_gue_discovery.py `
  --controls 5 `
  --out-dir smoke_N8000_gue_discovery
```

---

## What This Does and Does Not Claim

This repository claims:

```text
A fixed finite N=8000 Mangoldt-weighted self-adjoint Jacobi operator passes a selected GUE benchmark and beats 200 matched controls in the reference run.
```

This repository does **not** claim:

```text
A proof of the Riemann Hypothesis.
A proof of GUE universality.
A proof of arithmetic uniqueness.
A proof that the result survives all dimensions, all unfoldings, or all controls.
```

The result should be read as a finite numerical discovery and a reproducible benchmark target for further analysis.

---

## Suggested Wording For Sharing

A careful summary is:

> I reduced the computation to a minimal finite benchmark. At `N=8000`, the Mangoldt-weighted self-adjoint Jacobi operator passes a GUE local-statistics gate: KS-to-GUE is `0.1093`, the unfolding-free adjacent-gap-ratio mean is `0.59699` against the GUE target `≈0.5996`, and its number variance is closer to GUE than to Poisson. In a matched control panel of 200 Gaussian, permuted, sign-flip, and random-support controls, no control beats the Mangoldt candidate under the selected gate score. I am not claiming an RH proof here; the claim is only that this finite arithmetic Jacobi construction has a reproducible GUE-like spectral signature that survives this control panel.

---

## Reproducibility Notes

The control random number generation is deterministic and stable, using a SHA-256 based seeding scheme. The script writes raw datasets, plots, and JSON summaries so the result can be inspected without relying only on terminal output.

The most important files to inspect after a run are:

```text
DISCOVERY_SUMMARY.md
DAVID_MINIMAL_NOTE.txt
summary.json
data/mangoldt_vs_controls.csv
data/all_trials.csv
plots/control_score_by_mode.png
plots/gap_ratio_histogram_mangoldt.png
plots/number_variance_mangoldt.png
```

---

## Citation / Attribution

This is part of the ROT-RH finite spectral numerics project by Buddhika Weerasooriya.
