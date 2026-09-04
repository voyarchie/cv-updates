# Dense Object Detection & Classification — Recent Advances

*Compiled 2026-Aug-30 (America/Los_Angeles).*

Next installment in the running CV-updates log. Earlier entries:
[Apr-30](../2026-Apr-30/2026-Apr-30_CV_updates.md),
[May-01](../2026-May-01/2026-May-01_CV_updates.md),
[May-02](../2026-May-02/2026-May-02_CV_updates.md),
[May-04](../2026-May-04/2026-May-04_CV_updates.md),
[May-05](../2026-May-05/2026-May-05_CV_updates.md),
[May-07](../2026-May-07/2026-May-07_CV_updates.md),
[May-08](../2026-May-08/2026-May-08_CV_updates.md),
[May-15](../2026-May-15/2026-May-15_CV_updates.md),
[May-16](../2026-May-16/2026-May-16_CV_updates.md),
[May-17](../2026-May-17/2026-May-17_CV_updates.md),
[Jun-09](../2026-Jun-09/2026-Jun-09_CV_updates.md),
[Jun-10](../2026-Jun-10/2026-Jun-10_CV_updates.md),
[Jun-12](../2026-Jun-12/2026-Jun-12_CV_updates.md),
[Jun-15](../2026-Jun-15/2026-Jun-15_CV_updates.md),
[Jun-16](../2026-Jun-16/2026-Jun-16_CV_updates.md),
[Jun-17](../2026-Jun-17/2026-Jun-17_CV_updates.md),
[Jun-19](../2026-Jun-19/2026-Jun-19_CV_updates.md),
[Jun-21](../2026-Jun-21/2026-Jun-21_CV_updates.md),
[Jun-22](../2026-Jun-22/2026-Jun-22_CV_updates.md),
[Jun-23](../2026-Jun-23/2026-Jun-23_CV_updates.md),
[Jun-24](../2026-Jun-24/2026-Jun-24_CV_updates.md),
[Jun-25](../2026-Jun-25/2026-Jun-25_CV_updates.md),
[Jun-27](../2026-Jun-27/2026-Jun-27_CV_updates.md),
[Jun-29](../2026-Jun-29/2026-Jun-29_CV_updates.md),
[Jun-30](../2026-Jun-30/2026-Jun-30_CV_updates.md),
[Jul-04](../2026-Jul-04/2026-Jul-04_CV_updates.md),
[Jul-07](../2026-Jul-07/2026-Jul-07_CV_updates.md),
[Jul-08](../2026-Jul-08/2026-Jul-08_CV_updates.md),
[Jul-10](../2026-Jul-10/2026-Jul-10_CV_updates.md),
[Jul-15](../2026-Jul-15/2026-Jul-15_CV_updates.md),
[Jul-17](../2026-Jul-17/2026-Jul-17_CV_updates.md),
[Jul-18](../2026-Jul-18/2026-Jul-18_CV_updates.md),
[Jul-21](../2026-Jul-21/2026-Jul-21_CV_updates.md),
[Jul-22](../2026-Jul-22/2026-Jul-22_CV_updates.md),
[Jul-24](../2026-Jul-24/2026-Jul-24_CV_updates.md),
[Jul-26](../2026-Jul-26/2026-Jul-26_CV_updates.md),
[Jul-27](../2026-Jul-27/2026-Jul-27_CV_updates.md),
[Jul-30](../2026-Jul-30/2026-Jul-30_CV_updates.md),
[Aug-01](../2026-Aug-01/2026-Aug-01_CV_updates.md),
[Aug-02](../2026-Aug-02/2026-Aug-02_CV_updates.md),
[Aug-04](../2026-Aug-04/2026-Aug-04_CV_updates.md),
[Aug-07](../2026-Aug-07/2026-Aug-07_CV_updates.md),
[Aug-10](../2026-Aug-10/2026-Aug-10_CV_updates.md),
[Aug-11](../2026-Aug-11/2026-Aug-11_CV_updates.md),
[Aug-13](../2026-Aug-13/2026-Aug-13_CV_updates.md),
[Aug-15](../2026-Aug-15/2026-Aug-15_CV_updates.md),
[Aug-16](../2026-Aug-16/2026-Aug-16_CV_updates.md),
[Aug-18](../2026-Aug-18/2026-Aug-18_CV_updates.md),
[Aug-19](../2026-Aug-19/2026-Aug-19_CV_updates.md),
[Aug-21](../2026-Aug-21/2026-Aug-21_CV_updates.md),
[Aug-22](../2026-Aug-22/2026-Aug-22_CV_updates.md),
[Aug-24](../2026-Aug-24/2026-Aug-24_CV_updates.md),
[Aug-26](../2026-Aug-26/2026-Aug-26_CV_updates.md),
[Aug-29](../2026-Aug-29/2026-Aug-29_CV_updates.md).

The last entry closed on the **radio-interferometric image** — a sky synthesized
from the correlations between pairs of antennas. This one keeps the
interferometer but shrinks the array to a *single*, kilometres-long Michelson and
points it not at the sky's light but at *spacetime itself*. The **gravitational-
wave (GW) detector output** is the primitive: a one-dimensional strain time
series `h(t)` that measures the fractional stretching of space, `ΔL/L ≈ 10⁻²¹`,
and whose natural computer-vision surface is the **whitened time–frequency
spectrogram** (the Q-transform / constant-Q scan) on which a binary merger writes
a rising "chirp" and the instrument writes a bestiary of "glitches." It is the
same interferometric lineage as [Aug-29](../2026-Aug-29/2026-Aug-29_CV_updates.md),
but where a radio array *forms an image of the sky*, a GW detector *rings like a
single wire* when the whole sky passes through it — and the entire vision problem
is to tell an astrophysical signal apart from a non-Gaussian, non-stationary,
glitch-ridden noise background.

> **Scope note & honest caveats.** This is a genuinely cross-disciplinary corner
> of ML: most of the strongest work lives in physics venues (*Phys. Rev. D*,
> *PRL*, *Class. Quantum Grav.*, *Mach. Learn.: Sci. Technol.*, *Nature Physics*)
> rather than CV venues, and it borrows CV architectures (CNNs, ResNets,
> transformers, normalizing flows, VAEs) wholesale. Links were gathered under
> scraping/API limits and are provided best-effort; where a landing page was
> flaky, an arXiv or DOI mirror is given. A handful of foundational works
> (matched filtering, BayesWave, the first Gravity Spy paper, the GW150914
> discovery) predate 2023 and are included as lineage anchors for otherwise-
> recent threads. Where an identifier is very recent and I could not fully
> re-verify it, it is flagged inline.

---

## Table of contents

1. [Why this pass: the GW time–frequency image as its own primitive](#1--why-this-pass-the-gw-timefrequency-image-as-its-own-primitive)
2. [The primitive — strain, whitening, the Q-transform, and matched filtering](#2--the-primitive--strain-whitening-the-q-transform-and-matched-filtering)
3. [Dense detection — deep-learning searches for buried signals](#3--dense-detection--deep-learning-searches-for-buried-signals)
4. [Dense classification — the glitch zoo and detector characterization](#4--dense-classification--the-glitch-zoo-and-detector-characterization)
5. [Parameter estimation as inference — simulation-based, amortized posteriors](#5--parameter-estimation-as-inference--simulation-based-amortized-posteriors)
6. [Denoising, glitch subtraction & generative waveforms](#6--denoising-glitch-subtraction--generative-waveforms)
7. [Instruments, open data, challenges & tools](#7--instruments-open-data-challenges--tools)
8. [Why a GW spectrogram is *not* a natural image](#8--why-a-gw-spectrogram-is-not-a-natural-image)
9. [Open problems / what to watch](#9--open-problems--what-to-watch)
10. [Sources](#10--sources)

## 1 · Why this pass: the GW time–frequency image as its own primitive

Six properties make the gravitational-wave detector output worth treating as a
first-class dense-vision surface rather than "a grayscale audio spectrogram":

1. **You never image a scene — you read a single scalar ringing.** A GW detector
   is one Michelson interferometer whose output is a single time series `h(t)`,
   the differential strain of two 4-km arms. There is no lens, no pixel grid, no
   direction of arrival in a single detector; a source anywhere on the sky
   couples into the same one-dimensional channel. The "image" is a *chosen
   representation* — the time–frequency spectrogram — not a measured picture.

2. **The signal is fantastically weak and the waveform is known.** A binary
   black-hole merger shifts the arm lengths by ~10⁻¹⁸ m, a thousandth of a proton
   width. The classical detection method is therefore **matched filtering**:
   correlate the data against a bank of hundreds of thousands of theoretical
   waveform templates. This is the modality's defining asymmetry — the *signal
   model* is exquisite (general relativity), while the *noise model* is the hard
   part.

3. **The noise is non-Gaussian, non-stationary, and full of "objects."** The
   detector produces thousands of transient **glitches** a day — blips,
   scattered-light arches, koi-fish, whistles, tomtes — each a localized shape in
   the spectrogram with its own instrumental cause. They are, quite literally, a
   dense-classification problem: a morphological zoo of noise transients that can
   mimic or hide a real signal. Detector characterization is inseparable from
   astrophysics here.

4. **Ground truth is almost entirely simulated.** Confirmed astrophysical events
   number in the low hundreds (the GWTC catalogues), while noise is effectively
   unlimited. Every supervised detector and inference network is trained on
   *simulated waveforms injected into real detector noise* — the modality's
   sim-to-real problem is not a nuisance, it is the training set.

5. **The interesting decision is a full Bayesian posterior, computed fast.** The
   science payoff of a detection is not a bounding box but the source parameters
   — masses, spins, distance, sky localization. Classical stochastic sampling
   (MCMC / nested sampling) takes hours to days per event; the modality's
   deep-learning pivot is toward **amortized, simulation-based inference** that
   returns a calibrated posterior in seconds, which is what makes real-time
   multi-messenger alerts possible.

6. **Latency is a first-class constraint.** A binary-neutron-star merger must be
   localized *before* or within seconds of merger so telescopes can catch the
   electromagnetic counterpart. This turns detection and inference into
   hard-real-time problems, exactly the regime where a trained network's
   millisecond forward pass beats an iterative classical search.

![A gravitational-wave detector shown as a dense detection-and-classification scene: a km-scale interferometer reads strain, a whitened Q-transform turns it into a time-frequency image of chirps and glitches, and the vision jobs are to detect the buried signal coherently across detectors, classify the glitch morphologies, and infer the source parameters.](assets/gw-as-dense-scene.svg)

## 2 · The primitive — strain, whitening, the Q-transform, and matched filtering

**The measurement.** Advanced LIGO, Advanced Virgo and KAGRA are power-recycled,
Fabry–Pérot Michelson interferometers with kilometre-scale arms, read out at the
dark port as a calibrated strain `h(t)` sampled at 16384 Hz. A passing GW induces
a quadrupolar `+`/`×` strain that lengthens one arm while shortening the other.
The **first direct detection**, GW150914 (2015), confirmed the whole paradigm and
opened the field ([Abbott et al., PRL 116, 061102, 2016](https://arxiv.org/abs/1602.03837)).

**From time series to image.** Raw strain is dominated by steeply colored noise
(seismic at low frequency, quantum/shot noise at high frequency, spectral lines
from the suspensions and mains). Two steps make it a vision surface: **whitening**
(divide by the estimated amplitude spectral density so every frequency
contributes equally) and a **constant-Q / Q-transform** time–frequency
decomposition — the *Omega scan* / Q-scan (Chatterji et al.) — which renders a
transient as a compact blob whose shape encodes its physics. A binary inspiral
appears as a **chirp**: a track sweeping up in frequency as the two bodies spiral
in. `GWpy` and the LIGO/Virgo detchar stack provide the standard tooling.

**Matched filtering — the workhorse to beat.** Because the waveform is predicted
by general relativity, the optimal linear detection statistic for a signal of
known shape in stationary Gaussian noise is the **matched filter**: cross-
correlate the whitened data against a **template bank** covering the mass/spin
parameter space, and threshold on the signal-to-noise ratio, then demand
consistency across detectors. This is what the production pipelines **PyCBC**,
**GstLAL** and **MBTA** do for modeled (compact-binary) searches, while
**coherent WaveBurst (cWB)** runs a weakly-modeled, coherence-based search for
generic bursts. Matched filtering is optimal *only* when the noise is stationary
and Gaussian and the template bank covers the signal — and it is expensive: the
bank grows explosively as detectors reach lower frequencies and longer signals.
Every ML detection method in §3 is, in one way or another, an attempt to match or
beat matched filtering's sensitivity at a fraction of the compute, on *real*
(non-Gaussian, glitchy) noise.

**Where the dense-vision jobs live.** The panels of the figure above split the
modality into the jobs that map onto the sections below: **(A) detection** — find
a faint, buried signal and confirm it coherently across the network; **(B) glitch
classification** — assign every noise transient a morphology and route it to a
veto or an instrumental fix; and **(C) parameter estimation** — turn a detection
into a Bayesian posterior over the source. An upstream step — **denoising and
glitch subtraction** — cleans the strain before any of them, and is itself a
learned regression problem.

![The deep-learning landscape over a gravitational-wave observation: data conditioning and denoising (whitening, DeepClean, BayesWave), detection (matched filtering to CNN deep filtering to residual networks and the MLGWSC-1 challenge), glitch classification (Gravity Spy to self-supervised and anomaly detection), and parameter estimation (MCMC to CVAE to normalizing-flow simulation-based inference), with a time-arc of model families beneath.](assets/gw-pipeline-landscape.svg)

## 3 · Dense detection — deep-learning searches for buried signals

Detection is the modality's headline job: decide whether a segment of whitened
strain (or its spectrogram) contains an astrophysical signal, at a false-alarm
rate low enough to claim a discovery, ideally faster and cheaper than a
template-bank matched filter.

**The founding results.** Two 2018 papers seeded the field by showing deep
networks could reproduce matched-filtering behaviour. **"Deep Filtering"**
(George & Huerta) trained CNNs directly on time-series strain to detect and
roughly estimate parameters of binary-black-hole signals in Advanced LIGO data
([Phys. Rev. D 97, 044039](https://arxiv.org/abs/1701.00008); companion
[Phys. Lett. B 778, 64](https://arxiv.org/abs/1711.03121)). In parallel,
**Gabbard et al.** showed a CNN can *reproduce the sensitivity of matched
filtering* — "matching matched filtering with deep networks"
([PRL 120, 141103, 2018](https://arxiv.org/abs/1712.06041)). Together they
reframed detection as a learnable classification problem and set the benchmark:
match the matched filter.

**Getting real about real noise.** The catch is that a CNN trained on Gaussian
noise degrades on real, glitchy data. The current thread pushes toward
**deep residual networks trained and evaluated on real detector noise**:
**AResGW** and related deep-ResNet searches report sensitivity competitive with
production pipelines on real O3 backgrounds
([Nousi et al., "Deep residual networks for gravitational-wave detection", arXiv:2211.01520](https://arxiv.org/abs/2211.01520)).
Curriculum training, whitening choices, and data-augmentation with real glitches
are what close the sim-to-real gap.

**A standardized benchmark: MLGWSC-1.** The field built its own mock data
challenge to make claims comparable. The **first Machine Learning Gravitational-
Wave Search Mock Data Challenge (MLGWSC-1)** pitted ML pipelines against
matched-filter baselines on both synthetic and real O3a noise, and is now the
reference benchmark for "does this ML search actually work at low false-alarm
rate" ([Schäfer et al., Phys. Rev. D 107, 023021, 2023](https://arxiv.org/abs/2209.11146)).
Its honest verdict — ML methods approach but do not yet uniformly beat matched
filtering, and struggle most at the very low false-alarm rates discovery requires
— set the agenda for the sections below.

**Unmodeled and multi-messenger searches.** Not every source has a template.
Weakly-modeled **burst** searches complement cWB, and the burst pipeline itself
has been *enhanced with ML* — the O3 all-sky burst search ran cWB boosted by a
machine-learning classifier
([arXiv:2210.01754](https://arxiv.org/abs/2210.01754)) — with learned coherent
burst searches (e.g. the MLy approach) an active parallel thread (identifier
best-effort). And because a binary-neutron-star inspiral is *in band* for up to a
minute before merger, **early-warning / pre-merger detection** is a natural ML
target: networks that forecast a merger and localize it *seconds before* it
happens to trigger electromagnetic follow-up — deep-learning BNS forecasting
([Wei & Huerta, arXiv:2010.09751](https://arxiv.org/abs/2010.09751)), pre-merger
sky localization ([arXiv:2301.03558](https://arxiv.org/abs/2301.03558)), and LSTM
early-warning detection ([arXiv:2402.04589](https://arxiv.org/abs/2402.04589)) are
a very active low-latency thread through O4. ML searches have now surfaced
candidate events beyond the standard pipelines
([arXiv:2407.07820](https://arxiv.org/abs/2407.07820), 2024).

**Why detection here is unusually hard.** The signal is rare, the background is
non-Gaussian and drifting, glitches produce loud non-astrophysical triggers, and
discovery requires false-alarm rates of one per thousands of years — a regime
where you have essentially no real positive examples and must extrapolate the
tail of a learned statistic. It is a needle-in-haystack detection problem where
the haystack actively imitates the needle.

## 4 · Dense classification — the glitch zoo and detector characterization

If detection is the headline, **glitch classification** is the modality's purest
dense-classification task — and the one where the citizen-science-plus-CNN recipe
this log has tracked elsewhere was pioneered.

**Gravity Spy — the anchor.** **Gravity Spy** fused Advanced LIGO detector
characterization, machine learning, and citizen science: volunteers on Zooniverse
label Q-scan spectrograms of glitches into ~20 morphological classes (blip,
koi-fish, scattered light, whistle, tomte, low-frequency burst, …), those labels
train CNN classifiers, and the classifiers in turn triage the millions of
transients no human could review
([Zevin et al., Class. Quantum Grav. 34, 064003, 2017](https://arxiv.org/abs/1611.04596);
dataset and CNN, [Bahaadini et al., *Information Sciences*, 2018](https://www.sciencedirect.com/science/article/pii/S0020025518301634)).
The **multi-view** design — feeding the classifier several fixed time-windows of
the same glitch — is a nice modality-specific inductive bias, since glitch
duration varies over orders of magnitude; the O4-era classifier fuses those views
with attention ([arXiv:2401.12913](https://arxiv.org/abs/2401.12913), ~94%
accuracy).

**Beyond fixed classes.** Two problems drive the recent work: the class taxonomy
keeps growing as detectors change between observing runs (an *open-set* problem),
and expert/volunteer labels are scarce and noisy. The field's response mirrors the
self-supervised pivot seen across this log:

- **Gravity Spy 2.0 and lessons learned** consolidate the dataset and workflow and
  push toward better handling of new and rare classes
  ([Zevin et al., *Eur. Phys. J. Plus*, 2024, arXiv:2308.15530](https://arxiv.org/abs/2308.15530)),
  with **vision transformers** now applied to the glitch spectrogram
  ([arXiv:2510.06273](https://arxiv.org/abs/2510.06273), 2025).
- **Self-supervised / representation learning and unsupervised clustering** learn a
  glitch (or signal) embedding without labels, then cluster to *discover* new
  morphologies — the "classify what you have, surface what you don't" recipe
  (early transfer-learning + clustering of LIGO glitches,
  [George, Shen & Huerta, "Deep Transfer Learning", arXiv:1706.07446](https://arxiv.org/abs/1706.07446);
  self-supervised signal ID, [arXiv:2302.00295](https://arxiv.org/abs/2302.00295);
  self-supervised non-Gaussian features,
  [arXiv:2403.04350](https://arxiv.org/abs/2403.04350)).
- **Anomaly / out-of-distribution detection** flags transients that fit no known
  class — the natural framing when the next observing run will introduce glitches
  no training set has seen. **GWAK** (Gravitational-Wave Anomalous Knowledge) uses
  recurrent autoencoders for semi-supervised anomaly detection spanning CBCs,
  glitches, and *novel* transients
  ([arXiv:2309.11537](https://arxiv.org/abs/2309.11537); *MLST* 5, 025020, 2024).

**Classifying the candidate itself: real signal vs. glitch.** A complementary task
takes a *detection candidate* and asks whether it is astrophysical or an
instrumental artifact from the low-latency alert products. **GWSkyNet** classifies
public GW candidates as real vs. noise using the sky-map and metadata
([Cabero, Mahabal & McIver, ApJL 904, L9, 2020](https://arxiv.org/abs/2010.11829), identifier best-effort;
refined pipeline **GWSkyNet II**, [arXiv:2408.06491](https://arxiv.org/abs/2408.06491), 2024),
and **GWSkyNet-Multi** extends it to distinguish glitch / BBH / NS-bearing
candidates, with an interpretable, O3-updated **GWSkyNet-Multi II**
([arXiv:2502.00297](https://arxiv.org/abs/2502.00297), 2025; explainability,
[arXiv:2308.12357](https://arxiv.org/abs/2308.12357)) — a fast triage layer for
the alert stream.

The through-line matches the rest of this log: glitch handling has become a
*representation-learning and open-set* problem, where the useful metrics are
label-efficiency, novel-class discovery, and calibrated confidence, not just top-1
accuracy on a fixed, saturated taxonomy.

## 5 · Parameter estimation as inference — simulation-based, amortized posteriors

Once a signal is found, the science is in the **posterior** over source parameters
`p(θ | data)` — component masses, spins, luminosity distance, inclination, and sky
position. This is a Bayesian inference problem, and it is where GW analysis has
been most thoroughly rewritten by deep learning, because it is a clean instance of
**simulation-based (likelihood-free) inference**: you can simulate `(θ → waveform
→ + noise → data)` endlessly, but classical sampling of the posterior is
punishingly slow.

**The classical baseline.** Stochastic samplers — MCMC and nested sampling in
**LALInference** and **Bilby** — are the trusted, accurate standard, but take
hours to weeks per event, which does not scale to a catalogue of hundreds (soon
thousands) of events, nor to real-time alerts.

**The arc (amortized approximate → normalizing flows → verified).**

- **Conditional variational autoencoders.** **VItamin** demonstrated amortized
  Bayesian posteriors for GW parameters from a CVAE, producing posteriors in a
  fraction of a second that broadly matched Bilby
  ([Gabbard et al., Nature Physics 18, 112, 2022](https://arxiv.org/abs/1909.06296)).
- **Normalizing-flow neural posterior estimation.** The dominant modern thread
  trains a conditional normalizing flow to map data → posterior directly.
  **Autoregressive flows** first showed full-parameter inference matching
  LALInference on real events
  ([Green, Simpson & Gair, Phys. Rev. D 102, 104057, 2020](https://arxiv.org/abs/2002.07656);
  complete GW150914 inference, [Green & Gair, arXiv:2008.03312](https://arxiv.org/abs/2008.03312)).
- **DINGO — production-grade amortized inference.** **DINGO** (Deep INference for
  Gravitational-wave Observations) combined normalizing flows with
  group-equivariant embedding and, crucially, **importance sampling** to *verify*
  and reweight the network posterior against the true likelihood — turning a fast
  approximation into a result with a computable accuracy guarantee
  ([Dax et al., "Real-time gravitational-wave science with neural posterior
  estimation", PRL 127, 241103, 2021](https://arxiv.org/abs/2106.12594);
  importance-sampled variant, [Dax et al., PRL 130, 171403, 2023](https://arxiv.org/abs/2210.05686), best-effort).
- **DINGO-BNS — low-latency binary-neutron-star inference.** The most recent step
  targets the hardest, longest signals and the tightest latency: near-instant
  posteriors (including sky localization) for BNS mergers to drive
  electromagnetic follow-up before or just after merger
  ([Dax et al., *Nature* (2025), DOI 10.1038/s41586-025-08593-z](https://doi.org/10.1038/s41586-025-08593-z);
  arXiv:2407.09602, best-effort).

**The tension that defines the stage.** A fast neural posterior is only useful for
science if it is *trustworthy*. The credible line of work therefore keeps the
physics likelihood in the loop — importance sampling gives an unbiased check and
an effective-sample-size diagnostic — rather than trusting the network's density
blindly. The other half of trust is **robustness to noise-distribution shift**:
a flow trained on one run's PSD must not silently mis-estimate on the next, and
adapting flow-based inference across changing noise is an explicit research target
([Wildberger et al., *Phys. Rev. D* 107, 084046, arXiv:2211.08801](https://arxiv.org/abs/2211.08801)).
That "verify the amortized posterior against the true likelihood, and track the
noise it was trained on" discipline is the field's answer to the same generative-
model-hallucination worry that runs through the reconstruction sections of the
MRI, photoacoustic, and radio-interferometry entries in this log.

## 6 · Denoising, glitch subtraction & generative waveforms

Two upstream/adjacent jobs are themselves learned problems and feed everything
above.

**Denoising & noise regression.** **DeepClean** is a convolutional network that
learns the coupling between *witness* auxiliary sensors and the strain channel and
subtracts the predicted technical noise, widening the sensitive band
([Ormiston et al., Phys. Rev. Research 2, 033066, 2020](https://arxiv.org/abs/2005.06534)),
now extended toward autonomous, coherence-guided operation
([Coherence DeepClean, arXiv:2501.04883](https://arxiv.org/abs/2501.04883), 2025)
and real-time ML noise regression
([arXiv:2306.11366](https://arxiv.org/abs/2306.11366)). Transformer denoisers such
as **WaveFormer** reconstruct a clean waveform from noisy strain
([arXiv:2212.14283](https://arxiv.org/abs/2212.14283)) — useful for visualization
and preprocessing, with the standing caveat that any learned denoiser must be
audited for *hallucinating* signal-like structure that was not there.

**Glitch subtraction.** When a glitch overlaps a real event (as in the celebrated
GW170817 BNS, which sat on top of a loud glitch), it must be modeled and
subtracted before parameter estimation. **BayesWave** models both glitch and
signal as sums of wavelets in a trans-dimensional Bayesian framework
([Cornish & Littenberg, Class. Quantum Grav. 32, 135012, 2015](https://arxiv.org/abs/1410.3835));
learned glitch models are an active complement, especially for the
overlapping-signal problem that next-generation detectors will face constantly.

**Generative waveforms & noise.** Deep generative models (autoencoders, GANs,
normalizing flows) are used to (a) build fast **surrogate waveform** emulators
that stand in for expensive numerical-relativity or effective-one-body models
inside a search or inference loop
([residual-error surrogate modeling, arXiv:2203.08434](https://arxiv.org/abs/2203.08434);
ML-generated surrogates feeding PE, [arXiv:2608.20222](https://arxiv.org/abs/2608.20222),
2026, best-effort), and (b) generate **realistic detector noise / glitches** for
augmentation and testing — turning the sim-to-real problem from a limitation into
a trainable data engine. Notably, a cleanly-verified *diffusion-model* generator
for GW waveforms or noise does not yet appear to exist in the literature: it is an
open direction, not an existing citation.

## 7 · Instruments, open data, challenges & tools

The methods above are pulled forward by an instrument network, a genuinely open
data ecosystem, and a small set of shared benchmarks.

**Instruments & network.**

- **Advanced LIGO** (Hanford, Livingston), **Advanced Virgo**, and **KAGRA** form
  the current network; coincident/coherent detection across ≥2 sites is what
  rejects local glitches and localizes sources. The O4 observing run (2023–)
  is the current data source; O5 and detector upgrades follow.
- **Next generation.** The **Einstein Telescope (ET)** and **Cosmic Explorer
  (CE)** on the ground, and **LISA** in space, will increase event rates by orders
  of magnitude and make **overlapping signals** and near-continuous inspirals the
  norm — which is precisely why fast, amortized ML inference and robust glitch
  handling are treated as prerequisites, not conveniences. The ET science case
  ([arXiv:2503.12263](https://arxiv.org/abs/2503.12263)) and the **LISA Data
  Challenges** ([arXiv:2204.12142](https://arxiv.org/abs/2204.12142); the "Sangria"
  set with overlapping massive-black-hole and Galactic-binary signals) are the
  emerging benchmarks, already driving SBI/flow-based inference for LISA sources
  ([arXiv:2603.20431](https://arxiv.org/abs/2603.20431), 2026, best-effort).

**Open data & catalogues.**

- **GWOSC** — the [Gravitational Wave Open Science Center](https://gwosc.org/) —
  releases calibrated strain, event data, and software, and is the substrate for
  essentially all external ML work.
- **GWTC** — the Gravitational-Wave Transient Catalogues — are the confirmed-event
  ground truth (the few hundred real positives the whole field trains against):
  GWTC-3 through O3b ([arXiv:2111.03606](https://arxiv.org/abs/2111.03606)) and the
  latest **GWTC-4.0** adding O4a ([arXiv:2508.18082](https://arxiv.org/abs/2508.18082),
  2025), which also defines the current O3→O4 domain shift.
- **Gravity Spy dataset** — the large labeled glitch-spectrogram corpus (on
  Zenodo) that made glitch classification a standard ML benchmark.
- **MLGWSC-1 dataset** — the mock-data-challenge data and injection sets that make
  detection claims comparable.

**Tools.** `GWpy` (time-series + Q-transform imaging), `PyCBC` and `GstLAL`
(matched-filter search), `Bilby`/`LALInference` (Bayesian PE baselines),
`BayesWave` (wavelet glitch/signal modeling), and the Gravity Spy / Zooniverse
stack remain the deployed baselines and reference implementations that any learned
method is measured against.

**The review to start from.** For the whole landscape, the community review
**"Enhancing Gravitational-Wave Science with Machine Learning"** is the canonical
map of where ML helps versus where it does not
([Cuoco et al., Mach. Learn.: Sci. Technol. 2, 011002, 2021](https://arxiv.org/abs/2005.03745)),
now updated by its 2024 successor surveying ML with the current detectors
([Cuoco et al., arXiv:2412.15046](https://arxiv.org/abs/2412.15046)).

## 8 · Why a GW spectrogram is *not* a natural image

Pulling the thread together — the recurring theme of this log is that each
modality's *physics* dictates the vision, and gravitational-wave data is a sharp
example precisely because it is so often mistaken for "just another spectrogram":

- **It is a representation, not a measurement.** The image is a *chosen* time–
  frequency transform of a 1-D scalar, not a picture of a scene. Change the
  Q-value or the whitening and the "object" changes shape; there is no
  view-invariant ground-truth image underneath.
- **The signal model is near-perfect; the noise model is the problem.** Unlike
  natural images, the thing you are looking for is predicted to exquisite
  precision by general relativity. All the difficulty — and all the ML value — is
  in modeling a non-Gaussian, non-stationary, glitch-ridden *background*.
- **The background actively imitates the foreground.** Glitches share the chirp's
  time–frequency real estate and sometimes its shape. Detection is adversarial
  against the instrument, not just against noise.
- **Discovery lives in the extreme tail.** Claiming an event requires false-alarm
  rates of one per thousands of years — a part of the statistic's distribution
  where you have no real training examples and must trust extrapolation.
- **Ground truth is simulated.** There is no "labeled real signal" set to speak
  of; there are a few hundred confirmed events and an endless supply of
  waveforms-injected-into-noise. Sim-to-real *is* the training paradigm.
- **The output is a calibrated posterior under a hard clock.** The useful result
  is a full Bayesian posterior delivered in seconds for multi-messenger follow-up
  — a requirement no natural-image detector faces.

## 9 · Open problems / what to watch

- **Beat matched filtering at discovery-grade false-alarm rates.** ML searches
  approach matched-filter sensitivity but not yet uniformly at the one-per-many-
  years false-alarm rates that discovery demands, especially on real noise. The
  MLGWSC series is the scoreboard to watch.
- **Trustworthy amortized inference.** Normalizing-flow posteriors are fast; the
  frontier is *guaranteeing* they are right — importance sampling, calibration
  under distribution shift, and coverage tests — before they replace stochastic
  samplers for catalogue science.
- **Open-set, cross-run glitch handling.** Each observing run brings new glitch
  classes and a shifted noise PSD (O3→O4→O5). Robust anomaly detection,
  continual/transfer learning, and novel-class discovery are the practical
  bottleneck for detector characterization.
- **Overlapping signals for ET/CE/LISA.** Next-generation detectors will have many
  signals in band simultaneously; joint detection-and-inference over overlapping
  chirps is a largely unsolved dense-vision problem the current benchmarks barely
  touch.
- **Foundation models & self-supervision for strain.** Pre-training on the vast
  unlabeled strain/spectrogram archive and fine-tuning for detection, glitch
  classification, and PE — the recipe that reshaped every other modality in this
  log — is now underway: **GraviBERT** applies BERT-style self-supervised
  pre-training to GW time series
  ([arXiv:2512.21390](https://arxiv.org/abs/2512.21390), best-effort), and
  transformers are being used for flexible, variable-length multi-detector PE
  ([arXiv:2512.02968](https://arxiv.org/abs/2512.02968)) and even end-to-end
  *population* inference directly from strain
  ([arXiv:2605.11274](https://arxiv.org/abs/2605.11274), 2026, best-effort).
- **Persistent sources: continuous & stochastic signals.** Beyond transients,
  deep learning is being pushed at **continuous-wave** searches (nearly-monochromatic
  signals from spinning neutron stars) — a huge parameter-space problem where CNNs
  trade sensitivity for compute
  ([Dreissigacker et al., arXiv:1904.13291](https://arxiv.org/abs/1904.13291);
  multi-detector/realistic noise, [arXiv:2005.04140](https://arxiv.org/abs/2005.04140);
  wide-parameter large-kernel CNNs, [arXiv:2408.07070](https://arxiv.org/abs/2408.07070))
  — and at the stochastic background; both are dense per-pixel detection problems
  in long-integration time–frequency maps.
- **Real-time, low-latency everything.** Pre-merger BNS detection, instant
  localization, and instant PE for multi-messenger astronomy push the whole stack
  toward millisecond inference and on-line, streaming operation.
- **Hallucination audits for generative denoising/reconstruction.** As with the
  learned-reconstruction sections elsewhere in this log, learned denoisers and
  generative waveform/noise models must be audited so they never invent
  signal-like structure — data-consistency and uncertainty reporting are
  intrinsic, not optional.

## 10 · Sources

**Reviews, context, baselines & the primitive** *(the cleanest modern expositions
of "why GW data is not a natural image" live in the introductions of the Cuoco
reviews and the MLGWSC-1 and DINGO papers below; the canonical discovery paper is
GW150914).*

- Observation of Gravitational Waves from a Binary Black Hole Merger (GW150914) — *PRL* 116, 061102 (2016), arXiv:1602.03837 — https://arxiv.org/abs/1602.03837
- Enhancing Gravitational-Wave Science with Machine Learning (review) — Cuoco et al., *Mach. Learn.: Sci. Technol.* 2, 011002 (2021), arXiv:2005.03745 (arXiv id best-effort; DOI 10.1088/2632-2153/abb93a) — https://arxiv.org/abs/2005.03745
- Applications of ML in GW research with current interferometric detectors (2024 review) — Cuoco et al., arXiv:2412.15046 — https://arxiv.org/abs/2412.15046
- Dawning of a New Era in GW Data Analysis via AI — a systematic review — arXiv:2311.15585 — https://arxiv.org/abs/2311.15585
- Machine Learning Applications in Gravitational Wave Astronomy (review) — arXiv:2401.07406 — https://arxiv.org/abs/2401.07406
- The PyCBC search for GWs from compact binary coalescence (matched filtering) — Usman et al., *Class. Quantum Grav.* 33, 215004 (2016), arXiv:1508.02357 — https://arxiv.org/abs/1508.02357
- Method for detection & reconstruction of GW transients with networks of advanced detectors (coherent WaveBurst) — Klimenko et al., *Phys. Rev. D* 93, 042004 (2016) — https://doi.org/10.1103/PhysRevD.93.042004
- QoQ: a Q-transform based test for gravitational-wave transient events — arXiv:2305.08257 — https://arxiv.org/abs/2305.08257
- Gravitational Wave Open Science Center (open data & software) — https://gwosc.org/
- GWpy (time-series analysis & Q-transform) — https://gwpy.github.io/ · PyCBC — https://pycbc.org/ · Bilby — https://lscsoft.docs.ligo.org/bilby/

**Detection — deep-learning searches**

- Deep Neural Networks to Enable Real-time Multimessenger Astrophysics (Deep Filtering) — George & Huerta, *Phys. Rev. D* 97, 044039 (2018), arXiv:1701.00008 — https://arxiv.org/abs/1701.00008
- Deep Learning for real-time gravitational wave detection and parameter estimation with Advanced LIGO data — George & Huerta, *Phys. Lett. B* 778, 64 (2018), arXiv:1711.03121 — https://arxiv.org/abs/1711.03121
- Matching Matched Filtering with Deep Networks for Gravitational-Wave Astronomy — Gabbard et al., *PRL* 120, 141103 (2018), arXiv:1712.06041 (arXiv id best-effort) — https://arxiv.org/abs/1712.06041
- Deep residual networks for gravitational-wave detection (AResGW) — Nousi et al., *Phys. Rev. D* 108, 024022 (2023), arXiv:2211.01520 — https://arxiv.org/abs/2211.01520
- MLGWSC-1: the first Machine Learning Gravitational-Wave Search Mock Data Challenge — Schäfer et al., *Phys. Rev. D* 107, 023021 (2023), arXiv:2209.11146 — https://arxiv.org/abs/2209.11146
- All-sky search for GW bursts in the O3 run with cWB enhanced by machine learning — arXiv:2210.01754 — https://arxiv.org/abs/2210.01754
- Deep learning for GW forecasting of neutron-star mergers (early warning) — Wei & Huerta, arXiv:2010.09751 — https://arxiv.org/abs/2010.09751
- Pre-merger sky localization of GWs from BNS mergers using deep learning — arXiv:2301.03558 — https://arxiv.org/abs/2301.03558
- LSTM for early-warning detection of gravitational waves — arXiv:2402.04589 — https://arxiv.org/abs/2402.04589
- Deep-learning detection & classification of GWs from neutron star–black hole mergers — arXiv:2210.15888 — https://arxiv.org/abs/2210.15888
- New Gravitational Wave Discoveries Enabled by Machine Learning — arXiv:2407.07820 (2024) — https://arxiv.org/abs/2407.07820

**Glitch classification & detector characterization**

- Gravity Spy: Integrating Detector Characterization, Machine Learning & Citizen Science — Zevin et al., *Class. Quantum Grav.* 34, 064003 (2017), arXiv:1611.04596 — https://arxiv.org/abs/1611.04596
- Machine learning for Gravity Spy: glitch classification and dataset — Bahaadini et al., *Information Sciences* (2018) — https://www.sciencedirect.com/science/article/pii/S0020025518301634
- Deep Transfer Learning: a new deep-learning glitch classification method for advanced LIGO — George, Shen & Huerta, arXiv:1706.07446 — https://arxiv.org/abs/1706.07446
- Advancing Glitch Classification in Gravity Spy: multi-view fusion with attention for O4 — arXiv:2401.12913 (2024), *Class. Quantum Grav.* DOI 10.1088/1361-6382/adf58b — https://arxiv.org/abs/2401.12913
- Gravity Spy: lessons learned and a path forward — Zevin et al., *Eur. Phys. J. Plus* (2024), arXiv:2308.15530 — https://arxiv.org/abs/2308.15530
- Vision Transformer for transient noise classification (Gravity Spy) — arXiv:2510.06273 (2025) — https://arxiv.org/abs/2510.06273
- Evaluating deep-learning models for multiclass classification of LIGO glitches — arXiv:2604.08796 (2026, best-effort) — https://arxiv.org/abs/2604.08796
- GWAK: Gravitational-Wave Anomalous Knowledge with recurrent autoencoders — arXiv:2309.11537, *MLST* 5, 025020 (2024) — https://arxiv.org/abs/2309.11537
- Self-supervised learning for gravitational-wave signal identification — arXiv:2302.00295 — https://arxiv.org/abs/2302.00295
- Extract non-Gaussian features in GW observation data using self-supervised learning — arXiv:2403.04350, *Phys. Rev. D* 111, 063520 — https://arxiv.org/abs/2403.04350
- GWSkyNet: a real-time classifier for public gravitational-wave candidates — Cabero, Mahabal & McIver, *ApJL* 904, L9 (2020), arXiv:2010.11829 (arXiv id best-effort) — https://arxiv.org/abs/2010.11829
- GWSkyNet II: a refined ML pipeline for real-time classification of public GW alerts — arXiv:2408.06491 (2024) — https://arxiv.org/abs/2408.06491
- GWSkyNet-Multi II: updated ML model for rapid classification of GW events — arXiv:2502.00297 (2025), *ApJ* — https://arxiv.org/abs/2502.00297 · explaining GWSkyNet-Multi predictions — arXiv:2308.12357 — https://arxiv.org/abs/2308.12357

**Parameter estimation — simulation-based inference**

- Bayesian parameter estimation using conditional variational autoencoders (VItamin) — Gabbard et al., *Nature Physics* 18, 112 (2022), arXiv:1909.06296 — https://arxiv.org/abs/1909.06296
- Gravitational-wave parameter estimation with autoregressive neural network flows — Green, Simpson & Gair, *Phys. Rev. D* 102, 104057 (2020), arXiv:2002.07656 — https://arxiv.org/abs/2002.07656
- Complete parameter inference for GW150914 using deep learning — Green & Gair, arXiv:2008.03312 — https://arxiv.org/abs/2008.03312
- Real-time gravitational-wave science with neural posterior estimation (DINGO) — Dax et al., *PRL* 127, 241103 (2021), arXiv:2106.12594 — https://arxiv.org/abs/2106.12594
- Neural importance sampling for rapid and reliable GW inference (DINGO) — Dax et al., *PRL* 130, 171403 (2023), arXiv:2210.05686 (best-effort) — https://arxiv.org/abs/2210.05686
- DINGO-BNS: real-time inference for binary neutron-star mergers — Dax et al., *Nature* (2025), DOI 10.1038/s41586-025-08593-z; arXiv:2407.09602 (best-effort) — https://doi.org/10.1038/s41586-025-08593-z
- Adapting to noise-distribution shifts in flow-based GW inference — Wildberger et al., *Phys. Rev. D* 107, 084046 (2023), arXiv:2211.08801 — https://arxiv.org/abs/2211.08801
- Tuning neural posterior estimation for gravitational-wave inference — arXiv:2403.02443 (2024) — https://arxiv.org/abs/2403.02443

**Denoising, glitch subtraction & generative models**

- Noise reduction in gravitational-wave data via deep learning (DeepClean) — Ormiston et al., *Phys. Rev. Research* 2, 033066 (2020), arXiv:2005.06534 — https://arxiv.org/abs/2005.06534
- Coherence DeepClean: toward autonomous denoising of GW detector data — arXiv:2501.04883 (2025) — https://arxiv.org/abs/2501.04883
- Demonstration of ML-assisted real-time noise regression in GW detectors — arXiv:2306.11366 — https://arxiv.org/abs/2306.11366
- WaveFormer: transformer-based denoising for gravitational-wave data — arXiv:2212.14283 — https://arxiv.org/abs/2212.14283
- BayesWave: a Bayesian wavelet method for reconstructing GW bursts & glitches — Cornish & Littenberg, *Class. Quantum Grav.* 32, 135012 (2015), arXiv:1410.3835 — https://arxiv.org/abs/1410.3835
- Deep residual-error & bag-of-tricks learning for GW surrogate modeling — arXiv:2203.08434 — https://arxiv.org/abs/2203.08434
- GW parameter estimation with ML-generated surrogate waveforms — arXiv:2608.20222 (2026, best-effort) — https://arxiv.org/abs/2608.20222

**Recent advances, next-generation detectors & data**

- GraviBERT: transformer-based self-supervised inference for GW time series — arXiv:2512.21390 (best-effort), *MLST* DOI 10.1088/2632-2153/ae5c58 — https://arxiv.org/abs/2512.21390
- Flexible gravitational-wave parameter estimation with transformers — arXiv:2512.02968 — https://arxiv.org/abs/2512.02968
- Enhancing reliability in ML GW parameter estimation with attention-based models — arXiv:2501.10486 — https://arxiv.org/abs/2501.10486
- End-to-end population inference from GW strain using transformers — arXiv:2605.11274 (2026, best-effort) — https://arxiv.org/abs/2605.11274
- Deep-learning continuous gravitational waves — Dreissigacker et al., *Phys. Rev. D* 100, 044009 (2019), arXiv:1904.13291 — https://arxiv.org/abs/1904.13291 · multiple detectors & realistic noise — *Phys. Rev. D* 102, 022005 (2020), arXiv:2005.04140 — https://arxiv.org/abs/2005.04140 · large-kernel CNNs for wide-parameter CW searches — arXiv:2408.07070 — https://arxiv.org/abs/2408.07070
- The Science of the Einstein Telescope — arXiv:2503.12263 (2025) — https://arxiv.org/abs/2503.12263
- The LISA Data Challenges — arXiv:2204.12142 (2022) — https://arxiv.org/abs/2204.12142 · accurate & efficient SBI for LISA MBHBs — arXiv:2603.20431 (2026, best-effort) — https://arxiv.org/abs/2603.20431
- GWTC-3: compact binary coalescences (O3b) — LVK, arXiv:2111.03606 — https://arxiv.org/abs/2111.03606 · GWTC-4.0 (O4a) — arXiv:2508.18082 (2025) — https://arxiv.org/abs/2508.18082

*Diagrams in this entry are hand-authored standalone SVG (no external URLs), with
explicit light-card / dark-panel fills so they render legibly in both light and
dark viewers. Some links were gathered under scraping/API limits and are provided
best-effort; where a landing page was unreachable, an arXiv or DOI mirror is
listed alongside, and recent 2023–2026 identifiers I could not fully re-verify are
flagged inline (arXiv IDs in the 25xx–26xx range correspond to 2025–2026 and
should be sanity-checked against the abstract page before citing downstream;
identifiers gathered under egress limits are marked "best-effort"). A few pre-2023
works (GW150914, matched filtering, BayesWave, the first Gravity Spy paper, Deep
Filtering) are included as lineage anchors for otherwise-recent threads. No
diffusion-model waveform/noise generator is cited because none could be verified
to exist — it is flagged as an open direction, not an existing result.*
