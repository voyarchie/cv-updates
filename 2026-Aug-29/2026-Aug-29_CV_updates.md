# Dense Object Detection & Classification — Recent Advances

*Compiled 2026-Aug-29 (America/Los_Angeles).*

Next installment in the running CV-updates log. Earlier entries on
`main`:
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
[Aug-26](../2026-Aug-26/2026-Aug-26_CV_updates.md).

The last entry closed on the **weather-radar volume** — a spinning beam that
paints a multi-channel polar scene of storms. This one keeps the antenna but
moves it off the ground and points it at the sky, and swaps the single dish for
an *array*. The **radio-interferometric image** is the primitive: a picture of
the sky that is never measured directly but synthesized from the correlations
between antenna pairs — a sparsely sampled Fourier transform that must be
deconvolved before anything can be detected or classified in it. It is the
astronomical counterpart to the optical survey image of
[Aug-22](../2026-Aug-22/2026-Aug-22_CV_updates.md), but with a fundamentally
different measurement physics, and it is about to become one of the largest
dense-vision problems on Earth as the [Square Kilometre
Array](https://www.skao.int/) comes online.

> **Scope note & honest caveats.** This is a fast-moving, cross-disciplinary
> corner of ML where a good deal of the strongest work lives in astronomy
> journals (MNRAS, A&A, PASA, RASTI) rather than CV venues. Links were gathered
> under scraping/API limits and are provided best-effort; where a landing page
> was flaky, an arXiv or DOI mirror is given. A handful of foundational works
> (CLEAN, PyBDSF, Aegean, MiraBest, Radio Galaxy Zoo) predate 2023 and are
> included as lineage anchors for otherwise-recent threads. Where an item is
> very recent and I could not fully re-verify the identifier, it is flagged
> inline.

---

## Table of contents

1. [Why this pass: the interferometric image as its own primitive](#1--why-this-pass-the-interferometric-image-as-its-own-primitive)
2. [The primitive — visibilities, the dirty beam, and CLEAN](#2--the-primitive--visibilities-the-dirty-beam-and-clean)
3. [The imaging inverse problem — learned deconvolution & diffusion priors](#3--the-imaging-inverse-problem--learned-deconvolution--diffusion-priors)
4. [Dense detection — source finding at survey scale](#4--dense-detection--source-finding-at-survey-scale)
5. [Dense classification — radio-galaxy morphology & foundation models](#5--dense-classification--radio-galaxy-morphology--foundation-models)
6. [RFI flagging — segmentation in the dynamic spectrum](#6--rfi-flagging--segmentation-in-the-dynamic-spectrum)
7. [Instruments, surveys & datasets driving this](#7--instruments-surveys--datasets-driving-this)
8. [Why a radio map is *not* a natural image](#8--why-a-radio-map-is-not-a-natural-image)
9. [Open problems / what to watch](#9--open-problems--what-to-watch)
10. [Sources](#10--sources)

## 1 · Why this pass: the interferometric image as its own primitive

Six properties make the radio-interferometric image worth treating as a
first-class dense-vision surface rather than "a grayscale astronomy photo":

1. **You never measure the image — you measure its Fourier transform, sparsely.**
   An interferometer cross-correlates pairs of antennas; each pair (a
   *baseline*) yields one complex *visibility*, a single sample of the sky's
   2-D Fourier transform at the spatial frequency set by the baseline geometry.
   The array measures a scattered set of points in the *uv-plane*, Earth
   rotation smears them into arcs, and everything the sky contains at
   unmeasured spatial frequencies is simply missing. Reconstruction is an
   ill-posed inverse problem before it is a vision problem.

2. **The point-spread function is pathological and field-wide.** The inverse
   transform of that sparse sampling — the *dirty beam* — is not a tidy
   Gaussian; it is a sharp core wrapped in ringing *sidelobes* that spray
   structure from every bright source across the whole field. The measured
   *dirty image* is the true sky convolved with this beam. "Deconvolution" here
   is not cosmetic sharpening; nothing downstream is trustworthy until it is
   done.

3. **The dynamic range is brutal and the labels are faint.** A single bright
   source can be millions of times stronger than the faint galaxies you
   actually want to catalogue, and its sidelobes bury them. Detection is a
   needle-in-haystack problem where the haystack is an artifact of the
   instrument, not the sky.

4. **It is a dense-detection problem at a scale no other imaging modality
   faces.** SKA-precursor continuum surveys already catalogue tens of millions
   of sources; the SKA era is forecast in the *tens of billions*. Source
   finding, deblending, and component-to-source association at that volume is a
   dense-vision problem measured in exabytes.

5. **Morphology is the class label, and it is rotation- and scale-free.** A
   radio galaxy has no canonical "up"; whether it is edge-brightened (FR-II) or
   core-brightened (FR-I), one-sided, bent, or a compact point is written in a
   diffuse, low-signal morphology that can appear at any orientation and size.
   This is exactly the regime where rotation-equivariant and self-supervised
   vision earns its keep.

6. **The archive is enormous and mostly unlabeled.** Decades of visibilities
   and images exist; expert morphological labels number in the thousands.
   That mismatch — exabytes of pixels, a few thousand labels — is the defining
   constraint, and it is precisely where the field's 2024–2026 pivot to
   foundation models and label-efficient learning comes from.

![A radio interferometer shown as a dense detection-and-classification scene: an antenna array samples the sky's Fourier plane, the incomplete sampling produces a sidelobe-riddled dirty image, and deconvolution recovers a sky of sources to reconstruct, detect, and classify.](assets/radio-interferometer-as-dense-scene.svg)

## 2 · The primitive — visibilities, the dirty beam, and CLEAN

**The measurement.** Under the van Cittert–Zernike theorem, the correlation of
the voltages from two antennas measures one complex visibility
`V(u, v)` — a sample of the sky brightness's 2-D Fourier transform at the
baseline's projected coordinates `(u, v)`. An `N`-antenna array gives
`N(N−1)/2` baselines per instant, and Earth rotation sweeps each baseline
through the uv-plane over a night, "synthesizing" a much larger effective
aperture (*aperture synthesis*, the idea that won Ryle the 1974 Nobel). But the
sampling is always **incomplete**: there is a central hole (no zero-spacing
information → the mean sky level is unmeasured), gaps between baselines, and a
maximum baseline that sets the resolution.

**The dirty image and dirty beam.** Inverse-transforming the *sampled*
visibilities (weighted, gridded onto a regular grid, FFT'd) yields the **dirty
image** `I_D = I ⋆ B_dirty`, the true sky `I` convolved with the **dirty beam**
`B_dirty` — itself the inverse transform of the uv sampling function. Because
the sampling is sparse, `B_dirty` has large, structured sidelobes. Every source
in the field contributes a copy of this sidelobe pattern, so the dirty image is
a superposition of overlapping PSFs. Recovering `I` is *deconvolution under
missing data* — mathematically underdetermined, because infinitely many skies
agree with the measured visibilities and differ only in the unsampled Fourier
modes.

**CLEAN — the 50-year-old workhorse.** Högbom's **CLEAN** (1974) models the sky
as a sum of point sources: iteratively find the brightest peak in the residual
dirty image, subtract a scaled dirty beam at that location, record a "clean
component," repeat, then restore the components with an idealized clean beam.
Cotton–Schwab and Clark variants moved the subtraction into the visibility
domain; **multiscale CLEAN** added extended-emission components; **WSClean** is
the modern wide-field, `w`-aware implementation. CLEAN is fast, robust, and
astronomer-trusted — and it is also greedy, non-regularized, resolution-limited
to the clean beam, and it bakes in a point-source prior that struggles on
diffuse structure. Bayesian alternatives (**RESOLVE** and its information-field-
theory successors) replace the greedy loop with a principled prior and give
uncertainty maps, at much higher compute cost. The entire modern ML thread is,
in one way or another, an attempt to *learn* a better prior than "a pile of
delta functions" while keeping the measurement physics honest.

**Where the dense-vision jobs live.** The right panel of the figure above
splits the modality into three jobs, and they map onto the pipeline stages
below: **(A) imaging/deconvolution** — turn visibilities into an artifact-free
sky image; **(B) source finding** — detect, deblend, and characterize every
source in that image; and **(C) morphological classification** — assign each
extended source a morphology and match it to its host galaxy. Two upstream data-
quality steps — **RFI flagging** and calibration — are themselves dense
per-pixel classification problems on the time–frequency data, and are where a
lot of the recent deep-learning volume actually sits.

![The deep-learning landscape over a radio observation: RFI flagging and calibration as segmentation of the dynamic spectrum, imaging/deconvolution as an inverse problem evolving from CLEAN to diffusion priors, source finding evolving from PyBDSF to YOLO-CIANNA, and morphological classification evolving from CNNs to self-supervised foundation models, with a time-arc of model families beneath.](assets/radio-pipeline-landscape.svg)

## 3 · The imaging inverse problem — learned deconvolution & diffusion priors

This is the stage where radio imaging is being rewritten, because it is a clean
instance of a **linear inverse problem with a learnable prior** — the same shape
as MRI ([Aug-07](../2026-Aug-07/2026-Aug-07_CV_updates.md)) and photoacoustic
([Aug-13](../2026-Aug-13/2026-Aug-13_CV_updates.md)) reconstruction, but with a
sparser, uglier forward operator.

**The arc (greedy → regularized → learned → generative).**

- **Classical.** CLEAN (greedy matching pursuit) and RESOLVE/IFT (Bayesian,
  prior-driven) anchor the two ends of the pre-deep-learning spectrum — speed
  vs. principled uncertainty.
- **Learned CLEAN — the R2D2 series.** The flagship "keep the physics, learn
  the update" line replaces CLEAN's greedy minor-cycle with a *series* of
  trained DNNs, each predicting a residual image correction from the current
  estimate and the back-projected data — a learned matching pursuit that reaches
  CLEAN-beating fidelity and dynamic range in a handful of iterations
  (**R2D2**, [arXiv:2403.05452](https://arxiv.org/abs/2403.05452);
  interpreted explicitly as a learned CLEAN while imaging Cygnus A,
  [arXiv:2309.03291](https://arxiv.org/abs/2309.03291)). It now has an
  uncertainty-quantified variant ([arXiv:2403.18052](https://arxiv.org/abs/2403.18052)),
  a wide-field spherical extension (**S-R2D2**,
  [arXiv:2503.01462](https://arxiv.org/abs/2503.01462)), and a robustness/
  architecture study ([arXiv:2503.02554](https://arxiv.org/abs/2503.02554)).
- **Plug-and-play & unrolled optimization.** Treat imaging as regularized
  optimization and drop a *learned denoiser* in as the proximal operator
  (**AIRI**, [arXiv:2202.12959](https://arxiv.org/abs/2202.12959);
  robustness/variations, [arXiv:2312.07137](https://arxiv.org/abs/2312.07137);
  a hyperspectral **HyperAIRI**,
  [arXiv:2510.15198](https://arxiv.org/abs/2510.15198)), or unroll a fixed number
  of iterations into a trainable network, or use an untrained CNN as an implicit
  prior (deep image prior). These keep the measurement operator explicit and
  learn only the regularizer — attractive when there is no ground truth.
- **Score-based / Bayesian posterior sampling.** Rather than one image, learn a
  score/diffusion prior and *sample the posterior* of skies consistent with the
  data, giving pixel-wise uncertainty: Bayesian imaging with score-based priors
  ([arXiv:2311.18012](https://arxiv.org/abs/2311.18012)), **IRIS** with
  expressive score priors ([arXiv:2501.02473](https://arxiv.org/abs/2501.02473)),
  scalable data-driven-prior UQ
  ([arXiv:2312.00125](https://arxiv.org/abs/2312.00125)), and fast-UQ generative
  imaging ([arXiv:2507.21270](https://arxiv.org/abs/2507.21270)). The principled
  non-DL comparison remains **RESOLVE**/information-field-theory Bayesian imaging
  ([aim-resolve, arXiv:2512.04840](https://arxiv.org/abs/2512.04840)).
- **Conditional diffusion / score-based priors.** The most active 2023–2026
  thread. A denoising diffusion model learns a prior over "what real radio skies
  look like" and is then conditioned on the observed data — visibilities, the
  dirty image, or both — to sample plausible reconstructions. Because it is
  generative, it yields an *ensemble* of images consistent with the data, i.e.
  posterior samples and uncertainty, rather than a single deterministic map.
  Representative work: a **conditional DDPM for radio-interferometric
  reconstruction** ([arXiv:2305.09121](https://arxiv.org/abs/2305.09121)),
  **VIC-DDPM** — visibility-and-image-conditioned diffusion
  ([arXiv:2402.10204](https://arxiv.org/abs/2402.10204)) — and, at the
  event-horizon-imaging extreme, **diffusion priors conditioned on closure
  invariants** that reconstruct EHT sources such as 3C 279 and Centaurus A
  independently of the calibration chain
  ([arXiv:2602.21507](https://arxiv.org/abs/2602.21507), 2026, identifier
  best-effort). A **Denoising Diffusion Restoration** approach that is agnostic
  to the measured data and folds the measurement physics into the sampler
  appeared in early 2026 ([arXiv:2601.15844](https://arxiv.org/abs/2601.15844),
  best-effort).
- **Deep deconvolution for transients & semi-supervised reconstruction.**
  Learned deconvolution specialized to interferometric *transient*
  reconstruction ([arXiv:2306.13909](https://arxiv.org/abs/2306.13909)) and
  semi-supervised visibility reconstruction (**VisRec**,
  [arXiv:2403.00897](https://arxiv.org/abs/2403.00897)) target the
  data-scarcity and out-of-distribution problems head-on.
- **Super-resolution beyond the beam.** **POLISH**
  ([arXiv:2111.03249](https://arxiv.org/abs/2111.03249)) framed radio imaging as
  a learned super-resolution problem, recovering structure finer than the
  nominal clean beam — with the standing caveat that any super-resolution
  method must be audited for hallucinated structure at unmeasured spatial
  frequencies.

**The tension that defines the stage.** A generative prior is exactly what lets
you fill the uv-gaps plausibly — and exactly what can invent sources that were
never measured. So the credible line of work keeps the forward operator in the
loop (data-consistency steps, closure quantities, or explicit likelihoods) and
reports uncertainty, rather than treating imaging as image-to-image translation.
A 2026 effort toward a *multiscale predictive model* of image recovery
([arXiv:2607.12396](https://arxiv.org/abs/2607.12396), best-effort) is
symptomatic of the field trying to characterize *when* learned reconstruction
can be trusted.

## 4 · Dense detection — source finding at survey scale

Once there is an image, **source finding** is the dense-detection job: locate
every source, separate blended neighbours, measure flux and shape, and — for
extended radio galaxies — associate the multiple disconnected *components*
(e.g. two lobes and a core) into one physical *source*, then match it to an
optical/IR host. At SKA-precursor scale this runs over tens of millions of
sources; at SKA scale it is one of the defining big-data problems in science.

**Classical detectors (still the baselines to beat).** **PyBDSF** (the PyBDSM
successor) fits Gaussians to islands of emission and is the de-facto survey
standard; **Aegean/BANE** uses a flood-fill on a background-and-noise map;
**ProFound** brings a segmentation-based, dilated-isophote approach from optical
photometry; **SExtractor** remains a cross-domain reference. These are fast and
well-understood but tuned for compact/Gaussian sources and struggle with
extended, multi-component radio galaxies and with sidelobe artifacts; the
**Hydra II** study ([PASA 2023](https://www.cambridge.org/core/journals/publications-of-the-astronomical-society-of-australia/article/hydra-ii-characterisation-of-aegean-caesar-profound-pybdsf-and-selavy-source-finders/DC245D86E75644800D682F7E0FC3D7D9))
characterizes and cross-compares five of them and is the current reference for
"what the baselines actually do."

**Deep detectors and segmentors.**

- **ClaRAN** (Classifying Radio sources Automatically with Neural networks;
  [arXiv:1805.12008](https://arxiv.org/abs/1805.12008)) adapted **Faster/Mask
  R-CNN** to jointly *detect and morphologically classify* radio sources — the
  detection-as-vision reframing that seeded much of what followed.
- **DeepSource** ([arXiv:1807.02701](https://arxiv.org/abs/1807.02701)) treats
  faint-source detection as CNN-based peak enhancement plus thresholding, beating
  PyBDSF down to SNR ≈ 4.
- **Instance segmentation at survey scale.** **HeTu-v2 / caesar-mrcnn**
  ([arXiv:2306.01426](https://arxiv.org/abs/2306.01426)) combines Mask R-CNN with
  a transformer to segment *and* classify sources, and produced FIRST-HeTu
  (~835k sources); related Mask R-CNN detectors segment compact + extended
  emission on continuum maps ([arXiv:2212.02538](https://arxiv.org/abs/2212.02538)),
  and there is now a dedicated **benchmark** of detector/segmenter families on
  radio data ([arXiv:2303.04506](https://arxiv.org/abs/2303.04506)).
- **Transformers arrive.** A **RF-DETR** set-prediction detector was applied to
  LOFAR deep-field continuum images for joint detection + morphology, pitting
  DETR against Mask R-CNN and YOLO
  ([arXiv:2605.30500](https://arxiv.org/abs/2605.30500), 2026, best-effort).
- **U-Net segmentation** proved decisive in the **SKA Science Data Challenge 2**:
  the FORSKA-Sweden team used a ResNet-encoder U-Net for 3-D **H i** source
  finding ([A&A 2023](https://www.aanda.org/articles/aa/full_html/2023/03/aa45139-22/aa45139-22.html)),
  and the challenge's overall lesson — that a *combination* of complementary ML
  methods beat any single technique — is documented in the
  [SDC2 analysis and results (MNRAS 2023)](https://academic.oup.com/mnras/article/523/2/1967/7157129).
- **YOLO-CIANNA** ([arXiv:2509.12082](https://arxiv.org/abs/2509.12082), 2025)
  generalized a YOLO-style regression detector to 3-D radio cubes and reports
  *winning* the SDC2 — a strong signal that real-time, single-shot detectors
  have arrived for radio source finding, not just for natural images.
- **Catalogue-construction pipelines** such as **RG-CAT** for the EMU pilot
  survey wrap detection, component association, and host cross-identification
  into an end-to-end deep pipeline (identifier best-effort;
  [arXiv:2403.14235](https://arxiv.org/abs/2403.14235)) — the piece that turns
  "boxes on an image" into an astrophysical catalogue.

**Component-to-source association is its own problem.** A single radio galaxy
often appears as several disconnected islands (two lobes, a core), and grouping
them into one physical source — then matching to an optical/IR host — is
genuinely ambiguous even to experts. Recent work frames it as **multi-modal deep
learning**, fusing image and tabular features to associate LOFAR components
([arXiv:2405.18584](https://arxiv.org/abs/2405.18584)), and a dedicated
benchmark, **RadioGalaxyNET**, targets joint detection of extended radio galaxies
*and* their infrared hosts
([arXiv:2312.00306](https://arxiv.org/abs/2312.00306)).

**Why detection here is unusually hard.** The objects are diffuse and
low-surface-brightness, they blend with sidelobe residuals, the grouping of
components into sources is ambiguous, and the interesting objects are rare
against tens of millions of ordinary ones. It is arguably as much a
graph/relational problem as a detection one, and remains a live research target
— see the 2026 **SKAO source-finding review**
([arXiv:2607.03736](https://arxiv.org/abs/2607.03736), best-effort) for where
classical and ML methods still fall short at SKA scale.

## 5 · Dense classification — radio-galaxy morphology & foundation models

Classifying the morphology of extended radio sources — the historic
**Fanaroff–Riley** FR-I (core-bright, edge-darkened) vs. FR-II (edge-bright,
hot-spotted) dichotomy, plus bent-tail, hybrid, one-sided, and compact classes
— is the modality's canonical classification task, and it is where the
label-scarcity problem bites hardest.

**Supervised CNNs and the label wall.** Early work fine-tuned ImageNet-style
CNNs on curated sets like **MiraBest** — a machine-learning-ready FR benchmark
of ~1,256 labelled images ([arXiv:2305.11108](https://arxiv.org/abs/2305.11108))
— and crowd-sourced labels from **Radio Galaxy Zoo (RGZ)**. These work, but
expert/consensus labels number in the thousands, classes are imbalanced and
ambiguous, and models are brittle to the orientation and instrument of the
training set. Because those labels genuinely disagree, **Bayesian deep learning**
for calibrated FR uncertainty is an active thread
([arXiv:2405.18351](https://arxiv.org/abs/2405.18351)).

**Symmetry-aware architectures.** Because a radio galaxy has no preferred
orientation, **rotation-equivariant / group-equivariant CNNs** (Scaife & Porter,
[arXiv:2102.08252](https://arxiv.org/abs/2102.08252)) bake the symmetry into the
network, improving data efficiency and consistency — a clean example of encoding
the modality's physics as an inductive bias rather than learning it from
augmentation.

**The 2024–2026 pivot: self-supervised & foundation models.** With exabytes of
unlabeled images and only thousands of labels, the field has moved decisively to
pre-train on the unlabeled archive and fine-tune with few labels:

- **Radio Galaxy Zoo: towards the first multi-purpose foundation model** —
  BYOL self-supervised pre-training on RGZ DR1, cutting misclassification rate
  roughly in half versus supervised-only when fine-tuned
  ([arXiv:2305.16127](https://arxiv.org/abs/2305.16127);
  [RASTI 2024](https://academic.oup.com/rasti/article/3/1/19/7491070)).
- **Contrastive pre-training** of radio data improves detection, classification,
  *and* surfaces peculiar/anomalous objects, with a few-percent gain over
  supervised baselines ([arXiv:2404.18462](https://arxiv.org/abs/2404.18462)).
- **Fanaroff–Riley classification with a pre-trained foundation model** —
  a late-2025 application fine-tuning the RGZ foundation model to type ~14,000
  radio galaxies (~5,900 FR-I / ~8,100 FR-II)
  ([arXiv:2509.11988](https://arxiv.org/abs/2509.11988);
  [MNRAS 2025](https://dx.doi.org/10.1093/mnras/staf1942)).
- **Self-supervised classification into many morphological classes** —
  extending SSL to 12-way radio-source morphology
  ([arXiv:2503.19111](https://arxiv.org/abs/2503.19111)) and a **benchmark of
  SSL methods for radio-source classification**
  ([Springer 2025](https://link.springer.com/chapter/10.1007/978-3-031-88217-3_32)),
  with a 2026 **variational-views** take on SSL objectives for radio images
  ([RASTI 2026](https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzag037/8676723)).
- **General radio foundation models** — **STRADAViT**, a ViT trained by
  self-supervised transfer toward a foundational radio model
  ([arXiv:2603.29660](https://arxiv.org/abs/2603.29660), 2026, best-effort), and
  **RGC**, a radio-AGN classifier over 12 morphological classes
  ([arXiv:2510.22190](https://arxiv.org/abs/2510.22190), 2025).
- **Vision-language models.** **radio-llava** fine-tunes a LLaVA-style VLM for
  radio-source description and classification
  ([PASA 2025](https://www.cambridge.org/core/journals/publications-of-the-astronomical-society-of-australia/article/radiollava-advancing-visionlanguage-models-for-radio-astronomical-source-analysis/5E14BA0AE0C6196B63E8041CEB934B35))
  — the first steps toward zero/few-shot, text-promptable radio morphology.
- **Uncertainty under ambiguous ground truth.** Calibrated uncertainty is a
  first-class requirement; recent work applies **Monte-Carlo conformal
  prediction** to radio-galaxy classification under ambiguous labels
  ([arXiv:2603.20000](https://arxiv.org/abs/2603.20000), 2026, identifier
  best-effort). See also the broad
  [review of unsupervised learning in astronomy](https://arxiv.org/abs/2406.17316)
  (2024) for context.

The through-line: morphology classification has become a *representation-
learning* problem first and a classifier problem second, and the interesting
metrics are now label-efficiency and calibrated uncertainty, not just top-1
accuracy on a saturated benchmark.

## 6 · RFI flagging — segmentation in the dynamic spectrum

Before imaging, the raw data must be cleared of **radio-frequency interference**
— satellites, aircraft, radar, phones, the instrument itself — which appears in
the time–frequency **dynamic spectrum** as bright streaks (narrowband, persistent
in frequency) and blobs (broadband, transient in time). Flagging is a
**dense per-pixel (binary or instance) segmentation** problem, and it is where a
large share of the deep-learning-for-radio volume actually lives, because labels
are cheap-ish (simulatable) and the payoff is immediate.

- **U-Net as the template.** RFI detection was reframed as image segmentation
  with U-Net (Akeret et al., 2017); **RFI-Net** added residual blocks/batch-norm
  for FAST dynamic spectra ([arXiv:2001.06669](https://arxiv.org/abs/2001.06669)),
  and a MeerKAT **R-Net**-style deep ResNet beat the default flagger and plain
  U-Nets ([arXiv:2005.08992](https://arxiv.org/abs/2005.08992)); a robust CNN
  classifier for interferometric RFI followed
  ([arXiv:2203.00298](https://arxiv.org/abs/2203.00298)).
- **Joint flag-and-restore.** **RFI-DRUnet**
  ([arXiv:2402.13867](https://arxiv.org/abs/2402.13867);
  [ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S2213133724000374))
  not only detects RFI in pulsar dynamic spectra but *restores* the corrupted
  samples — segmentation and inpainting in one network.
- **Learning without clean labels.** "Learning to detect RFI without seeing it"
  ([MNRAS 2022](https://academic.oup.com/mnras/article/516/4/5367/6692884))
  tackles the no-ground-truth problem, and a 2024 **comparison framework**
  benchmarks deep RFI detectors and a transfer-learning/fine-tuning recipe,
  finding low-capacity models resilient to noisy flags and high-capacity models
  best with clean flags
  ([MNRAS 2024](https://academic.oup.com/mnras/article/530/1/613/7637224)).
- **Foundation-segmentation transfer.** A 2024 study probes the **Segment
  Anything Model** for RFI/event detection in radio data
  ([arXiv:2410.22497](https://arxiv.org/abs/2410.22497)) — the same "does SAM
  transfer to my weird modality?" question this log has tracked across OCT,
  endoscopy, and SAR.
- **The reality check.** A December 2025 review of **real-time** RFI mitigation
  and its practical limits ([arXiv:2512.01954](https://arxiv.org/abs/2512.01954))
  is a useful corrective: throughput, latency, and the cost of a false flag
  (thrown-away science data) constrain what can actually be deployed at the
  correlator.

## 7 · Instruments, surveys & datasets driving this

The methods above are pulled forward by an instrument ramp toward the SKA, and
by a handful of labeled datasets that carry the whole ML sub-field.

**Instruments & surveys.**

- **SKA** ([skao.int](https://www.skao.int/)) — the exascale driver; its data
  rates are the reason "source finding at survey scale" is an ML problem at all.
- **ASKAP / EMU** — the Evolutionary Map of the Universe, a wide continuum
  survey forecast to catalogue tens of millions of sources
  ([Norris et al., PASA 2021, arXiv:2103.10163](https://arxiv.org/abs/2103.10163)).
- **MeerKAT** — deep continuum via **MIGHTEE DR1**
  ([arXiv:2411.04958](https://arxiv.org/abs/2411.04958)) and the ~495k-source
  **MALS DR1** ([arXiv:2308.12347](https://arxiv.org/abs/2308.12347)).
- **LOFAR** — the **LoTSS** two-metre sky survey
  ([Shimwell et al. 2022, arXiv:2202.11733](https://arxiv.org/abs/2202.11733))
  and its deep-fields host-ID value-added catalogues
  ([arXiv:2011.08201](https://arxiv.org/abs/2011.08201)), plus **MWA** and the
  VLA's **VLASS** all-sky 2–4 GHz survey (~2 PB of data products;
  [design, arXiv:1907.01981](https://arxiv.org/abs/1907.01981)).

**Datasets & challenges (the labeled backbone).**

- **Radio Galaxy Zoo (RGZ)** — citizen-science morphology and host-association
  labels; the substrate for the RGZ foundation-model line.
- **MiraBest** — a machine-learning-ready FR-I/FR-II benchmark distilled from
  earlier expert samples.
- **RadioGalaxyNET** ([arXiv:2312.00306](https://arxiv.org/abs/2312.00306)) — a
  dataset + algorithms for joint detection of extended radio galaxies and their
  IR hosts, purpose-built for CV research.
- **Radio Galaxy Zoo: EMU** — the citizen-science-plus-AI framework (active
  learning over ~4M extended EMU sources) that will label the ASKAP era
  ([arXiv:2506.16138](https://arxiv.org/abs/2506.16138),
  [arXiv:2509.19787](https://arxiv.org/abs/2509.19787)).
- **SKA Science Data Challenges** — **SDC1** (continuum source finding &
  characterization, [Bonaldi et al. 2021, arXiv:2009.13346](https://arxiv.org/abs/2009.13346))
  and **SDC2** (H i source finding in a 3-D cube,
  [analysis & results, MNRAS 2023](https://academic.oup.com/mnras/article/523/2/1967/7157129))
  are the closest thing the field has to standardized, physics-grounded
  detection benchmarks — with realistic RFI, noise, and confusion baked in.

**Transients as real-time dense detection.** Beyond static maps, **fast radio
burst** detection runs the same detection logic on the streaming time–frequency
plane under hard latency: a deployed real-time end-to-end DL FRB detector
([A&A 2025](https://www.aanda.org/articles/aa/full_html/2025/10/aa55217-25/aa55217-25.html))
and DL morphological characterization of repeating vs. non-repeating bursts
([arXiv:2509.06208](https://arxiv.org/abs/2509.06208)) show the modality's
detection problem extends into the time domain.

**Tooling.** Classical source finders (PyBDSF, Aegean/BANE, ProFound) and
imagers (WSClean, CASA `tclean`) remain the deployed baselines and the
reference implementations that any learned method is measured against.

## 8 · Why a radio map is *not* a natural image

Pulling the thread together — the recurring theme of this log is that each
modality's *physics* dictates the vision, and radio interferometry is one of the
sharpest examples:

- **The data lives in Fourier space, sampled sparsely.** There is no complete
  image to start from; the missing spatial frequencies mean many valid skies fit
  the data. A generative prior helps and can hallucinate in equal measure.
- **The PSF is field-wide, structured, and source-dependent.** Sidelobes couple
  every bright source into the noise budget of every faint one — non-local,
  non-stationary corruption unlike the local blur/noise of a photo.
- **Correlated, non-Gaussian noise and enormous dynamic range.** Standard
  assumptions behind off-the-shelf detectors and losses do not hold.
- **No canonical orientation or scale.** Rotation- and scale-equivariance are
  correct inductive biases, not optional augmentations.
- **No ground truth.** There is a *model that fits the visibilities*, not a
  measured true sky — so uncertainty quantification and data-consistency are
  intrinsic, not add-ons.
- **The label budget is inverted.** Exabytes of unlabeled pixels, thousands of
  labels — the defining constraint that makes self-supervision and foundation
  models the natural, not fashionable, choice.

## 9 · Open problems / what to watch

- **End-to-end, calibration-aware pipelines.** Calibration → flagging → imaging
  → detection → classification are still largely separate stages with separate
  error models. Differentiable, uncertainty-propagating pipelines (and
  calibration-agnostic tricks like closure-quantity conditioning) are an obvious
  frontier.
- **Trustworthy generative reconstruction.** Diffusion imaging is powerful and
  dangerous for the same reason. Expect a hard focus on **hallucination audits**,
  data-consistency guarantees, and posterior calibration before generative maps
  are trusted for science.
- **Instrument-agnostic foundation models.** A model pre-trained on LOFAR should
  transfer to MeerKAT and ASKAP; cross-instrument domain shift (resolution,
  frequency, uv-coverage, noise) is the analogue of the cross-scanner problem
  seen in medical imaging, and largely unsolved at scale — learned imaging that
  generalizes across *varying visibility coverage*
  ([arXiv:2405.08958](https://arxiv.org/abs/2405.08958)) is an early attack on
  the imaging side of it. The 2026 review **The Role of Artificial Intelligence
  in the SKA Era** ([arXiv:2606.28493](https://arxiv.org/abs/2606.28493),
  best-effort) is the best forward-looking map of where vision-foundation models
  (CLIP/SAM/DINO adaptation) help versus fail on radio data.
- **Detection + association as one learned problem.** Component-to-source
  grouping and host cross-identification are where classical catalogues bleed
  errors; framing them jointly with detection (graph/relational vision) is
  wide open.
- **Real-time at exascale.** SKA data cannot be stored raw; a great deal of
  flagging, and perhaps detection, must happen *in-stream* under hard latency
  budgets — pushing efficient architectures and on-instrument inference.
- **Calibrated uncertainty under ambiguous labels.** Expert disagreement is
  irreducible for many morphologies; conformal and Bayesian methods that report
  honest uncertainty (rather than overconfident top-1) are becoming a
  requirement, not a nicety.

## 10 · Sources

**Reviews, context & the primitive** *(the cleanest modern expositions of "why a
radio image is not a natural image" live in the introductions of the R2D2, AIRI
and SKAO source-finding papers below; the canonical textbook is Thompson, Moran
& Swenson, *Interferometry and Synthesis in Radio Astronomy*).*

- Högbom (1974), *Aperture synthesis with a non-regular distribution of
  interferometer baselines* (CLEAN) — *A&AS* 15:417 (lineage anchor).
- Image reconstruction algorithms in radio interferometry: from handcrafted to learned regularization (AIRI context/survey) — *MNRAS* (2023), arXiv:2202.12959 — https://arxiv.org/abs/2202.12959
- Source Finding and Characterisation for SKAO Science (review) — arXiv:2607.03736 (2026, best-effort) — https://arxiv.org/abs/2607.03736
- The Role of Artificial Intelligence in the SKA Era (review) — arXiv:2606.28493 (2026, best-effort) — https://arxiv.org/abs/2606.28493
- A review of unsupervised learning in astronomy (2024) — arXiv:2406.17316 — https://arxiv.org/abs/2406.17316
- SKA Observatory — https://www.skao.int/

**Imaging / deconvolution — learned CLEAN, plug-and-play & generative**

- R2D2: deep neural network series paradigm for fast precision imaging (learned CLEAN) — *ApJS* (2024), arXiv:2403.05452 — https://arxiv.org/abs/2403.05452
- CLEANing Cygnus A deep and fast with R2D2 — *ApJL* (2024), arXiv:2309.03291 — https://arxiv.org/abs/2309.03291
- R2D2 with model uncertainty quantification — arXiv:2403.18052 — https://arxiv.org/abs/2403.18052 · S-R2D2 (spherical, wide-field) — arXiv:2503.01462 — https://arxiv.org/abs/2503.01462 · robust R2D2 training/architecture — arXiv:2503.02554 — https://arxiv.org/abs/2503.02554
- AIRI plug-and-play (learned denoiser as proximal operator): variations & robustness — *MNRAS* (2025), arXiv:2312.07137 — https://arxiv.org/abs/2312.07137 · HyperAIRI (hyperspectral) — arXiv:2510.15198 — https://arxiv.org/abs/2510.15198
- Bayesian imaging for radio interferometry with score-based priors — arXiv:2311.18012 — https://arxiv.org/abs/2311.18012
- IRIS: Bayesian image reconstruction with expressive score-based priors — arXiv:2501.02473 — https://arxiv.org/abs/2501.02473
- Scalable Bayesian UQ with data-driven priors for RI imaging — RASTI (2024), arXiv:2312.00125 — https://arxiv.org/abs/2312.00125 · fast-UQ generative imaging — arXiv:2507.21270 — https://arxiv.org/abs/2507.21270
- Learned RI imaging for varying visibility coverage — arXiv:2405.08958 — https://arxiv.org/abs/2405.08958
- RESOLVE / information-field-theory Bayesian imaging (non-DL comparison): aim-resolve — arXiv:2512.04840 — https://arxiv.org/abs/2512.04840
- A Conditional DDPM for Radio Interferometric Image Reconstruction (visibility-and-image conditioned) — arXiv:2305.09121 — https://arxiv.org/abs/2305.09121
- Radio-astronomical Image Reconstruction with a Conditional Denoising Diffusion Model — arXiv:2402.10204 — https://arxiv.org/abs/2402.10204 · *A&A* (2024) — https://www.aanda.org/articles/aa/full_html/2024/03/aa47948-23/aa47948-23.html
- Deep-learning deconvolution for interferometric radio transient reconstruction — *A&A* (2023), arXiv:2306.13909 — https://arxiv.org/abs/2306.13909
- VisRec: semi-supervised radio interferometric data reconstruction — arXiv:2403.00897 — https://arxiv.org/abs/2403.00897
- Conditional image diffusion with interferometric closure invariants (EHT: Cen A, 3C 279) — arXiv:2602.21507 (2026, best-effort) — https://arxiv.org/abs/2602.21507
- Radio-Interferometric Image Reconstruction with Denoising Diffusion Restoration Models — arXiv:2601.15844 (2026, best-effort) — https://arxiv.org/abs/2601.15844
- Toward a multiscale predictive model of image recovery from radio interferometers — arXiv:2607.12396 (2026, best-effort) — https://arxiv.org/abs/2607.12396
- POLISH: deep super-resolution radio-interferometric imaging (DSA-2000) — *MNRAS* (2022), arXiv:2111.03249 — https://arxiv.org/abs/2111.03249 · wide-field, high-dynamic-range POLISH — arXiv:2603.09162 (2026, best-effort) — https://arxiv.org/abs/2603.09162

**Source finding / dense detection**

- Hydra II: characterisation of Aegean, Caesar, ProFound, PyBDSF & Selavy source finders — *PASA* (2023) — https://www.cambridge.org/core/journals/publications-of-the-astronomical-society-of-australia/article/hydra-ii-characterisation-of-aegean-caesar-profound-pybdsf-and-selavy-source-finders/DC245D86E75644800D682F7E0FC3D7D9
- Radio source extraction with ProFound — *MNRAS* (2019), arXiv:1902.01440 — https://arxiv.org/abs/1902.01440
- ClaRAN — Radio Galaxy Zoo: a deep-learning classifier for radio morphologies (Faster/Mask R-CNN) — *MNRAS* (2019), arXiv:1805.12008 — https://arxiv.org/abs/1805.12008
- DeepSource: point-source detection with deep learning — *MNRAS* (2019), arXiv:1807.02701 — https://arxiv.org/abs/1807.02701
- HeTu-v2 / caesar-mrcnn — radio-source segmentation & classification with deep learning — *Astron. & Comput.* (2023), arXiv:2306.01426 — https://arxiv.org/abs/2306.01426
- Astronomical source detection in radio continuum maps with deep neural networks — arXiv:2212.02538 — https://arxiv.org/abs/2212.02538
- Radio astronomical images object detection & segmentation: a deep-learning benchmark — arXiv:2303.04506 — https://arxiv.org/abs/2303.04506
- Transformer-based (RF-DETR) source detection & morphological classification in LOFAR deep fields — arXiv:2605.30500 (2026, best-effort) — https://arxiv.org/abs/2605.30500
- Identification of multi-component LOFAR sources with multi-modal deep learning (association) — *MNRAS* (2024), arXiv:2405.18584 — https://arxiv.org/abs/2405.18584
- CNNs for H i source finding — FORSKA-Sweden, SKA SDC2 — *A&A* (2023) — https://www.aanda.org/articles/aa/full_html/2023/03/aa45139-22/aa45139-22.html
- YOLO-CIANNA (II): winning the SKA SDC2 with a generalized 3D-YOLO — arXiv:2509.12082 (2025) — https://arxiv.org/abs/2509.12082
- SKA Science Data Challenge 2: analysis and results — *MNRAS* 523:1967 (2023) — https://academic.oup.com/mnras/article/523/2/1967/7157129
- SKA Science Data Challenge 1: analysis and results — arXiv:2009.13346 — https://arxiv.org/abs/2009.13346
- RG-CAT: detection pipeline & catalogue of radio galaxies in the EMU pilot survey — arXiv:2403.14235 — https://arxiv.org/abs/2403.14235
- RadioGalaxyNET: dataset & CV algorithms for extended radio galaxies + IR hosts — *PASA* (2023/24), arXiv:2312.00306 — https://arxiv.org/abs/2312.00306
- PyBDSF (source finder) — https://pybdsf.readthedocs.io/ · Aegean/BANE — https://github.com/PaulHancock/Aegean

**Morphological classification & foundation models**

- MiraBest: a dataset of morphologically classified radio galaxies for ML — arXiv:2305.11108 — https://arxiv.org/abs/2305.11108
- Rotation-/group-equivariant CNNs for FR classification (Scaife & Porter) — *MNRAS* (2021), arXiv:2102.08252 — https://arxiv.org/abs/2102.08252
- Evaluating Bayesian deep learning for radio-galaxy classification (UQ) — arXiv:2405.18351 — https://arxiv.org/abs/2405.18351
- Radio Galaxy Zoo: towards the first multi-purpose foundation model with self-supervised learning — arXiv:2305.16127 — https://arxiv.org/abs/2305.16127 · RASTI (2024) — https://academic.oup.com/rasti/article/3/1/19/7491070
- Self-supervised contrastive learning of radio data (detection, classification, anomaly discovery) — arXiv:2404.18462 — https://arxiv.org/abs/2404.18462
- Radio Galaxy Zoo: FR classification via self-supervised pre-training — arXiv:2509.11988 (2025) — https://arxiv.org/abs/2509.11988 · *MNRAS* — https://dx.doi.org/10.1093/mnras/staf1942
- Classification of radio sources through self-supervised learning — arXiv:2503.19111 — https://arxiv.org/abs/2503.19111
- Self-supervised learning for radio-astronomy source classification: a benchmark — Springer (2025) — https://link.springer.com/chapter/10.1007/978-3-031-88217-3_32
- Variational views for self-supervised learning in radio astronomy — RASTI (2026) — https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzag037/8676723
- STRADAViT: toward a foundational radio model via self-supervised transfer — arXiv:2603.29660 (2026, best-effort) — https://arxiv.org/abs/2603.29660
- RGC: a radio-AGN classifier (12 morphological classes) — arXiv:2510.22190 (2025) — https://arxiv.org/abs/2510.22190
- radio-llava: vision-language models for radio-source analysis — *PASA* (2025) — https://www.cambridge.org/core/journals/publications-of-the-astronomical-society-of-australia/article/radiollava-advancing-visionlanguage-models-for-radio-astronomical-source-analysis/5E14BA0AE0C6196B63E8041CEB934B35
- Monte-Carlo conformal prediction for radio-galaxy classification under ambiguous ground truth — arXiv:2603.20000 (2026, best-effort) — https://arxiv.org/abs/2603.20000

**RFI flagging**

- RFI-Net: deep residual RFI detection for FAST — *MNRAS* (2020), arXiv:2001.06669 — https://arxiv.org/abs/2001.06669
- Deep learning improves RFI identification (MeerKAT R-Net) — arXiv:2005.08992 — https://arxiv.org/abs/2005.08992
- A robust RFI identification for radio interferometry based on a CNN — arXiv:2203.00298 — https://arxiv.org/abs/2203.00298
- RFI-DRUnet: detect-and-restore dynamic spectra corrupted by RFI (pulsar obs) — arXiv:2402.13867 — https://arxiv.org/abs/2402.13867 · *Astron. Comput.* (2024) — https://www.sciencedirect.com/science/article/pii/S2213133724000374
- A comparison framework for deep-learning RFI-detection algorithms — *MNRAS* 530:613 (2024) — https://academic.oup.com/mnras/article/530/1/613/7637224
- Learning to detect RFI without seeing it — *MNRAS* 516:5367 (2022) — https://academic.oup.com/mnras/article/516/4/5367/6692884
- Performance of the Segment Anything Model in RFI/event detection — arXiv:2410.22497 (2024) — https://arxiv.org/abs/2410.22497
- Real-time RFI mitigation techniques and their practical limitations (review) — arXiv:2512.01954 (2025) — https://arxiv.org/abs/2512.01954

**Instruments, surveys & datasets**

- ASKAP EMU — Evolutionary Map of the Universe — *PASA* (2021), arXiv:2103.10163 — https://arxiv.org/abs/2103.10163
- Radio Galaxy Zoo: EMU — citizen science + AI for the ASKAP era — arXiv:2506.16138 — https://arxiv.org/abs/2506.16138 · follow-up — arXiv:2509.19787 — https://arxiv.org/abs/2509.19787
- MeerKAT — MIGHTEE continuum DR1 — arXiv:2411.04958 — https://arxiv.org/abs/2411.04958 · MALS DR1 (~495k sources) — arXiv:2308.12347 — https://arxiv.org/abs/2308.12347
- LOFAR Two-metre Sky Survey (LoTSS) DR2 — arXiv:2202.11733 — https://arxiv.org/abs/2202.11733 · deep-fields host-ID catalogues — arXiv:2011.08201 — https://arxiv.org/abs/2011.08201
- VLA Sky Survey (VLASS) — science case & survey design — arXiv:1907.01981 — https://arxiv.org/abs/1907.01981
- MiraBest (ML-ready FR benchmark, arXiv:2305.11108) · Radio Galaxy Zoo (citizen-science morphology labels) — dataset lineage anchors.
- Fast radio bursts as real-time dense detection: deployed end-to-end DL detector — *A&A* (2025) — https://www.aanda.org/articles/aa/full_html/2025/10/aa55217-25/aa55217-25.html · repeating vs non-repeating FRB morphology — arXiv:2509.06208 — https://arxiv.org/abs/2509.06208

*Diagrams in this entry are hand-authored standalone SVG (no external URLs),
with explicit light-card / dark-panel fills so they render legibly in both light
and dark viewers. Some links were gathered under scraping/API limits and are
provided best-effort; where a landing page was unreachable, an arXiv or DOI
mirror is listed alongside, and very recent 2026 arXiv identifiers I could not
fully re-verify are flagged inline (arXiv IDs in the 2601–2608 range correspond
to Jan–Aug 2026 and should be sanity-checked against the abstract page before
citing downstream). A few pre-2023 works (Högbom CLEAN, PyBDSF, Aegean, ClaRAN,
DeepSource, ProFound, rotation-equivariant CNNs, RFI-Net, POLISH, SDC1) are
included as lineage anchors for otherwise-recent threads.*
