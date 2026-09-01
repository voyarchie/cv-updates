# Dense Object Detection & Classification — Recent Advances

*Compiled 2026-Sep-01 (America/Los_Angeles).*

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
[Aug-26](../2026-Aug-26/2026-Aug-26_CV_updates.md),
[Aug-29](../2026-Aug-29/2026-Aug-29_CV_updates.md).

The last entry closed on the **radio-interferometric image** — a picture of the
sky you never measure directly but synthesize from a sparsely sampled Fourier
plane, an ill-posed inverse problem before it is a vision problem. This one
keeps the inverse problem and the sub-noise signal but shrinks the field of view
by fifteen orders of magnitude, from a galaxy cluster to a protein. The
**single-particle cryo-electron microscopy (cryo-EM) micrograph** is the
primitive: a near-invisible image of thousands of copies of one molecule, frozen
in random orientations in a film of glassy ice, each copy a faint blob buried
*below* the noise floor by construction. On that surface the computer-vision jobs
are unusually literal — **detect** every particle (particle picking), **reject**
the ice and carbon that look just like particles, **classify** the views, and
then solve a joint pose-and-volume inverse problem whose output is not one 3-D
object but a continuous *landscape* of conformations. Cryo-EM won the 2017 Nobel
Prize in Chemistry, and the deep-learning stack around it has become one of the
most consequential dense-vision problems in the life sciences.

> **Scope note & honest caveats.** This is a cross-disciplinary corner of ML
> where most of the strongest work lives in structural-biology and methods
> venues (*Nature Methods*, *Nature Communications*, *IUCrJ*, *Bioinformatics*,
> *Briefings in Bioinformatics*, *PNAS*) rather than CV proceedings — with a
> real but growing overlap into NeurIPS/ICCV/ECCV/MICCAI. Links were gathered
> under scraping/API limits and are provided best-effort; where a landing page
> was flaky, a DOI, PMC mirror, or bioRxiv preprint is given. A handful of
> now-standard tools (Topaz, crYOLO, RELION, cryoSPARC, cryoDRGN, DeepFinder)
> predate 2023 and are included as lineage anchors for otherwise-recent threads.
> Several 2025–2026 preprints are very new: where I could not fully re-verify an
> arXiv ID or DOI, the item is **flagged inline** and the exact title is given so
> it can be found. Treat flagged identifiers as leads, not citations.

---

## Table of contents

1. [Why this pass: the cryo-EM micrograph as its own primitive](#1--why-this-pass-the-cryo-em-micrograph-as-its-own-primitive)
2. [The primitive — projections, dose, and a signal below the noise](#2--the-primitive--projections-dose-and-a-signal-below-the-noise)
3. [Dense detection — particle picking at ≈0 dB](#3--dense-detection--particle-picking-at-0-db)
4. [The false-positive tax — ice, carbon, and contamination](#4--the-false-positive-tax--ice-carbon-and-contamination)
5. [Restoration — denoising and preprocessing before detection](#5--restoration--denoising-and-preprocessing-before-detection)
6. [Classification & the inverse problem — pose, ab-initio volumes](#6--classification--the-inverse-problem--pose-ab-initio-volumes)
7. [Dense classification over conformation — heterogeneity](#7--dense-classification-over-conformation--heterogeneity)
8. [Into the cell — cryo-electron tomography (cryo-ET)](#8--into-the-cell--cryo-electron-tomography-cryo-et)
9. [Benchmarks, datasets & the CZII challenge](#9--benchmarks-datasets--the-czii-challenge)
10. [Foundation & self-supervised models across the pipeline](#10--foundation--self-supervised-models-across-the-pipeline)
11. [Why a micrograph is *not* a natural image](#11--why-a-micrograph-is-not-a-natural-image)
12. [Open problems / what to watch](#12--open-problems--what-to-watch)
13. [Sources](#13--sources)

---

## 1 · Why this pass: the cryo-EM micrograph as its own primitive

Six properties make the cryo-EM micrograph worth treating as a first-class
dense-vision surface rather than "a grayscale biology photo":

1. **The signal is below the noise, by design.** A biological specimen tolerates
   only a tiny electron dose before radiation destroys it, so every micrograph is
   acquired at a signal-to-noise ratio near or below **0 dB** — the particle you
   must detect is, per-pixel, quieter than the shot noise around it. Detection
   here is not "find the object in the clutter"; it is "find the object *beneath*
   the clutter, thousands of times, without hallucinating the clutter."

2. **You never measure the object — you measure noisy 2-D projections of it, at
   unknown angles.** Each particle is a line-integral projection of a 3-D
   Coulomb potential taken from a *random, unknown* orientation. The 3-D
   structure exists nowhere in any single image; it must be reconstructed from
   ~10⁵–10⁷ projections whose poses are latent variables solved *jointly* with
   the volume. This is the cryo-EM analogue of the radio interferometer's
   missing Fourier coverage — an inverse problem before it is a vision problem.

3. **The point-spread function is oscillatory and must be estimated per image.**
   Defocus and the microscope's optics impose a **Contrast Transfer Function
   (CTF)** — a signed, ringing modulation of spatial frequency that flips
   contrast at intervals and zeroes out whole frequency bands. Nothing downstream
   is trustworthy until the CTF is estimated and corrected. "Sharpening" a
   micrograph without it makes things worse, not better.

4. **The distractors are indistinguishable from the targets.** Crystalline ice,
   the carbon or gold support film's edges, denatured aggregates, and ethane
   contamination all produce high-contrast blobs that a naïve detector picks
   preferentially — they are *brighter* than the real particles. A picker that
   optimizes raw contrast picks exactly the wrong things.

5. **Detection is recall-first for a physical reason.** A missed particle view is
   not just one lost sample; if a whole *orientation* is under-picked, it becomes
   a **hole in Fourier space** and the reconstruction develops directional
   artifacts (the "preferred-orientation" problem). The cost function is
   asymmetric in a way natural-image detection rarely is — the 2024–25 CZII
   benchmark literally scored with Fβ at β=4 to encode this.

6. **The output is a distribution, not an object.** Proteins are not rigid; they
   flex, bind partners, and swap subunits. The honest reconstruction target is a
   *continuous landscape* of conformations, so the final "classification" stage
   is dense in a peculiar sense: every particle gets a coordinate in a learned
   latent space of shapes, and the deliverable is a manifold of volumes.

Everything below follows the pipeline these six properties force. Two figures
frame it. The first treats a single micrograph as a dense scene — frozen
particles, a near-invisible image, picking, classes, and a moving structure. The
second lays out the whole deep-learning landscape as a chain of CV tasks, with a
parallel in-cell branch for cryo-electron tomography.

![The cryo-EM micrograph as a dense detection-and-classification scene: frozen particles in random poses, a near-invisible noisy micrograph, particle picking that must reject ice and carbon, 2-D class averages, and a 3-D reconstruction that is really a continuous landscape of conformations.](assets/cryoem-micrograph-as-dense-scene.svg)

---

## 2 · The primitive — projections, dose, and a signal below the noise

The measurement chain is short and brutal. A purified protein solution is
plunge-frozen so fast the water becomes **vitreous** (glassy, non-crystalline)
ice, trapping molecules in random 3-D orientations. A **low-dose** electron beam
projects the whole field onto a **direct-electron detector**, which reads out a
short *movie* of frames (so that beam-induced motion can be tracked and
corrected). The result is a micrograph in which:

- each particle is a **weak-phase-object projection** — approximately a
  line integral of the 3-D electrostatic potential along the beam;
- the per-particle **pose** (three Euler angles + a 2-D shift) is unknown;
- the image is modulated by the **CTF** and swamped by shot noise.

The classical single-particle-analysis (SPA) pipeline — implemented in
**RELION** ([Zivanov et al., *eLife* 2018](https://elifesciences.org/articles/42166))
and **cryoSPARC** ([Punjani et al., *Nature Methods* 2017](https://www.nature.com/articles/nmeth.4169))
— turns this into a 3-D map by: motion-correcting movies, estimating the CTF,
**picking** particles, **2-D-classifying** them to discard junk, computing an
**ab-initio** low-resolution volume, and **refining** poses and volume to
high resolution. Deep learning has now inserted itself into every one of those
boxes; the rest of this report walks them in order.

The key mental model for a vision reader: cryo-EM is **tomography without known
angles**. In medical CT ([Jul-15](../2026-Jul-15/2026-Jul-15_CV_updates.md),
[Jul-07](../2026-Jul-07/2026-Jul-07_CV_updates.md)) you know the projection
geometry and invert a Radon transform. In cryo-EM the geometry is a *latent
variable per image*, the signal is far weaker, and the object is
conformationally heterogeneous — which is exactly why the field has become a
magnet for learned amortized-inference and neural-field methods.

---

## 3 · Dense detection — particle picking at ≈0 dB

**Particle picking** is the dense-detection heart of the field: localize every
true particle in every micrograph, output a box (or a point) per particle, feed
the stack forward. It is where CV architectures land most directly.

**The standing baselines.** Three tools still anchor most real pipelines:

- **Topaz** — a **positive–unlabeled (PU) learning** CNN that trains from a
  handful of clicked particles by treating the vast unlabeled remainder
  correctly (as a mix of positives and negatives), not as background. It is the
  most widely used learned picker and generalizes across particle sizes.
  [Bepler et al., *Nature Methods* 2019](https://www.nature.com/articles/s41592-019-0575-8)
  · [preprint arXiv:1803.08207](https://arxiv.org/abs/1803.08207)
  · [topaz.csail.mit.edu](https://topaz.csail.mit.edu/)
- **SPHIRE-crYOLO** — a **YOLO** object detector adapted to micrographs; fast,
  with a general pretrained network that often works out-of-the-box.
  [Wagner et al., *Communications Biology* 2019](https://www.nature.com/articles/s42003-019-0437-z)
  (filament/denoise follow-up: [*Commun. Biol.* 2020](https://www.nature.com/articles/s42003-020-0790-y)).
- **cryoSPARC blob/template picker** and **RELION LoG/template autopick** — the
  fast classical detectors built into the two dominant SPA suites
  ([cryoSPARC blob-picker docs](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/particle-picking/job-blob-picker)).

Older CNN pickers remain lineage anchors: **DeepPicker** (sliding-window CNN,
[Wang et al., *J. Struct. Biol.* 2016 — exact DOI unverified, likely 10.1016/j.jsb.2016.07.006]),
**DRPnet** (regression detector + false-positive classifier,
[Nguyen et al., *BMC Bioinformatics* 2021](https://link.springer.com/article/10.1186/s12859-020-03948-x)),
and **Warp/BoxNet** (real-time on-the-fly picking,
[Tegunov & Cramer, *Nature Methods* 2019](https://www.nature.com/articles/s41592-019-0580-y)).

**The 2023–2026 wave** brings transformers, foundation-model segmentation, and
label-efficiency:

- **CryoTransformer** — an end-to-end **DETR-style detection transformer**
  (ResNet-152 backbone + a denoising front end) that models long-range relations
  between particles; trained on CryoPPP, it improves F1 and, more tellingly, the
  *resolution of the reconstructed 3-D map* from its picks.
  [Dhakal et al., *Bioinformatics* 40(3):btae109, 2024](https://academic.oup.com/bioinformatics/article/40/3/btae109/7614090)
  · [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.10.19.563155v1)
  · code: `github.com/jianlin-cheng/CryoTransformer`.
- **CryoSegNet** — couples Meta's **Segment Anything Model (SAM)** with a
  specialized **attention-gated U-Net**: SAM alone fails on cryo-EM (it never saw
  such images in training), so the U-Net produces SAM-friendly inputs. Reported
  ~3.33 Å average map resolution, ahead of Topaz (3.58 Å) and crYOLO (3.87 Å).
  [Gyawali et al., *Briefings in Bioinformatics* 25(4):bbae282, 2024](https://academic.oup.com/bib/article/25/4/bbae282/7690949)
  · [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.10.02.560572v2).
- **UPicker** — a **semi-supervised DETR** picker with unsupervised pretraining
  on unlabeled micrographs, cutting the annotation burden.
  [*Briefings in Bioinformatics* 26(1):bbae636, 2025](https://academic.oup.com/bib/article/26/1/bbae636/7919967).
- **cryo-EMMAE** — a fully **annotation-free** self-supervised picker: cluster
  the latent space of a **masked autoencoder (MAE)**, converge from ~5
  micrographs, and generalize across EMPIAR entries with no labels at all.
  [*Cell Reports Methods* 2025](https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(25)00125-0)
  · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12296464/)
  · [code](https://github.com/azamanos/Cryo-EMMAE).
- **CryoMAE** — **few-shot** picking with masked autoencoders and a
  self-cross-similarity loss separating particle from non-particle latents.
  [arXiv:2404.10178, 2024](https://arxiv.org/abs/2404.10178).
- **CryoSAM** — **training-free**, prompt-based instance segmentation of
  particles using 2-D foundation models with cross-plane self-prompting (aimed
  largely at tomograms).
  [Zhao et al., *MICCAI 2024*](https://papers.miccai.org/miccai-2024/178-Paper0532.html)
  · [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72111-3_12).
- **EPicker** — exemplar-based **continual learning** that accumulates picking
  knowledge across datasets without forgetting.
  [*Nature Communications* 2022](https://www.nature.com/articles/s41467-022-29994-y).
- **CryoPromptSeg** — prompt-guided segmentation with integrated denoising
  (*Bioinformatics* advance article; **DOI string `btag327` unverified**),
  [PDF](https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btag327/68370724/btag327.pdf).
- **GTpick** — a 2025 deep detection network for particles
  (*ScienceDirect* [S2001037025004337](https://www.sciencedirect.com/science/article/pii/S2001037025004337); **exact DOI unverified**).

The trajectory mirrors natural-image detection's own history — hand-crafted →
CNN → YOLO/DETR → foundation-model-adapted → self-supervised/label-free — but
arriving a few years later and reshaped by the ≈0-dB noise floor: nearly every
advance is really a way of **buying signal back** (PU learning, denoise-then-pick,
MAE latents) rather than a new box regressor.

---

## 4 · The false-positive tax — ice, carbon, and contamination

Because the distractors are *brighter* than the targets (property 4 above),
false-positive suppression is a first-class problem, not a post-hoc filter:

- **MicrographCleaner** (`micrograph_cleaner_em`) — a U-Net that masks regions
  unsuitable for picking (carbon-film edges, crystalline-ice patches,
  high-contrast contamination) so downstream pickers never see them.
  [Sanchez-Garcia et al., *J. Struct. Biol.* 2020](https://www.sciencedirect.com/science/article/pii/S1047847720300642)
  (DOI 10.1016/j.jsb.2020.107498 — **likely, unverified**).
- **CRISP** — a cryo-EM segmentation/processing framework using **conditional
  random fields** to prune spurious detections.
  [arXiv:2502.08287, 2025](https://arxiv.org/abs/2502.08287).
- **Synthetic hard negatives.** **CryoCCD** uses a conditional cycle-consistent
  **diffusion** model with biophysical modeling to synthesize realistic
  micrographs; fine-tuning Topaz on its output improves particle/background
  discrimination (AUPRC).
  [arXiv:2505.23444, 2025](https://arxiv.org/abs/2505.23444).

Modern pickers also fold rejection into the detector itself: CryoSegNet's U-Net
learns to leave carbon/ice unsegmented, CryoTransformer's denoising front end
suppresses the high-contrast textures that mislead contrast-based pickers, and
labeled benchmarks now *include* common contaminants as explicit negatives so
classifiers learn to say no. This is the same "false-positive tax" seen in
medical CADe work ([polyp detection, Jul-26](../2026-Jul-26/2026-Jul-26_CV_updates.md)) —
recall-first detection creates a precision problem that a second learned stage
must pay down.

---

## 5 · Restoration — denoising and preprocessing before detection

Detection quality is bounded by what the restoration stage recovers, so learned
denoising is upstream and load-bearing:

- **Topaz-Denoise** — general **Noise2Noise**-style deep denoisers for both
  micrographs and tomograms, trained on independent noise realizations (e.g.
  even/odd frame halves); integrated into cryoSPARC, RELION, and Scipion.
  [Bepler et al., *Nature Communications* 2020](https://www.nature.com/articles/s41467-020-18952-1).
- **Warp** — real-time, deep-learning-assisted movie/micrograph preprocessing
  (motion, CTF, denoise, picking) on the fly during collection.
  [Tegunov & Cramer, *Nature Methods* 2019](https://www.nature.com/articles/s41592-019-0580-y).
- **Diffusion restoration** — a **diffusion**-based framework for robust
  single-particle denoising/restoration that preserves high-frequency structure
  ([arXiv:2401.01097, 2024](https://arxiv.org/abs/2401.01097)); and **CryoDDM**,
  a denoising diffusion model aimed at heterogeneous reconstruction (bioRxiv
  2025; **DOI string unusual, verify**).

The design constraint that governs all of these: cryo-EM denoising must *not*
invent high-frequency detail, because that detail is the scientific signal and
fabricating it silently corrupts the final atomic model. Noise2Noise's appeal is
precisely that it never sees a "clean" target to hallucinate toward — it only
learns to average independent noise away.

---

## 6 · Classification & the inverse problem — pose, ab-initio volumes

Once particles are picked, **2-D classification** groups them by apparent
orientation and averages within each group, yielding clean **class averages**
that (a) act as a second false-positive filter — junk classes are discarded —
and (b) reveal whether the data support a real, multi-view 3-D object at all.
Both RELION and cryoSPARC do this classically; the frontier work is in the
**3-D inverse problem** that follows.

The central difficulty is that **pose is unknown**. Classical SPA solves it by
exhaustive per-particle projection matching against a running volume — accurate
but expensive. The deep-learning move is **amortized inference**: train an
encoder to *predict* each particle's pose (and latent conformation), pair it
with a **physics-based decoder** that renders the implied projection, and
optimize the volume and the encoder jointly — a VAE over images with a
differentiable imaging model.

- **CryoAI** — amortized pose inference + a physics decoder writing into an
  **implicit neural volume** (SIREN), for **ab-initio homogeneous**
  reconstruction with gradient-based (not search-based) pose estimation.
  [Levy et al., *ECCV 2022* · arXiv:2203.08138](https://arxiv.org/abs/2203.08138)
  · [project page](https://www.computationalimaging.org/publications/cryoai/).
- **CryoFIRE** — extends amortized inference to **ab-initio heterogeneous**
  reconstruction with unknown poses: the encoder jointly estimates pose and
  conformation while a physics decoder aggregates images into an implicit
  representation of *conformational space* — reported ~10× speedups on
  million-image datasets.
  ["Amortized Inference for Heterogeneous Reconstruction in Cryo-EM", *NeurIPS 2022* · arXiv:2210.07387](https://arxiv.org/abs/2210.07387).
- **cryoDRGN-AI (DRGN-AI)** — neural **ab-initio** reconstruction robust enough
  for challenging cryo-EM *and* cryo-ET datasets, unifying pose search with the
  cryoDRGN latent model.
  [*Nature Methods* 2025 (DOI 10.1038/s41592-025-02720-4)](https://doi.org/10.1038/s41592-025-02720-4).
- **cryoDRGN2** — added joint ab-initio pose search to the cryoDRGN neural model
  on real images.
  [Zhong et al., *ICCV 2021*](https://openaccess.thecvf.com/content/ICCV2021/html/Zhong_CryoDRGN2_Ab_Initio_Neural_Reconstruction_of_3D_Protein_Structures_From_ICCV_2021_paper.html).
- **CryoGAN / Multi-CryoGAN** — unsupervised **distribution-matching**
  reconstruction: a GAN discriminator plays against a cryo-EM *physics simulator*
  generator, sidestepping explicit pose estimation.
  [Gupta et al., *IEEE Trans. Comput. Imaging* 2021](https://ieeexplore.ieee.org/document/9483649)
  · [Multi-CryoGAN (OpenReview)](https://openreview.net/forum?id=5PSL-CjHeP4).
- **ACE-EM** — an encoder–decoder scheme that improves ab-initio neural 3-D
  reconstruction efficiency.
  [arXiv:2302.06091, 2023](https://arxiv.org/abs/2302.06091).

**Preferred-orientation / pose bias** — the direct consequence of property 5 — is
now its own learned sub-problem:

- **spIsoNet** — self-supervised correction of map **anisotropy** and alignment
  errors caused by preferred orientation, restoring angular isotropy.
  [*Nature Methods* 2024 (DOI 10.1038/s41592-024-02505-1)](https://doi.org/10.1038/s41592-024-02505-1).
- **CryoPROS** — generates AI **auxiliary particles** that are co-refined with
  real particles to correct preferred-orientation misalignment.
  [arXiv:2309.14954](https://arxiv.org/abs/2309.14954)
  · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12084624/).

---

## 7 · Dense classification over conformation — heterogeneity

This is where cryo-EM's "classification" stops resembling ordinary image
classification and becomes **dense estimation of a continuous latent variable
per particle**. Instead of one 3-D map, the goal is a *landscape*: every particle
gets coordinates in a learned space whose axes are the molecule's real motions
and compositional states. The pioneering and now-standard neural approach:

- **cryoDRGN** — a VAE / coordinate-network that learns a **continuous latent
  space of 3-D density maps**, exposing residual heterogeneity (ribosome states,
  spliceosome motions) invisible to discrete classification.
  [Zhong et al., *Nature Methods* 2021](https://www.nature.com/articles/s41592-020-01049-4)
  · [code](https://github.com/ml-struct-bio/cryodrgn).

The field around it has since diversified into neural, physics-motivated, and
"white-box" statistical variants:

- **3DFlex (3D Flexible Refinement)** — a physically motivated **neural
  deformation-field** model of continuous motion, in cryoSPARC.
  [Punjani & Fleet, *Nature Methods* 2023](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/variability/job-3d-flexible-refinement-3dflex-beta)
  (bioRxiv 2021: 10.1101/2021.04.22.440893).
- **cryoSTAR** — uses an **atomic-model structural prior** to resolve continuous
  heterogeneity, emitting both density maps *and* coarse-grained models.
  [*Nature Methods* 2024 (DOI 10.1038/s41592-024-02486-1)](https://doi.org/10.1038/s41592-024-02486-1).
- **RECOVAR** — a **white-box** alternative: regularized covariance **PCA** plus
  adaptive kernel-regression reconstruction, competitive with neural methods
  while remaining interpretable.
  [Gilles & Singer, *PNAS* 2025 (DOI 10.1073/pnas.2419140122)](https://doi.org/10.1073/pnas.2419140122)
  · [code](https://github.com/ma-gilles/recovar).
- **DynaMight** — learns per-particle **3-D deformation fields** of a Gaussian
  consensus model; shipped in RELION-5.
  [Schwab et al., *Nature Methods* 21:1855–1862, 2024 (DOI 10.1038/s41592-024-02377-5)](https://doi.org/10.1038/s41592-024-02377-5).
- **OPUS-DSD** — a 3-D convolutional encoder–decoder for **deep structural
  disentanglement** of the heterogeneity landscape.
  [Luo et al., *Nature Methods* 2023 (DOI 10.1038/s41592-023-02031-6)](https://doi.org/10.1038/s41592-023-02031-6).
- **e2gmm** — a deep-network **Gaussian-mixture** model mapping
  conformational/compositional variability to a latent space (EMAN2).
  [Chen & Ludtke, *Nature Methods* 2021 (DOI 10.1038/s41592-021-01220-5)](https://doi.org/10.1038/s41592-021-01220-5).
- **ManifoldEM** — the manifold-embedding (diffusion-map) baseline that recovers
  continuous conformational *trajectories* — a classic non-deep method still
  cited as the statistical reference point (Dashti et al., "Retrieving functional
  pathways of biomolecules from single-particle snapshots", *Nature
  Communications* 2020 — **find by title; exact DOI unverified**).

For a CV reader the analogy is precise: this is **dense regression of a
per-instance latent code** — the same shape as monocular depth or point tracking
([Jun-22](../2026-Jun-22/2026-Jun-22_CV_updates.md)) — except the "instance" is a
molecule, the latent axis is a physical motion, and the supervision comes entirely
through a differentiable imaging model rather than labels.

---

## 8 · Into the cell — cryo-electron tomography (cryo-ET)

Single-particle analysis needs *purified* protein. **Cryo-electron tomography**
drops that requirement: it tilts a frozen cell (or a thin lamella milled from
one) through a series of angles and back-projects the tilt series into a **3-D
tomogram** of the molecular machinery *in its native, crowded context*. The
vision problem changes shape — now it is **dense 3-D detection and
classification inside a volume**, with two extra curses: an even lower SNR, and
the **"missing wedge"** (angles the stage cannot reach leave a wedge of Fourier
space unmeasured, smearing structures anisotropically — the direct 3-D cousin of
the interferometer's incomplete uv-coverage from [Aug-29](../2026-Aug-29/2026-Aug-29_CV_updates.md)).

**3-D particle localization (dense detection in tomograms):**

- **DeepFinder** — a 3-D-CNN that produces a multi-class segmentation voxel map
  and localizes complexes by clustering; the SHREC-challenge front-runner.
  [Moebel et al., *Nature Methods* 2021 (DOI 10.1038/s41592-021-01275-4)](https://doi.org/10.1038/s41592-021-01275-4).
- **TomoTwin** — **deep metric learning**: embed sub-volumes so that same-species
  particles cluster, enabling **de-novo, retraining-free** localization of new
  proteins by picking exemplars in embedding space.
  [Rice et al., *Nature Methods* 2023 (DOI 10.1038/s41592-023-01878-z)](https://www.nature.com/articles/s41592-023-01878-z).
- **DeePiCt** — a combined **2-D + 3-D CNN** framework ("deep picker in context")
  that segments cellular compartments *and* localizes complexes, using context to
  improve both.
  [de Teresa-Trueba et al., *Nature Methods* 2023 (DOI 10.1038/s41592-022-01746-2)](https://doi.org/10.1038/s41592-022-01746-2)
  · [code](https://github.com/ZauggGroup/DeePiCt).
- **DeepETPicker** — fast, accurate 3-D picking via **weakly supervised** deep
  learning (few coarse labels).
  [Liu et al., *Nature Communications* 2024 (DOI 10.1038/s41467-024-46041-0)](https://www.nature.com/articles/s41467-024-46041-0).

**Denoising & the missing wedge:**

- **CryoCARE** — Noise2Noise denoising of tomograms from independent even/odd
  frame halves (Buchholz et al., 2019 — **find by title "cryoCARE content-aware
  image restoration cryo-ET"; exact DOI unverified**).
- **IsoNet** — self-supervised recovery of missing-wedge information and SNR gain,
  producing near-isotropic tomograms *without* subtomogram averaging.
  [Liu et al., *Nature Communications* 2022 (DOI 10.1038/s41467-022-33957-8)](https://doi.org/10.1038/s41467-022-33957-8).
- **DeepDeWedge** — simultaneous denoising **and** missing-wedge reconstruction
  from tilt series, Noise2Noise-based.
  [*Nature Communications* 2024 (DOI 10.1038/s41467-024-51438-y)](https://www.nature.com/articles/s41467-024-51438-y)
  · [arXiv:2311.05539](https://arxiv.org/abs/2311.05539).

**In-situ heterogeneity** brings the cryoDRGN idea into the cell:

- **tomoDRGN** — the cryoDRGN architecture adapted to **subtomograms** for
  per-particle conformational/compositional heterogeneity in situ.
  [Powell & Davis, *Nature Methods* 2024 (DOI 10.1038/s41592-024-02210-z)](https://doi.org/10.1038/s41592-024-02210-z).
- **cryoDRGN-ET** — heterogeneous reconstruction of dynamic biomolecules from
  in-cell tilt-series data.
  [*Nature Methods* 2024 (DOI 10.1038/s41592-024-02340-4)](https://doi.org/10.1038/s41592-024-02340-4).

---

## 9 · Benchmarks, datasets & the CZII challenge

Progress here has been gated less by architectures than by **labeled ground
truth** — hand-annotating particles in ≈0-dB images is slow and expert-only.

- **EMPIAR** — the Electron Microscopy Public Image Archive, the raw-data
  repository nearly every benchmark draws from.
  [ebi.ac.uk/empiar](https://www.ebi.ac.uk/empiar/) (canonical citation Iudin et
  al. — **exact reference to confirm**).
- **CryoPPP** — the de-facto ML picking benchmark: an expert-curated labeled set
  spanning **34 EMPIAR proteins, 9,893 micrographs, ~2.6 TB**, validated against
  2-D-class and 3-D-map gold standards.
  [Dhakal et al., *Scientific Data* 2023 (DOI 10.1038/s41597-023-02280-2)](https://www.nature.com/articles/s41597-023-02280-2)
  · [code](https://github.com/BioinfoMachineLearning/cryoppp).
- **CryoBench** — datasets and metrics targeting the *heterogeneity* problem
  across cryoDRGN-family methods.
  [arXiv:2408.05526, 2024](https://arxiv.org/abs/2408.05526).

**The event that reset cryo-ET picking — the CZII CryoET Object Identification
challenge.** Run on Kaggle by the **Chan Zuckerberg Imaging Institute** from
**November 2024 to February 2025** with ~900+ teams, it asked competitors to
localize **six particle classes** (apo-ferritin, β-amylase, β-galactosidase,
ribosome, thyroglobulin, virus-like particle) in real tomograms. Two design
choices are worth a vision reader's attention:

1. It scored with **Fβ at β=4**, weighting recall four-to-one over precision — a
   direct encoding of property 5 (a missed orientation is a hole in Fourier
   space).
2. The winning solutions were overwhelmingly **3-D U-Net semantic-segmentation**
   pipelines with heavy augmentation and ensembling — heatmap regression, not box
   detection — a useful data point on what actually wins dense 3-D localization
   under extreme noise and tiny label budgets.

[Competition](https://www.kaggle.com/competitions/czii-cryo-et-object-identification)
· [CryoET Data Portal dataset ds-10441](https://cryoetdataportal.czscience.com/datasets/10441)
· lessons-learned write-ups: [*Nature Methods* 2026 (DOI 10.1038/s41592-026-03198-4)](https://www.nature.com/articles/s41592-026-03198-4)
and [*Microsc. Microanal.* 2025](https://academic.oup.com/mam/article/31/Supplement_1/ozaf048.496/8212398)
· a strong baseline model, **TopCUP** (Top CryoET U-Net Picker), is published on
CZI's [Virtual Cells Platform](https://virtualcellmodels.cziscience.com/model/topcup).

---

## 10 · Foundation & self-supervised models across the pipeline

The newest turn is the same one every dense-vision modality in this log has taken:
away from one hand-tuned model per protein, toward a **pretrained backbone**
adapted with few labels.

- **Cryo-IEF + CryoWizard** — a foundation model pretrained **unsupervised on
  ~65M particle images**, feeding a fully automated SPA pipeline.
  ["A comprehensive foundation model for cryo-EM image processing", *Nature Methods* 2025 (DOI 10.1038/s41592-025-02916-8)](https://doi.org/10.1038/s41592-025-02916-8)
  (companion automation paper DOI 10.1038/s41592-025-02917-7 — **pairing to confirm**).
- **CryoEngine + APT-ViT** — a synthetic-data engine (**904k subtomograms across
  452 classes**) plus a phase-tokenized, SE(3)-aware ViT for cryo-ET.
  [arXiv:2509.24311, 2025](https://arxiv.org/abs/2509.24311).
- **cryo-EMMAE** (above) and **CryoMAE** — MAE-latent self-supervision applied
  directly to picking, the label-free end of the same idea.
- **CryoLVM** — JEPA-style large-vision-model pretraining on experimental density
  maps (**arXiv:2602.02620 — 2026 ID, verify**;
  [OpenReview](https://openreview.net/forum?id=9xcvEF2BRi)).

Surveys tracking the whole shift:

- ["Artificial intelligence in cryo-EM protein particle picking: recent advances
  and remaining challenges", *Briefings in Bioinformatics* 26(1):bbaf011, 2025](https://academic.oup.com/bib/article/26/1/bbaf011/7958312).
- ["Artificial intelligence in cryo-EM ... particle picking, map reconstruction,
  modelling", *BMC Artificial Intelligence* 2025 (DOI 10.1186/s44398-025-00017-2)](https://link.springer.com/article/10.1186/s44398-025-00017-2).
- ["Deep generative priors for biomolecular 3D heterogeneous reconstruction from
  cryo-EM projections", 2024 (PubMed 38432598)](https://pubmed.ncbi.nlm.nih.gov/38432598/).
- ["Deep Generative Modeling for Volume Reconstruction in Cryo-Electron
  Microscopy", survey, arXiv:2201.02867](https://arxiv.org/abs/2201.02867).
- ["Cryo-electron tomography: Challenges and computational strategies for particle
  picking", *Curr. Opin. Struct. Biol.* 2025](https://www.sciencedirect.com/science/article/pii/S0959440X25001319).

---

## 11 · Why a micrograph is *not* a natural image

A compact contrast for anyone importing an off-the-shelf detector or classifier:

| Property | Natural image | Cryo-EM micrograph |
|---|---|---|
| **SNR** | high; object brighter than background | **≈0 dB or below** — signal *beneath* the noise |
| **What you measure** | the object's appearance | a **noisy 2-D projection** at an **unknown angle** |
| **PSF** | mild, roughly Gaussian blur | oscillating, sign-flipping **CTF**, estimated per image |
| **Distractors** | usually dimmer / different | **brighter** than targets (ice, carbon, aggregates) |
| **Error asymmetry** | precision ≈ recall | **recall-first** — a missed pose is a Fourier-space hole |
| **Label supply** | abundant (web-scale) | scarce, expert-only, per-dataset |
| **Output** | a box / class / mask | a **pose + a continuous conformational manifold** |
| **Supervision** | direct labels | often via a **differentiable physics decoder** |
| **3-D variant (cryo-ET)** | multi-view with known geometry | tilt series with a **missing wedge**, in-cell crowding |

The through-line with the rest of this series: like SAR ([Jul-22](../2026-Jul-22/2026-Jul-22_CV_updates.md)),
radio interferometry ([Aug-29](../2026-Aug-29/2026-Aug-29_CV_updates.md)), and
photoacoustics ([Aug-13](../2026-Aug-13/2026-Aug-13_CV_updates.md)), cryo-EM is a
**computational-imaging** modality where the "image" is the *output* of an
ill-posed inverse problem, and the most effective learned methods embed the
forward physics (projection + CTF + noise) directly in the loss rather than
treating pixels as ground truth.

![The deep-learning cryo-EM and cryo-ET pipeline as a chain of dense-vision tasks: restoration (Warp, Topaz-Denoise), dense particle detection (Topaz, crYOLO, CryoTransformer, CryoSegNet, cryo-EMMAE), 2-D classification, the pose-and-volume inverse problem (CryoAI, CryoFIRE, cryoDRGN-AI), and conformational heterogeneity (cryoDRGN, 3DFlex, RECOVAR, cryoSTAR, DynaMight), with a parallel in-cell cryo-ET branch (DeepFinder, TomoTwin, DeePiCt, DeepETPicker; CryoCARE, IsoNet, DeepDeWedge; the CZII challenge) and a foundation-model band (Cryo-IEF, CryoEngine).](assets/cryoem-pipeline-landscape.svg)

---

## 12 · Open problems / what to watch

- **Trustworthy denoising and generative priors.** Diffusion restoration and
  synthetic-particle methods (CryoCCD, CryoPROS, diffusion denoisers) all risk
  **hallucinating high-frequency structure** — the exact quantity being measured.
  The open question is calibration: how to add signal without adding fiction, and
  how to *prove* to a structural biologist that you did.
- **Label-free picking as the default.** cryo-EMMAE and CryoMAE suggest fully
  self-supervised picking is within reach; whether it reaches the map-resolution
  bar of supervised Topaz/CryoSegNet across hard, small, or low-symmetry targets
  is the near-term test.
- **Amortized inference at production quality.** CryoAI/CryoFIRE/cryoDRGN-AI trade
  per-particle search for a learned encoder; robustness on pathological datasets
  (severe preferred orientation, tiny particles, strong heterogeneity) vs. the
  battle-tested RELION/cryoSPARC refiners is still being litigated.
- **The missing wedge as a learned prior.** IsoNet/DeepDeWedge treat missing-wedge
  recovery as self-supervised inpainting; how far a learned prior can honestly
  fill unmeasured Fourier space — without inventing biology — is the cryo-ET
  analogue of the deconvolution-trust problem in radio and photoacoustics.
- **One benchmark, one metric, is not enough.** CZII was transformative but
  single-institute and six-class; the field needs broader, harder,
  multi-institution dense-detection benchmarks with agreed recall-first metrics —
  and 3-D-map resolution, not just F1, as the terminal score.
- **Foundation models that span the chain.** Cryo-IEF and CryoEngine hint at one
  backbone from movie to map; whether a single pretrained model can serve
  restoration, picking, pose, and heterogeneity — or whether the physics forces
  specialization at each stage — is the defining question for 2026–27.
- **2-D detection vs. 3-D segmentation for localization.** CZII's U-Net
  segmentation sweep suggests heatmap/segmentation beats box detection under
  extreme noise and scarce labels — a finding worth stress-testing against the
  DETR-style pickers winning in 2-D SPA.

---

## 13 · Sources

Grouped roughly by section; identifiers flagged **[verify]** were taken from
search listings and not fully re-confirmed this pass — use the title to locate
the canonical record before citing.

**Baseline & classic pickers**
- Topaz — [*Nature Methods* 2019](https://www.nature.com/articles/s41592-019-0575-8) · [arXiv:1803.08207](https://arxiv.org/abs/1803.08207) · [tool](https://topaz.csail.mit.edu/)
- SPHIRE-crYOLO — [*Commun. Biol.* 2019](https://www.nature.com/articles/s42003-019-0437-z) · [2020 follow-up](https://www.nature.com/articles/s42003-020-0790-y)
- RELION-3 autopick — [*eLife* 2018](https://elifesciences.org/articles/42166)
- cryoSPARC — [*Nature Methods* 2017](https://www.nature.com/articles/nmeth.4169) · [blob-picker docs](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/particle-picking/job-blob-picker)
- DeepPicker — Wang et al., *J. Struct. Biol.* 2016 **[verify DOI 10.1016/j.jsb.2016.07.006]**
- DRPnet — [*BMC Bioinformatics* 2021](https://link.springer.com/article/10.1186/s12859-020-03948-x)
- Warp — [*Nature Methods* 2019](https://www.nature.com/articles/s41592-019-0580-y)
- EPicker — [*Nature Communications* 2022](https://www.nature.com/articles/s41467-022-29994-y)

**Recent pickers (2023–2026)**
- CryoTransformer — [*Bioinformatics* 2024](https://academic.oup.com/bioinformatics/article/40/3/btae109/7614090) · [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.10.19.563155v1)
- CryoSegNet — [*Brief. Bioinform.* 2024](https://academic.oup.com/bib/article/25/4/bbae282/7690949) · [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.10.02.560572v2)
- UPicker — [*Brief. Bioinform.* 2025](https://academic.oup.com/bib/article/26/1/bbae636/7919967)
- cryo-EMMAE — [*Cell Reports Methods* 2025](https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(25)00125-0) · [code](https://github.com/azamanos/Cryo-EMMAE)
- CryoMAE — [arXiv:2404.10178](https://arxiv.org/abs/2404.10178)
- CryoSAM — [MICCAI 2024](https://papers.miccai.org/miccai-2024/178-Paper0532.html) · [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72111-3_12)
- CryoPromptSeg — [PDF](https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btag327/68370724/btag327.pdf) **[verify DOI]**
- GTpick — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2001037025004337) **[verify DOI]**

**Contamination / false positives & restoration**
- MicrographCleaner — [*J. Struct. Biol.* 2020](https://www.sciencedirect.com/science/article/pii/S1047847720300642) **[verify DOI]**
- CRISP — [arXiv:2502.08287](https://arxiv.org/abs/2502.08287)
- CryoCCD — [arXiv:2505.23444](https://arxiv.org/abs/2505.23444)
- Topaz-Denoise — [*Nature Communications* 2020](https://www.nature.com/articles/s41467-020-18952-1)
- Diffusion restoration — [arXiv:2401.01097](https://arxiv.org/abs/2401.01097)

**Inverse problem, pose & orientation bias**
- CryoAI — [arXiv:2203.08138](https://arxiv.org/abs/2203.08138) · [project](https://www.computationalimaging.org/publications/cryoai/)
- CryoFIRE — [arXiv:2210.07387](https://arxiv.org/abs/2210.07387)
- cryoDRGN-AI — [*Nature Methods* 2025](https://doi.org/10.1038/s41592-025-02720-4)
- cryoDRGN2 — [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhong_CryoDRGN2_Ab_Initio_Neural_Reconstruction_of_3D_Protein_Structures_From_ICCV_2021_paper.html)
- CryoGAN — [*IEEE TCI* 2021](https://ieeexplore.ieee.org/document/9483649) · [Multi-CryoGAN](https://openreview.net/forum?id=5PSL-CjHeP4)
- ACE-EM — [arXiv:2302.06091](https://arxiv.org/abs/2302.06091)
- spIsoNet — [*Nature Methods* 2024](https://doi.org/10.1038/s41592-024-02505-1)
- CryoPROS — [arXiv:2309.14954](https://arxiv.org/abs/2309.14954) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12084624/)

**Heterogeneity**
- cryoDRGN — [*Nature Methods* 2021](https://www.nature.com/articles/s41592-020-01049-4) · [code](https://github.com/ml-struct-bio/cryodrgn)
- 3DFlex — [cryoSPARC 3DFlex](https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/variability/job-3d-flexible-refinement-3dflex-beta)
- cryoSTAR — [*Nature Methods* 2024](https://doi.org/10.1038/s41592-024-02486-1)
- RECOVAR — [*PNAS* 2025](https://doi.org/10.1073/pnas.2419140122) · [code](https://github.com/ma-gilles/recovar)
- DynaMight — [*Nature Methods* 2024](https://doi.org/10.1038/s41592-024-02377-5)
- OPUS-DSD — [*Nature Methods* 2023](https://doi.org/10.1038/s41592-023-02031-6)
- e2gmm — [*Nature Methods* 2021](https://doi.org/10.1038/s41592-021-01220-5)
- ManifoldEM — Dashti et al., *Nature Communications* 2020 **[find by title; verify DOI]**
- CryoBench — [arXiv:2408.05526](https://arxiv.org/abs/2408.05526)

**Cryo-ET**
- DeepFinder — [*Nature Methods* 2021](https://doi.org/10.1038/s41592-021-01275-4)
- TomoTwin — [*Nature Methods* 2023](https://www.nature.com/articles/s41592-023-01878-z)
- DeePiCt — [*Nature Methods* 2023](https://doi.org/10.1038/s41592-022-01746-2) · [code](https://github.com/ZauggGroup/DeePiCt)
- DeepETPicker — [*Nature Communications* 2024](https://www.nature.com/articles/s41467-024-46041-0)
- IsoNet — [*Nature Communications* 2022](https://doi.org/10.1038/s41467-022-33957-8)
- DeepDeWedge — [*Nature Communications* 2024](https://www.nature.com/articles/s41467-024-51438-y) · [arXiv:2311.05539](https://arxiv.org/abs/2311.05539)
- CryoCARE — Buchholz et al. 2019 **[find by title; verify DOI]**
- tomoDRGN — [*Nature Methods* 2024](https://doi.org/10.1038/s41592-024-02210-z)
- cryoDRGN-ET — [*Nature Methods* 2024](https://doi.org/10.1038/s41592-024-02340-4)

**Datasets, benchmarks & the CZII challenge**
- CryoPPP — [*Scientific Data* 2023](https://www.nature.com/articles/s41597-023-02280-2) · [code](https://github.com/BioinfoMachineLearning/cryoppp)
- EMPIAR — [ebi.ac.uk/empiar](https://www.ebi.ac.uk/empiar/) **[verify canonical citation]**
- CZII CryoET Object Identification — [Kaggle](https://www.kaggle.com/competitions/czii-cryo-et-object-identification) · [dataset ds-10441](https://cryoetdataportal.czscience.com/datasets/10441) · [*Nature Methods* 2026](https://www.nature.com/articles/s41592-026-03198-4) · [*Microsc. Microanal.* 2025](https://academic.oup.com/mam/article/31/Supplement_1/ozaf048.496/8212398) · [TopCUP model](https://virtualcellmodels.cziscience.com/model/topcup)

**Foundation models & surveys**
- Cryo-IEF + CryoWizard — [*Nature Methods* 2025](https://doi.org/10.1038/s41592-025-02916-8) **[companion pairing to confirm]**
- CryoEngine / APT-ViT — [arXiv:2509.24311](https://arxiv.org/abs/2509.24311)
- CryoLVM — arXiv:2602.02620 **[2026 ID, verify]** · [OpenReview](https://openreview.net/forum?id=9xcvEF2BRi)
- Picking survey — [*Brief. Bioinform.* 2025 (bbaf011)](https://academic.oup.com/bib/article/26/1/bbaf011/7958312)
- Cryo-EM AI survey — [*BMC AI* 2025](https://link.springer.com/article/10.1186/s44398-025-00017-2)
- Heterogeneous-reconstruction review — [PubMed 38432598](https://pubmed.ncbi.nlm.nih.gov/38432598/)
- Volume-reconstruction generative survey — [arXiv:2201.02867](https://arxiv.org/abs/2201.02867)
- Cryo-ET picking review — [*Curr. Opin. Struct. Biol.* 2025](https://www.sciencedirect.com/science/article/pii/S0959440X25001319)

---

*Generated as part of the recurring CV-updates series. Diagrams are original
standalone SVGs (no external URLs), authored to render legibly in both light and
dark viewers. Where identifiers could not be fully re-verified under this pass's
scraping/API limits they are flagged inline; titles are given so the canonical
record can be found.*
