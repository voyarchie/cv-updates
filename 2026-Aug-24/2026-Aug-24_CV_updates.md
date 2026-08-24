# Dense Object Detection & Classification — Recent Advances

*Compiled 2026-Aug-24 (America/Los_Angeles).*

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
[Aug-22](../2026-Aug-22/2026-Aug-22_CV_updates.md).

The last entry closed on the **astronomical survey image** — an additive,
transparent scene with a known noise model, where detection means *un-summing*
overlapping light into instances. Two entries before that (Aug-15) the log
passed through **seismic reflection imaging**, and inside it briefly met a
sensor that never got its own installment: the telecom fiber turned into a
strain array by **distributed acoustic sensing (DAS)**. That mention was
confined to earthquake seismology — phase picking with PhaseNet-DAS, denoising
with DAS-N2N. But DAS is a much larger detection primitive than "another
seismometer." A single dark fiber under a city street, a highway, a railway, a
gas pipeline, or the seabed becomes **thousands of virtual sensors** whose
strain-rate traces stack into a two-dimensional **channel × time image**, and in
that image *cars, footsteps, diggers, trains, earthquakes and whales each draw a
recognizable shape*. This pass treats that channel-time record as its own
first-class dense-vision modality: find every event, segment its footprint in
the plane, classify what made it, and read off its kinematics — at
whole-fiber, all-day throughput.

## Table of contents

1. [Why this pass: the DAS record as its own primitive](#1--why-this-pass-the-das-record-as-its-own-primitive)
2. [The primitive — a fiber is a dense strain image](#2--the-primitive--a-fiber-is-a-dense-strain-image)
3. [The learning stack — from raw strain to decision](#3--the-learning-stack--from-raw-strain-to-decision)
4. [Denoising & label scarcity — the two structural problems](#4--denoising--label-scarcity--the-two-structural-problems)
5. [Self-supervision & foundation models for DAS](#5--self-supervision--foundation-models-for-das)
6. [Application verticals & recent results](#6--application-verticals--recent-results)
7. [Why a DAS waterfall is *not* a natural image](#7--why-a-das-waterfall-is-not-a-natural-image)
8. [Open problems / what to watch](#8--open-problems--what-to-watch)
9. [Sources](#9--sources)

## 1 · Why this pass: the DAS record as its own primitive

Five properties make a DAS record worth treating as a first-class
dense-detection modality rather than "a stack of geophones":

- **The sensor array is free and already in the ground.** DAS interrogates an
  ordinary single-mode fiber — a spare strand in an existing telecom cable —
  with laser pulses and reads the phase of Rayleigh backscatter (phase-sensitive
  OTDR, Φ-OTDR). Every few metres of fiber becomes one **channel** measuring
  local strain rate. A 40 km cable at 10 m gauge spacing is a **4,000-element
  array** with no field installation. That is the deployment story that keeps
  DAS in the news: monitoring infrastructure and hazards over tens of kilometres
  at the cost of a laser box at one end.
- **The native output is an image, not a point reading.** Stacking every
  channel's trace produces a **channel × time matrix** — the *waterfall plot*.
  This is the modality's fundamental object, and it is genuinely 2-D: structure
  runs along *both* axes. A moving source becomes a **slanted streak** whose
  slope is its inverse speed; a distant impulsive source becomes a **coherent
  moveout curve** across many channels; a localized disturbance is a **compact
  blob** on a few channels. Detection, segmentation and tracking in that plane
  are exactly computer-vision problems.
- **The scene is additive and semi-transparent.** Like the spectrogram (Aug-19),
  the LArTPC readout (Aug-21) and the survey image (Aug-22), overlapping sources
  **sum their strain** rather than occlude one another. Two vehicles crossing on
  the same fiber segment add their signatures; separating them is un-summing, not
  un-occluding — the same instance-segmentation-with-a-physics-twist theme the
  log keeps returning to.
- **The data volume is enormous and almost entirely unlabeled.** Thousands of
  channels at kHz sampling generate **terabytes per day**. The interesting events
  are rare, faint, and buried in strong coherent optical/instrument noise. Almost
  none of it is labeled, and the events that matter most (a slow leak, a first
  footstep, a small earthquake) are exactly the ones no one has annotated. This
  is the defining pressure of the field and it pushes hard toward
  **self-supervision**.
- **Labels, when they exist, come from other instruments.** DAS rarely labels
  itself. Ground truth is *borrowed*: a co-located seismometer's catalog, a
  traffic camera, a loop detector, a controlled tap test, an AIS vessel track, or
  a physics simulator. Cross-instrument, cross-modal supervision is therefore
  central — including, as of late 2025, **pretraining a DAS encoder against
  video** of the same scene.

Add the deployment momentum — reviews in *PhotoniX* and *Infrastructures* this
year surveying AI-driven DAS across geophysics, traffic, security and marine use;
the first public **event-classification datasets** with tens of thousands of
labeled records; and the first **DAS foundation models** — and the setting is
unmistakable: an enormous, cheaply-instrumented, image-shaped stream where the
job is dense detection and fine-grained classification under severe label
scarcity.

## 2 · The primitive — a fiber is a dense strain image

![A DAS record shown as a dense detection scene: a buried telecom fiber becomes thousands of virtual strain sensors whose traces stack into a channel-by-time waterfall image, in which a vehicle draws a slanted line, an earthquake a coherent moveout, a digger a localized burst, and footsteps a faint track — each to be detected, segmented and classified](assets/das-as-dense-scene.svg)

The forward picture is short. A pulsed laser is launched into the fiber; light
is continuously **Rayleigh-backscattered** by frozen-in density fluctuations in
the glass. Strain from a passing acoustic/seismic wave stretches the fiber
locally, shifting the *phase* of the returning light. Comparing the phase at two
points a **gauge length** apart gives the **strain rate** in that bin, at that
instant. Repeating for every range bin and every laser pulse fills a matrix:

- **Rows = channels** — position along the fiber (≈ position in the world, once
  you have tap-tests to map fiber metres to map coordinates).
- **Columns = time** — one column per interrogation, kHz-fast.
- **Pixel value = strain rate** — signed, roughly Gaussian noise plus signal.

Read as an image, the object grammar is remarkably clean:

| World event | Signature in the channel–time image |
|---|---|
| A vehicle driving along the road above the fiber | A **slanted line**; slope = 1/speed, length = how far it stayed near the fiber, brightness ∝ axle load |
| An earthquake / distant impulsive source | A **coherent moveout curve** spanning all channels, arriving as a P then S "edge" sweeping across rows |
| A digger, tap, or footstep near one point | A **spatially compact burst / blob** on a handful of channels, intermittent in time |
| A train | A very bright, long slanted band with internal structure (bogies) |
| A leak / continuous flow | A **stationary horizontal band** — same channels, persistent in time, narrowband |

So the tasks are the familiar dense-vision quartet, just in a strain plane:
**detect** the onset/footprint, **segment** which pixels belong to the event,
**classify** the source, and **regress** its kinematics (speed from the slope,
location from the channel index, magnitude from amplitude). Everything below is a
way of doing that under the two problems the primitive forces on you — noise and
missing labels.

## 3 · The learning stack — from raw strain to decision

![The DAS deep-learning stack in four layers — raw record, preprocessing and denoising, model family (1-D CNN/RNN, 2-D CNN/U-Net, transformer, self-supervised/foundation), and task/output — over five deployed verticals: geophysics, traffic/infrastructure, security, marine, and structural/rail](assets/das-pipeline-landscape.svg)

The community has converged on a four-layer stack, with the model layer split by
**how the record is presented to the network**:

- **1-D per-channel models (CNN / RNN).** Treat each channel (or a small
  neighbourhood) as a time series and classify the local event. This is the
  workhorse for security and pipeline monitoring, where "what is happening *at
  this point*?" is the question. Example: a **1-D CNN** for oil-and-gas pipeline
  intrusion (manual tapping vs. mechanical excavation vs. footsteps) reaching
  >95% accuracy by fusing raw traces with handcrafted features and transferring
  from public human-activity datasets ([PLOS ONE 2025](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0338205)).
  **RNN-DAS** ([*JGR: Solid Earth* 2025](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JB031756))
  uses LSTM layers for real-time volcano-tectonic event recognition at >97%
  accuracy and, critically, *generalizes to a fiber it was never trained on*.
- **2-D image models (CNN / U-Net).** Treat the waterfall as an image and reuse
  the whole detection/segmentation toolbox. Roadside-DAS traffic works cast
  vehicle counting as **object detection on the waterfall** (a YOLO-style head
  over the strain image) and speed as reading each streak's slope; a 2025 review
  in [*Infrastructures*](https://www.mdpi.com/2412-3811/10/9/228) catalogs the
  signal-processing-plus-deep-learning pipeline for road traffic. U-Nets do the
  segmentation/denoising duty.
- **Transformers & attention.** The reference point is **PhaseNet-DAS**
  (Zhu et al., *Nature Communications* 2023, met in the Aug-15 pass), which picks
  P/S arrivals on **2-D spatio-temporal DAS gathers** and exploits coherence
  across channels rather than picking each trace alone. For traffic, a 2026
  preprint adds **spatio-temporal attention in a recurrent net** for urban
  vehicle monitoring ([arXiv 2603.13903](https://arxiv.org/pdf/2603.13903)).
- **Self-supervised & foundation models.** The newest and most consequential
  layer — enough to warrant its own section below.

Layers 2 and 4 bracket the models: **preprocessing/denoising** (band-pass and
frequency–wavenumber (f–k) filtering to kill optical noise, plus learned
denoisers) feeds the network, and the **task head** emits picks, masks, class
labels, or regressed speed/location. The single most important design choice is
still layer 3's presentation question — *per-channel time series vs. whole
waterfall image* — because it decides whether cross-channel coherence, the one
feature that most cleanly separates real events from local noise, is even
visible to the model.

## 4 · Denoising & label scarcity — the two structural problems

Two problems dominate every DAS paper, and both are consequences of the
primitive rather than of any one application.

**Denoising, without clean targets.** DAS strain is swamped by coherent optical
noise, common-mode laser drift, and fading. There is never a "clean" copy of the
same record to regress against, so supervised denoising is out. The field's
answer is **self-supervised** in the Noise2Noise family: **DAS-N2N**
([*Geophysical Journal International* 2023/2024](https://academic.oup.com/gji/article/236/2/1026/7453669))
exploits the fact that two adjacent fiber channels see almost the same signal but
*independent* noise, so one can be regressed onto the other with no clean target
— and the trained network then generalizes to denoise cheaply in real time.
Diffusion-based restorers have since been layered on top. This matters for
detection directly: a cleaner waterfall makes faint streaks and moveouts
detectable that were previously under the noise floor.

**Label scarcity, and borrowing labels.** Because DAS does not annotate itself,
the field leans on three tricks, all of which the log has seen in other
modalities:

- **Weak / cross-instrument labels.** PhaseNet-DAS was trained by transferring a
  seismometer-trained picker's outputs onto the fiber via a teacher-plus-GaMMA
  association scheme — the co-located catalog is the label source, no human
  picks DAS gathers.
- **Curated public datasets, finally.** The **"Comprehensive Dataset for Event
  Classification Using DAS"** ([*Scientific Data* 2025](https://www.nature.com/articles/s41597-025-05088-4))
  is the field's first large, openly-released labeled benchmark for supervised
  and few-shot event classification — the DAS analogue of the reference
  benchmarks that reshaped other modalities in this log (FHIBE, MVTec AD 2).
- **Self-supervised pretraining**, discussed next, which sidesteps labels for
  representation learning and needs only a handful of labels to adapt.

## 5 · Self-supervision & foundation models for DAS

This is the year DAS got its own foundation-model moment, and it is the reason
this modality earns a fresh installment now rather than a footnote in the seismic
pass.

- **DAS-MAE — a masked autoencoder for the waterfall.**
  ([arXiv 2506.04552](https://arxiv.org/pdf/2506.04552); *Journal of Lightwave
  Technology* 2026, [OPG](https://opg.optica.org/jlt/abstract.cfm?uri=jlt-44-7-2544).)
  DAS-MAE pretrains a Vision-Transformer-style encoder on unlabeled waterfall
  patches with a **mask-and-reconstruct** objective — the exact recipe that made
  MAE work for natural images, transplanted to strain images. Without using any
  labels in pretraining it reaches **~1% error and a 64.5% relative improvement
  over a semi-supervised baseline in few-shot classification**, and transfers
  across tasks (event recognition, and a downstream detection task) — the first
  convincing "pretrain once, adapt cheaply" result for DAS.
- **A DAS foundation model with visual prompt tuning.**
  ([arXiv 2508.04316](https://arxiv.org/pdf/2508.04316).) Pretrains a
  general-purpose DAS signal-recognition backbone and then adapts it to
  downstream tasks with **visual prompt tuning** — freezing the backbone and
  learning small prompt tokens per task, the parameter-efficient adaptation
  pattern from vision-language models. It targets exactly the DAS pain point:
  many small, differently-instrumented deployments, none with enough labels to
  train from scratch.
- **Cross-modal pretraining against video.**
  ([arXiv 2511.09342](https://arxiv.org/pdf/2511.09342).) The most striking of the
  three: it improves DAS performance and generalization by **pretraining with
  paired video data** of the same scene — using the camera as a supervisory
  teacher for the fiber, so the DAS encoder learns representations aligned with
  what a co-located camera sees. This directly attacks the borrowed-label problem
  by turning an ordinary co-located camera into a scalable label source, and
  echoes the cross-modal supervision themes (touch↔vision, radar↔camera) from
  earlier passes.
- **Better hand-crafted front-ends still matter.** Not everything is
  end-to-end: **wavelet-packet-decomposition features** feeding a classifier
  remain competitive and interpretable for event classification
  ([*Technologies* 2025](https://www.mdpi.com/2227-7080/13/11/514)), and the
  Φ-OTDR perimeter-security study found **MFCC** front-ends topped raw-signal and
  FFT/DWT alternatives (85.6% vs. ~85% for RDFT) with a real-time trade-off
  ([IEEE 2025](https://ieeexplore.ieee.org/document/10955273/)) — a reminder that
  on this modality the *representation handed to the net* is often the whole
  ballgame.

Two field surveys tie it together: an [*PhotoniX* 2025 review](https://photonix.springeropen.com/articles/10.1186/s43074-025-00160-z)
on AI-driven DAS technology and engineering applications, and the traffic-focused
[*Infrastructures* 2025 review](https://www.mdpi.com/2412-3811/10/9/228).

## 6 · Application verticals & recent results

- **Geophysics & early warning.** The founding vertical. PhaseNet-DAS (picking),
  RNN-DAS (volcano-tectonic recognition, cross-fiber generalization), and DAS-N2N
  (denoising) form the core; DAS turns any submarine or urban fiber into a dense
  seismic array for detection, association and — increasingly — **earthquake
  early warning**, where the moveout across thousands of channels gives location
  and magnitude in seconds.
- **Traffic & infrastructure.** Vehicle detection, counting, speed and
  classification from roadside/under-road fiber, cast as **object detection on
  the waterfall** (YOLO-style heads) with slope→speed regression; spatio-temporal
  attention nets for noise-robust urban monitoring
  ([arXiv 2403.02791](https://arxiv.org/pdf/2403.02791),
  [arXiv 2603.13903](https://arxiv.org/pdf/2603.13903)); reviews in
  [*Infrastructures* 2025](https://www.mdpi.com/2412-3811/10/9/228). Bridge and
  railway structural-health monitoring and **train tracking** (Kalman filtering
  on DAS detections) are adjacent.
- **Perimeter & pipeline security.** The largest commercial vertical. Φ-OTDR +
  **CNN** for object classification in the fiber's vicinity
  ([IEEE 2025](https://ieeexplore.ieee.org/document/10955273/)); **1-D CNN**
  intrusion classification for oil-and-gas pipelines at >95% across tapping,
  excavation and footsteps with transfer learning
  ([PLOS ONE 2025](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0338205)).
  The headline metric here is **false-alarm rate**: distinguishing a threat from
  ordinary environmental noise over tens of kilometres is the whole product.
- **Marine mammal & ocean monitoring.** Seabed telecom cables as ocean-scale
  hydrophone arrays. Automated **fin-whale call detection** with DAS in the
  Arctic and Mediterranean ([*JASA* 2025](https://pubs.aip.org/asa/jasa/article/158/6/5057/3375686)),
  offshore-Oregon fin-whale and vessel observations
  ([*Frontiers in Marine Science* 2025](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1603541/full)),
  detecting "silent" whales via seabed fiber
  ([*PNAS* 2026](https://www.pnas.org/doi/abs/10.1073/pnas.2603077123)), and
  end-to-end CNN workflows for song detection, note characterization and
  localization ([arXiv 2608.02387](https://arxiv.org/abs/2608.02387));
  a general [deep-learning framework for marine acoustic + seismic DAS
  monitoring](https://arxiv.org/html/2603.14844) and a broad
  [DAS-for-ocean-applications survey](https://arxiv.org/pdf/2502.18344).
- **Structural & railway health.** Continuous monitoring of bridges, tunnels and
  track from fiber already run alongside them, with the same detect→classify→
  localize loop applied to defect signatures and rolling-stock events.

## 7 · Why a DAS waterfall is *not* a natural image

Reusing ImageNet-shaped detectors naively fails, for reasons specific to the
primitive — the same "know your modality" caution that ran through the
ultrasound, SAR and spectrogram passes:

- **Axes are not interchangeable.** One axis is space along a 1-D curve embedded
  in the world, the other is time. Rotation/flip augmentations that are free on
  photographs are **physically meaningless** here (flipping time reverses
  causality; the channel axis has a fixed metric via tap-tests). Augmentation has
  to respect the physics.
- **The "objects" are lines and moveouts, not blobs.** The most informative cue
  is a **slope** (speed) or a **curvature** (moveout ↔ distance), so the network
  must be sensitive to oriented, elongated structure spanning the whole image —
  which is why f–k / slant-stack front-ends and attention across channels help so
  much, and why plain small-receptive-field CNNs underperform.
- **Coupling is position-dependent and unknown.** How strongly a given event
  couples into the fiber depends on burial, soil, and the angle between the fiber
  and the wave — the same source looks different on two cables, so **cross-fiber
  generalization** (the property RNN-DAS explicitly claims) is the hard
  benchmark, not in-distribution accuracy.
- **Noise is coherent, not i.i.d.** Optical/laser noise is correlated across
  channels and time, so it *mimics real events*. Detectors that assume
  independent pixel noise (as many natural-image models implicitly do) hallucinate
  events; this is why self-supervised denoising and cross-channel coherence tests
  are load-bearing.
- **Labels are borrowed and sparse.** As in the LArTPC and survey passes, the
  hard regime is unlabeled; the field's progress is measured as much by *label
  efficiency* (few-shot, self-supervised, cross-modal) as by peak accuracy.

## 8 · Open problems / what to watch

- **A real DAS foundation model, cross-deployment.** DAS-MAE and the
  visual-prompt-tuning model are first steps; the open goal is one encoder that
  transfers across interrogators, gauge lengths, and burial conditions with a
  handful of labels — the thing that would make DAS "plug-in" the way ImageNet
  features are for photographs.
- **Camera/AIS/seismometer as scalable teachers.** Cross-modal pretraining
  against video ([2511.09342](https://arxiv.org/pdf/2511.09342)) hints that the
  label problem is solvable by pairing fiber with *any* co-located instrument;
  expect vessel-AIS, traffic-camera and seismometer teachers to proliferate.
- **Detection *and* localization as one head.** Speed/position are currently read
  off post-hoc from streak geometry; end-to-end heads that emit calibrated
  location + kinematics + uncertainty (as marine-localization workflows are
  starting to do) are the natural next step.
- **Standardized benchmarks & false-alarm reporting.** The *Scientific Data* 2025
  release is a start; the field still lacks shared, cross-fiber test sets and a
  norm of reporting false-alarm rates at fixed sensitivity for the security
  vertical.
- **Real-time at fiber scale.** Terabytes/day forces streaming, on-the-fly
  inference; efficient models and the quantization/edge story this log has
  tracked (Q-DETR-style INT8, May-15) apply directly here.

## 9 · Sources

**Foundation / self-supervised models**
- DAS-MAE — *A self-supervised pre-training framework for universal, high-performance representation learning of DAS* — arXiv:2506.04552 — https://arxiv.org/pdf/2506.04552 · *J. Lightwave Technol.* 44(7):2544 (2026) — https://opg.optica.org/jlt/abstract.cfm?uri=jlt-44-7-2544
- *A Foundation Model for DAS Signal Recognition and Visual Prompt Tuning for Downstream Tasks* — arXiv:2508.04316 — https://arxiv.org/pdf/2508.04316
- *A cross-modal pre-training framework with video data for improving DAS performance and generalization* — arXiv:2511.09342 — https://arxiv.org/pdf/2511.09342

**Geophysics / seismology**
- RNN-DAS — Fernández-Carabantes et al., *A New Deep Learning Approach for Detection and Real-Time Monitoring of Volcano-Tectonic Events Using DAS*, *JGR: Solid Earth* (2025) — https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JB031756
- PhaseNet-DAS — Zhu et al., *Nat. Commun.* 14:8192 (2023) — https://doi.org/10.1038/s41467-023-43355-3 · arXiv:2302.08747 — https://arxiv.org/abs/2302.08747
- DAS-N2N — *DAS denoising without clean data*, *Geophys. J. Int.* 236 (2023/2024) — https://academic.oup.com/gji/article/236/2/1026/7453669

**Datasets, reviews & front-ends**
- *Comprehensive Dataset for Event Classification Using DAS Systems*, *Scientific Data* (2025) — https://www.nature.com/articles/s41597-025-05088-4 · PubMed — https://pubmed.ncbi.nlm.nih.gov/40368970/
- *Artificial-intelligence-driven DAS technology and engineering application*, *PhotoniX* (2025) — https://photonix.springeropen.com/articles/10.1186/s43074-025-00160-z
- *DAS for Road Traffic Monitoring: Principles, Signal Processing, and Emerging Applications*, *Infrastructures* 10(9):228 (2025) — https://www.mdpi.com/2412-3811/10/9/228
- *Intelligent Feature Extraction and Event Classification in DAS Using Wavelet Packet Decomposition*, *Technologies* 13(11):514 (2025) — https://www.mdpi.com/2227-7080/13/11/514

**Traffic & infrastructure**
- *Intelligent Traffic Monitoring with DAS* — arXiv:2403.02791 — https://arxiv.org/pdf/2403.02791
- *DAS for Urban Traffic Monitoring: Spatio-Temporal Attention in Recurrent Neural Networks* — arXiv:2603.13903 — https://arxiv.org/pdf/2603.13903
- *Enhancing traffic monitoring with noise-robust DAS and deep learning*, *Measurement* (2024) — https://www.sciencedirect.com/science/article/abs/pii/S092698512400332X

**Security & pipeline**
- *Advancing Perimeter Security: Integrating DAS and CNN for Object Classification in Fiber Vicinity*, *IEEE* (2025) — https://ieeexplore.ieee.org/document/10955273/ · PDF — https://backend.orbit.dtu.dk/ws/files/400052508/Advancing_Perimeter_Security_Integrating_DAS_and_CNN_for_Object_Classification_in_Fiber_Vicinity.pdf
- *Identification and classification of oil-and-gas pipeline intrusion events based on 1-D CNN*, *PLOS ONE* (2025) — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0338205

**Marine / ocean**
- *Automated detection of fin whales with DAS in the Arctic and Mediterranean*, *JASA* 158(6):5057 (2025) — https://pubs.aip.org/asa/jasa/article/158/6/5057/3375686
- *Observations of fin whales and vessels offshore Oregon using fibre-optic DAS*, *Front. Mar. Sci.* (2025) — https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1603541/full
- *Detecting silent whales using seabed fiber-optic cables*, *PNAS* (2026) — https://www.pnas.org/doi/abs/10.1073/pnas.2603077123
- *An End-to-End Workflow for Fin Whale Song Detection, Note Characterization, and Localization with DAS* — arXiv:2608.02387 — https://arxiv.org/abs/2608.02387
- *A deep-learning framework for marine acoustic and seismic monitoring with DAS* — arXiv:2603.14844 — https://arxiv.org/html/2603.14844
- *Distributed acoustic sensing for ocean applications* (survey) — arXiv:2502.18344 — https://arxiv.org/pdf/2502.18344

*Diagrams in this entry are hand-authored standalone SVG (no external URLs), with
explicit light-card / dark-panel fills so they render legibly in both light and
dark viewers. Some links were gathered under scraping/API limits and are provided
best-effort; where a landing page was unreachable, an arXiv or DOI mirror is
listed alongside.*
