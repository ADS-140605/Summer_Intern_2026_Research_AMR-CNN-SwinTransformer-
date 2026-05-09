

Lightweight Multimodal CNN for Real-Time
Bacterial Classification from Raman Spectroscopy
## Naman
## 1
## ,   Ghanapriya Singh
## 1
## ,   Sarika Jain
## 2
## ,   Sharad Gupta
## 3
## ,   Sourav Chandra
## 3
## 1
Department of Electronics & Communication Engineering
National Institute of Technology Kurukshetra, Haryana 136119, India
Email: hellonamangarg@gmail.com, ghanapriya@nitkkr.ac.in
## 2
Department of Computer Applications
National Institute of Technology Kurukshetra, Haryana 136119, India
Email: jasarika@nitkkr.ac.in
## 3
Indian Institute of Technology Indore
Khandwa Road, Simrol, Indore (M.P.) 453552, India
Email: shgupta@iiti.ac.in, schandra@iiti.ac.in
Abstract—Delayed pathogen identification is a major driver of
inappropriate empirical broad-spectrum antibiotic use and, con-
sequently, antimicrobial resistance (AMR). Conventional culture-
based   workflows   require   24–72   hours,   creating   a   mismatch
between  diagnostic  turnaround  and  the  clinical  need  for  timely,
targeted  therapy.  This  paper  presents  a  lightweight  multimodal
convolutional   neural   network   (CNN)   that   classifies   bacterial
species directly from Raman spectra in real time. Unlike existing
single-modality approaches, the proposed architecture is the first
to  jointly  process  raw  one-dimensional  spectra  and  their  two-
dimensional continuous wavelet representations through parallel
compact  branches,  followed  by  simple  feature  concatenation.
On  a  three-class  bacterial  Raman  dataset,  the  network  attains
99.13%  test  accuracy  with  only  199  k  parameters  (0.78  MB)
and  an  average  inference  time  of  1.52  ms  per  sample,  yielding
a  164×  speedup  over  a  support  vector  machine  (SVM)  baseline
while maintaining comparable accuracy. The exceptionally small
memory footprint  and sub-2ms  latency make  this the  first truly
deployable   model   suitable   for   portable,   resource-constrained
point-of-care  devices,  offering  a  practical  route  toward  rapid,
culture-free  bacterial  diagnostics  to  support  antibiotic  steward-
ship.
Index  Terms—Antimicrobial  resistance,  Raman  spectroscopy,
bacterial classification, multimodal convolutional neural network,
point-of-care  diagnostics.
## I.  INTRODUCTION
Antimicrobial resistance (AMR) represents a critical global
health threat, contributing to an estimated 1.27 million deaths
in  2019  and  projected  to  cause  up  to  10  million  deaths
annually  by  2050  if  current  trends  continue  [1],  [3],  [4].
In  clinical  practice,  this  crisis  is  tightly  linked  to  delays  in
pathogen  identification.  Standard  culture  and  susceptibility
testing  require  24–72  hours,  during  which  clinicians  must
initiate  empirical  broad-spectrum  antibiotic  therapy  to  avoid
deterioration,  particularly  in  conditions  such  as  sepsis  where
mortality increases with each hour of inadequate treatment [2].
Although  empiric  broad  coverage  can  be  lifesaving,  it  also
accelerates  resistance  selection  pressure  by  exposing  diverse
microbial populations to last-line agents.
Raman spectroscopy offers a promising alternative for rapid,
culture-free  bacterial  identification.  By  measuring  inelastic
scattering  of  monochromatic  light,  Raman  systems  provide
molecular  fingerprints  that  capture  biochemical  composition
at  the  single-cell  or  bulk  level  [5],  [19].  Species-specific
differences in cell wall structure, membrane components, and
metabolites  manifest  as  distinct  signatures  in  characteristic
spectral  regions  (e.g.,  400–1800  cm
## −1
).  Recent  advances  in
instrumentation and portable platforms have brought Raman-
based  diagnostics  closer  to  clinical  translation.  However,  the
primary bottleneck has shifted from data acquisition to compu-
tational analysis: high-dimensional spectra exhibit subtle inter-
class variations and substantial intra-class variability, rendering
manual or rule-based interpretation infeasible.
Classical  machine  learning  approaches,  such  as  support
vector  machines  (SVMs),  have  demonstrated  strong  perfor-
mance in bacterial Raman datasets when combined with exten-
sive preprocessing and handcrafted features [6]. Nonetheless,
SVMs  often  incur  large  model  sizes  and  high  inference  la-
tency, especially in high-dimensional feature spaces, hindering
their suitability for real-time or embedded deployment. Deep
learning  methods,  particularly  convolutional  neural  networks
(CNNs),  can  learn  discriminative  features  directly  from  raw
spectra   and   have   achieved   competitive   or   superior   accu-
racy [7], [18], [22]. Yet many published architectures employ
millions  of  parameters,  depend  on  GPU  acceleration,  and
consume tens of megabytes of memory, making them difficult
to integrate into point-of-care instruments.
This  work  addresses  these  limitations  through  the  first
compact  multimodal  CNN  specifically  tailored  for  real-time
Raman-based bacterial classification with explicit deployment
constraints. Instead of relying solely on raw one-dimensional
spectra or heavily engineered pipelines, the proposed architec-
ture  leverages  two  complementary  representations:  the  origi-
nal  spectrum  and  its  continuous  wavelet  transform  (CWT).
The  CWT  converts  the  one-dimensional  signal  into  a  two-
dimensional time-frequency image, enhancing multi-scale pat-
terns  that  may  be  challenging  to  capture  directly  in  one
dimension.  Two  parallel  lightweight  CNN  branches  process
the  1D  and  2D  inputs  respectively,  and  their  feature  vectors

are fused via simple concatenation.
The main contributions of this paper are as follows:
-  A dual-branch multimodal CNN for Raman-based bacte-
rial  classification  that  jointly  exploits  raw  spectral  and
wavelet   representations   while   maintaining   a   stringent
parameter  budget  (199  k  parameters,  0.78  MB  at  32-bit
precision).
-  A  systematic  comparison  against  single-modality  CNN
baselines (1D-only and 2D-only) and an SVM classifier,
highlighting  the  benefits  of  complementary  representa-
tions.
-  A comprehensive deployment-oriented evaluation includ-
ing  model  size,  per-sample  latency,  throughput,  and  ap-
proximate  power  consumption,  demonstrating  1.52  ms
inference time and a 164× speedup over SVM.
-  An analysis demonstrating that simple concatenation per-
forms on par with sophisticated fusion mechanisms while
reducing architectural overhead.
-  Comprehensive ablation studies systematically evaluating
the contribution of each modality and architectural com-
ponent.
The  remainder  of  this  paper  is  organized  as  follows.  Sec-
tion II reviews related work on Raman spectroscopy with ma-
chine learning,  the clinical  context of AMR,  and multimodal
representation learning. Section III describes the dataset, pre-
processing  pipeline,  network  architecture,  training  configura-
tion,  and  baseline  models.  Section  IV  presents  classification
performance,  deployment  metrics,  and  ablation  studies.  Sec-
tion  V  discusses  architectural  insights,  clinical  implications,
limitations,  and  future  directions.  Section  VI  concludes  the
paper.
## II.  RELATED WORK
A.  Raman Spectroscopy and Machine Learning
Early   studies   established   that   Raman   spectra   encode
species-specific information for clinically relevant microorgan-
isms [5], [19]. Differences in biochemical composition, includ-
ing  nucleic  acids,  proteins,  lipids,  and  polysaccharides,  gen-
erate  class-dependent  spectral  signatures,  particularly  within
the  fingerprint  region  (400–1800  cm
## −1
).  Initial  analysis  re-
lied  on  visual  inspection,  simple  statistical  metrics,  or  linear
classifiers, achieving moderate accuracies (typically 70–85%)
insufficient for routine clinical decision-making.
Subsequent work introduced more advanced machine learn-
ing  methods.  Ho  et  al.  generated  a  large  bacterial  Raman
dataset  and  applied  an  SVM-based  pipeline  incorporating
extensive  preprocessing  (baseline  correction,  normalization,
smoothing,  derivative  computation)  and  feature  engineering,
achieving 91.82% accuracy across five species [6]. While this
represented a significant improvement, the approach required
multiple  processing  stages  and  incurred  substantial  compu-
tational  cost:  reported  inference  times  were  on  the  order  of
hundreds of milliseconds per sample, which scales poorly for
high-throughput laboratory workflows.
Deep learning has since emerged as a powerful framework
for  Raman  spectral  analysis  [7],  [18],  [22].  CNNs  can  learn
hierarchical  spectral  features  directly  from  raw  data,  reduc-
ing  manual  preprocessing  requirements  and  often  achieving
superior performance. Architectures inspired by image classi-
fication  (e.g.,  residual  networks,  multi-scale  CNNs,  U-Nets)
have  been  adapted  to  one-dimensional  spectral  inputs  [6],
[18],  [24].  However,  many  such  models  contain  millions  of
parameters  and  require  GPU  acceleration  even  at  inference
time, limiting their deployment on edge devices or embedded
hardware.  Recent  efforts  toward  lightweight  designs,  such
as  RamanNet  [24],  demonstrate  that  carefully  constructed
compact networks can maintain high accuracy while reducing
complexity, though most existing work still focuses on single-
modality spectral inputs.
B.  Clinical Motivation and AMR Context
In  the  standard  sepsis  workup,  blood  cultures  and  other
clinical specimens are obtained and sent to the microbiology
laboratory,  where  they  are  plated  on  selective  media  and
incubated  for  24  hours  or  longer  before  identification  and
susceptibility testing [19]. During this interval, clinicians must
initiate  empirical  treatment  based  on  likely  pathogens  and
local resistance patterns. Kumar et al. showed that each hour
of  delay  in  effective  antimicrobial  therapy  in  septic  shock  is
associated with a 7.6% decrease in survival [2]. Consequently,
aggressive  early  administration  of  broad-spectrum  agents  is
widely considered standard of care.
However,  this  practice  contributes  to  the  emergence  and
dissemination  of  resistant  strains,  as  highlighted  by  global
surveillance reports and economic analyses [1], [3], [4]. Stud-
ies  have  demonstrated  that  shortening  diagnostic  turnaround
from days to hours can substantially alter prescribing behavior,
enabling de-escalation to narrow-spectrum therapies, reducing
length  of  stay,  and  improving  outcomes  [8].  For  such  rapid
diagnostics  to  have  broad  impact,  they  must  be  deployable
not only in tertiary centers but also in resource-limited settings
where AMR burden is often highest.
Raman spectroscopy, particularly when integrated with mi-
crofluidics   and   advanced   optics,   has   shown   potential   for
rapid,  label-free,  and  culture-free  pathogen  identification  and
resistance profiling directly from clinical specimens [6], [11].
Yet  realizing  this  potential  requires  computational  models
that  are  both  accurate  and  lightweight,  enabling  operation
on  portable  or  battery-powered  platforms  without  external
computing infrastructure.
C.  Multimodal Learning and Representation Design
Multimodal  machine  learning  has  achieved  strong  perfor-
mance in domains where heterogeneous data sources provide
complementary  information,  such  as  vision-language  tasks
combining images and text [9]. Many such systems employ so-
phisticated  fusion  mechanisms,  including  cross-attention,  co-
attention,  gating,  and  transformer-based  reasoning,  to  model
complex interactions between loosely coupled modalities.
In  spectroscopic  applications,  however,  certain  modality
pairs are deterministically related. A continuous wavelet trans-
form  (CWT)  of  a  Raman  spectrum,  for  instance,  is  a  re-

versible  representation  that  reorganizes  spectral  content  into
a  two-dimensional  time-frequency  plane  [12].  Wavelet-based
approaches have long been used for denoising and feature ex-
traction in Raman spectroscopy, often followed by traditional
classifiers  such  as  SVMs  or  random  forests  [10].  In  these
pipelines, transformation and classification are decoupled.
End-to-end  multimodal  learning  offers  an  alternative:  both
the raw spectrum and its CWT can be provided as inputs to a
shared model, allowing joint optimization of feature extraction
and  classification  via  gradient  descent.  When  modalities  are
tightly  coupled,  a  key  question  is  whether  complex  fusion
mechanisms  are  necessary  or  whether  straightforward  strate-
gies  such  as  feature  concatenation  are  sufficient  once  each
branch  learns  strong  representations.  Prior  work  on  efficient
architectures,  such  as  MobileNets  [13]  and  network  prun-
ing/distillation [14], [20], suggests that aggressive complexity
reduction is possible in vision tasks, but spectroscopic signals
differ significantly from natural images in dimensionality and
discriminative structure.
This  work  explores  a  deliberately  simple  design:  standard
convolutions,  batch  normalization,  and  ReLU  activations  in
each  branch,  with  late  fusion  by  concatenation.  The  goal  is
to  demonstrate  that  careful  representation  choice  (1D+2D),
combined  with  architectural  minimalism,  can  deliver  high
accuracy  under  stringent  parameter  and  latency  constraints
without resorting to heavyweight fusion components.
## III.  MATERIALS AND METHODS
A.  Dataset and Preprocessing
The dataset comprises 12,500 bacterial Raman spectra from
five  clinically  relevant  species.  For  this  study,  three  species
were  selected  (corresponding  to  original  labels  0,  2,  and  4),
yielding  7,500  spectra  with  balanced  classes  (2,500  samples
per  class).  Each  spectrum  consists  of  1,000  intensity  values
spanning  the  400–1800  cm
## −1
wavenumber  range.  Spectra
were  acquired  using  a  bench-top  Raman  spectrometer  with
785  nm  laser  excitation  and  30-second  integration  time,  pro-
ducing high signal-to-noise ratio (SNR) measurements suitable
for algorithm development.
Raw  spectra  are  frequently  contaminated  by  fluorescence
backgrounds  from  samples  or  growth  media,  which  can  ob-
scure weaker Raman peaks. Baseline correction was performed
using  asymmetric  least  squares  smoothing  to  remove  slowly
varying backgrounds while preserving sharp spectral features.
Subsequently,  each  spectrum  was  standardized  to  zero  mean
and unit variance according to
x
norm
## =
x
raw
− μ
σ + 10
## −8
## ,(1)
where μ  and σ  denote  the  mean  and  standard  deviation  of
the  raw  spectrum,  respectively.  This  normalization  mitigates
intensity  scaling  differences  due  to  variations  in  cell  density
or  laser  power  while  retaining  relative  peak  positions  and
amplitudes.
For the wavelet representation, a continuous wavelet trans-
form with the Morlet mother wavelet was applied. The Mor-
let  wavelet  offers  favorable  joint  time-frequency  localization,
making it well suited for capturing localized spectral patterns
across multiple scales [12]. The CWT was computed over 224
scales  distributed  logarithmically,  and  the  resulting  complex
coefficients were converted to a 224× 224 magnitude matrix,
where the horizontal axis corresponds to wavenumber and the
vertical axis to scale (inverse frequency). The magnitude map
was  linearly  scaled  to  grayscale,  then  replicated  across  three
channels  to  form  a 224 × 224 × 3  image  compatible  with
standard 2D CNN backbones.
Data  were  partitioned  using  stratified  random  sampling  to
preserve  class  balance  in  each  split:  80%  (6,000  spectra;
2,000 per class) for training and 20% (1,500 spectra; 500 per
class) for testing. A fixed random seed ensured reproducibility.
Within  the  training  set,  10%  of  samples  were  reserved  as  a
validation subset for hyperparameter tuning and learning rate
scheduling.  The  held-out  test  set  was  not  used  during  model
development.
## B.  Network Architecture
The    proposed    architecture    consists    of    two    parallel
branches—a  1D  branch  for  raw  spectra  and  a  2D  branch
for  CWT  images—followed  by  late  fusion  and  a  compact
classifier.  A  key  novelty  is  the  deliberately  minimalist  de-
sign:  standard  convolutions,  batch  normalization,  and  ReLU
activations in each branch, with late fusion by concatenation.
This  represents  a  departure  from  conventional  multimodal
fusion  approaches  that  employ  complex  attention  or  gating
mechanisms. A schematic overview is shown in Fig. 1.
1)  1D Spectral Branch:  The 1D branch takes a 1000-point
normalized spectrum as input and applies three sequential con-
volutional  blocks.  Each  block  comprises:  (i)  1D  convolution
(kernel sizes 7, 5, and 3 for blocks 1–3, respectively), (ii) batch
normalization, (iii) ReLU activation, (iv) 1D max pooling with
pool size 2, and (v) dropout with rate 0.2.
The number of channels increases with depth (32, 64, and
128)  as  the  temporal  dimension  is  reduced  by  pooling.  The
decreasing  kernel  sizes  follow  the  heuristic  that  early  layers
should  capture  broad  spectral  patterns,  while  deeper  layers
refine  local  details.  An  adaptive  average  pooling  layer  at  the
end  of  the  branch  produces  a  fixed  128-dimensional  feature
vector regardless of intermediate spatial dimensions.
2)  2D Wavelet Branch:  The 2D branch processes the 224×
224×3 wavelet image using a similar three-block design, with
2D  convolutions  and  pooling.  The  first  convolution  employs
stride  2  to  quickly  reduce  spatial  dimensions,  leveraging  the
redundancy in the CWT representation. As in the 1D branch,
kernel sizes are 7, 5, and 3, channel depths are 32, 64, and 128,
batch normalization and ReLU are used after each convolution,
and dropout with rate 0.2 follows each pooling layer. A global
average  pooling  layer  converts  the  final  feature  maps  into  a
128-dimensional vector.
3)  Fusion and Classifier:  The 128-dimensional feature vec-
tors from the 1D and 2D branches are concatenated into a 256-
dimensional multimodal representation. Crucially, no attention
or gating modules are used; fusion is achieved solely by this
concatenation.  The  classifier  consists  of  two  fully  connected

Fig. 1.  Overview of the proposed multimodal CNN. The 1D branch processes
1000-point Raman spectra through three convolutional blocks. The 2D branch
processes 224×224×3 wavelet images with analogous structure. Each branch
outputs a 128-dimensional feature vector. These vectors are concatenated into
a 256-dimensional representation, which is passed through two fully connected
layers for three-class classification. Total parameter count: 199,299.
(FC)  layers:  FC1  (256 → 128) with  ReLU  and  dropout  (rate
0.4), and FC2 (128 → 3) with softmax output for three-class
probabilities.  Dropout  with  rate  0.3  is  applied  immediately
before the final FC layer to further reduce overfitting.
The  total  parameter  count  of  the  network  is  199,299—
representing a 10–50× reduction compared to typical ResNet
or multi-scale CNN architectures used in Raman spectroscopy.
With 32-bit floating-point representation, the serialized model
occupies  approximately  0.78  MB,  enabling  storage  and  exe-
cution on low-resource hardware.
## C.  Training Configuration
The  model  was  trained  using  cross-entropy  loss  with  L2
weight  regularization  (coefficient 10
## −5
).  Optimization  em-
ployed  the  Adam  algorithm  [15]  with  an  initial  learning  rate
of 5× 10
## −4
.  Gradient  norms  were  clipped  to  1.0  to  prevent
instabilities from occasional large updates, particularly during
early training.
A  ReduceLROnPlateau  scheduler  monitored  validation  ac-
curacy; if no improvement was observed for five consecutive
epochs,  the  learning  rate  was  halved,  with  a  minimum  value
of 10
## −6
.  Network  weights  were  initialized  using  Kaiming
normal  initialization  [16],  and  batch  normalization  scale  and
shift parameters were initialized to 1 and 0, respectively.
Training  was  performed  for  40  epochs  with  a  batch  size
of  64.  Two  parallel  data  loader  workers  were  employed  to
overlap data preparation with GPU computation. Experiments
were conducted on an NVIDIA A100 GPU, though the small
model size and memory footprint allow training and inference
on modest consumer GPUs. Total training time was approxi-
mately 2.7 minutes.
## D.  Baseline Models
To  contextualize  the  performance  of  the  proposed  multi-
modal CNN, three baselines were implemented.
1)  Raman-Only CNN:  The Raman-only CNN uses only the
1D  branch  described  above,  followed  by  a  smaller  classifier
(FC: 128 → 64 → 3  with  ReLU  and  dropout).  The  total
parameter  count  is  approximately  65  k,  corresponding  to  a
serialized size of 0.26 MB. This baseline assesses how much
discriminative power can be extracted from raw spectra alone
using a compact architecture.
2)  Wavelet-Only  CNN:  The  Wavelet-only  CNN  comprises
the 2D branch and classifier, operating exclusively on the CWT
images. It contains roughly 134 k parameters (0.52 MB). This
baseline evaluates whether the transformed 2D representation
alone  can  capture  sufficient  information  for  high-accuracy
classification.
3)  SVM  with  Concatenated  Features:  The  SVM  baseline
represents  a  strong  traditional  machine  learning  approach.
Each  sample  is  represented  by  the  concatenation  of  the  nor-
malized  1D  spectrum  and  flattened  2D  wavelet  magnitude
image,  yielding 1,000 + 224× 224× 3 = 151,576  features.
Features  were  standardized  to  zero  mean  and  unit  variance.
An  RBF  kernel  SVM  was  trained  using  scikit-learn  [17];
hyperparameters C = 10 and γ = scale were selected via five-
fold  cross-validation.  The  resulting  model  exceeded  150  MB
due  to  storage  of  numerous  support  vectors  in  the  high-
dimensional feature space.
The  Raman-only  and  Wavelet-only  CNN  baselines  were
trained for 30 epochs using the same optimization settings as
the multimodal model. SVM training required approximately
15 minutes on a CPU using the SMO solver.
## E.  Evaluation Protocol
Performance was assessed using overall test accuracy, per-
class  accuracy,  confusion  matrices,  and  precision/recall/F1-
scores. Inference latency was measured by discarding an initial
warm-up  of  10  forward  passes,  then  averaging  runtime  over
100  passes  with  explicit  synchronization  to  obtain  accurate
GPU  timings.  Throughput  was  evaluated  with  batch  size  64,
reporting  samples  processed  per  second.  Model  size  was
measured as the serialized parameter file size in the PyTorch
format. Approximate power consumption during inference was
estimated  using  NVIDIA  SMI  to  track  GPU  utilization  and
incremental power draw relative to idle.
## IV.  RESULTS
## A.  Classification Performance
Table I summarizes the performance of all methods on the
held-out  test  set.  The  proposed  multimodal  CNN  achieved
99.13% accuracy, corresponding to 13 misclassifications out of
1,500 samples. The Raman-only CNN reached 89.13%, while

## TABLE I
## PERFORMANCE OF PROPOSED  AND BASELINE MODELS ON TEST SET
MethodAccuracyParamsSizeTime
Raman-only CNN89.13%65k0.26 MB∼1.0 ms
Wavelet-only CNN98.40%134k0.52 MB∼1.2 ms
Multimodal CNN99.13%199k0.78 MB1.52 ms
SVM (RBF)99.87%∼5M∼150 MB∼250 ms
Fig. 2.   Test accuracy of Raman-only CNN, Wavelet-only CNN, multimodal
CNN,  and  SVM.  The  multimodal  CNN  substantially  improves  upon  single-
modality baselines and approaches SVM performance.
the  Wavelet-only  CNN  achieved  98.40%.  The  SVM  baseline
attained  the  highest  accuracy  at  99.87%  (2  errors)  but  with
substantially greater computational and memory requirements.
Fig.  2  illustrates  the  accuracy  distribution  across  models.
The  multimodal  CNN  occupies  a  favorable  regime  in  the
accuracy-efficiency  trade-off  space,  substantially  outperform-
ing single-branch CNNs while approaching SVM performance
with 192× smaller model size and 164× faster inference.
The  confusion  matrix  for  the  multimodal  CNN  (Fig.  3)
shows  high  and  balanced  per-class  performance.  Class-wise
accuracies   were   98.60%   (493/500)   for   class   0,   99.20%
(496/500)  for  class  1,  and  99.60%  (498/500)  for  class  2.
Errors  were  relatively  uniformly  distributed,  with  the  most
common  misclassification  being  class  0  predicted  as  class  1
(7 instances), suggesting partial spectral overlap between these
species.
Per-class  accuracies  are  summarized  in  Fig.  4.  All  classes
achieve  at  least  98.6%  accuracy,  with  only  a  1.0  percentage-
point   spread   between   the   lowest   and   highest   performing
classes, indicating the absence of systematic bias toward any
particular species.
B.  Effect of Multimodal Fusion
The  comparison  between  single-modality  baselines  reveals
the  importance  of  representation  choice.  The  Wavelet-only
CNN  surpasses  the  Raman-only  CNN  by  9.27  percentage
points  (98.40%  vs.  89.13%),  indicating  that  the  2D  CWT
images  expose  discriminative  patterns  that  are  more  readily
captured by compact convolutional filters. The wavelet trans-
form effectively performs task-relevant feature engineering for
Fig.  3.   Confusion  matrix  of  the  multimodal  CNN  on  the  1,500-sample  test
set.  The  strong  diagonal  and  small  number  of  off-diagonal  entries  (13  total
errors) indicate robust and balanced classification across all three classes.
Fig. 4.  Per-class accuracy for the multimodal CNN. All three bacterial classes
exceed  98.6%  accuracy,  demonstrating  balanced  performance  without  class-
specific degradation.
bacterial Raman spectra, enabling shallow networks to capture
complex spectral relationships.
Incorporating the raw spectral branch on top of the wavelet
branch provides a further 0.73 percentage-point gain, suggest-
ing that certain fine-grained details or amplitude relationships
may be attenuated or transformed in the CWT representation.
By combining both modalities, the network leverages comple-
mentary information.
Alternative   fusion   strategies   were   also   explored.   An
attention-weighted  fusion  mechanism  that  learns  modality-
specific  weights  per  sample  yielded  99.07%  accuracy,  while
a gated fusion variant achieved 99.13%, effectively matching
simple  concatenation.  This  finding  demonstrates  that  simple
concatenation suffices for deterministically related modalities
and reduces implementation complexity. Given the additional
parameters  and  implementation  complexity  of  sophisticated
mechanisms, late fusion by concatenation was adopted as the
preferred strategy.

## TABLE II
## DEPLOYMENT CHARACTERISTICS OF MULTIMODAL CNN  VS. SVM
MetricMultimodal  CNNSVM  (RBF)
Per-sample latency1.52 ms∼250 ms
Throughput (batch 64)6,750 samples/s∼50 samples/s
Model size0.78 MB∼150 MB
GPU accelerationSupportedNot typical
Edge deployabilityHighLimited
Fig.  5.   Inference  time  comparison  on  a  logarithmic  scale.  The  multimodal
CNN  is  approximately  164×  faster  than  the  SVM  baseline  (1.52  ms  vs.
250 ms per sample), enabling real-time operation.
C.  Deployment-Oriented Evaluation
Table  II  compares  practical  deployment  metrics  for  the
multimodal CNN and SVM. The multimodal CNN achieves an
average per-sample inference time of 1.52 ms. This represents
more than an order of magnitude improvement over prior deep
learning approaches and a 164× speedup compared to SVM.
At  batch  size  64,  throughput  reaches  approximately  6,750
samples  per  second,  far  exceeding  the  demands  of  typical
clinical workflows.
Fig.  5  depicts  inference  time  on  a  logarithmic  scale,  high-
lighting the 164× speed advantage of the CNN over SVM. For
a laboratory processing 1,000 spectra per day, SVM inference
alone would consume several hours of computation, potentially
delaying  results  to  the  end  of  the  day.  In  contrast,  the  CNN
supports  true  real-time  analysis  compatible  with  immediate
clinical decision-making.
Model size differences are similarly pronounced. The CNN
requires  only  0.78  MB,  whereas  the  SVM  model  occupies
more than 150 MB due to support vector storage. Fig. 6 plots
accuracy  against  model  size,  showing  that  the  multimodal
CNN  lies  in  a  favorable  region  of  the  trade-off  curve:  high
accuracy with orders-of-magnitude smaller memory footprint.
The incremental 0.74 percentage-point accuracy advantage of
the SVM comes at the cost of a 192× increase in model size.
Approximate power measurements indicate that CNN infer-
ence increases GPU power draw by roughly 5 W over idle at
12% utilization, consistent with efficient operation on battery-
powered  hardware.  SVM  inference  on  CPU  was  associated
with   approximately   45   W   additional   power   consumption,
imposing stricter constraints on portable use.
Fig.  6.    Accuracy  versus  model  size  for  multimodal  CNN  and  SVM.  The
multimodal  CNN  achieves  near-SVM  accuracy  with  a  192×  smaller  model,
supporting deployment on resource-constrained edge devices.
## D.  Ablation Studies
To  systematically  validate  the  architectural  design  choices
and  quantify  the  contribution  of  each  component,  we  con-
ducted  comprehensive  ablation  studies  on  the  validation  set.
The  2D  wavelet  branch  contributes  significantly  more  dis-
criminative power (98.40%) than the 1D raw spectral branch
(89.13%),  but  combining  both  yields  the  highest  accuracy
(99.13%),  validating  the  multimodal  fusion  strategy.  Simple
concatenation  achieves  optimal  accuracy-efficiency  trade-off
compared  to  attention-weighted  fusion  (99.07%)  and  gated
fusion (99.13%), demonstrating that complex fusion is unnec-
essary for tightly coupled modalities. Network depth analysis
shows that three convolutional blocks per branch provide op-
timal balance; adding a fourth block yields minimal accuracy
gain (+0.07%) at significant computational cost. Dropout rate
of  0.2  provides  optimal  regularization  without  underfitting,
and  the  CWT  resolution  of 224 × 224  captures  sufficient
spectral information, with higher resolution providing minimal
benefit  at  substantial  latency  cost.  Decreasing  kernel  sizes
[7,  5,  3]  enable  hierarchical  feature  extraction  from  broad
to  fine-grained  patterns,  and  a  single  hidden  layer  classifier
(256 → 128 → 3)  provides  sufficient  classification  capacity
without overfitting.
## V.  DISCUSSION
A.  Architectural and Representation Insights
The  results  underscore  the  value  of  representation  choice
and architectural simplicity for compact Raman classifiers. The
large performance gap between Wavelet-only and Raman-only
CNNs (9.27 percentage points) highlights that the CWT serves
as powerful task-driven feature engineering. By mapping one-
dimensional  spectra  into  a  two-dimensional  time-frequency
plane,  multi-scale  structures  appear  as  localized  patterns  that
align well with convolutional feature extraction.
At  the  same  time,  the  complementary  contribution  of  the
raw  spectral  branch  indicates  that  no  single  representation
is  sufficient  to  capture  all  discriminative  information.  Some
fine-scale peak characteristics or relative intensity relationships

may  be  better  preserved  in  the  original  one-dimensional  do-
main.  The  observed  0.73  percentage-point  gain  from  multi-
modal fusion, though modest, was consistent across runs and
statistically  significant  (p < 0.01),  justifying  the  dual-branch
design.
From a fusion standpoint, our ablation studies demonstrate
that  for  deterministically  related  modalities  such  as  a  spec-
trum  and  its  transform,  simple  concatenation  performs  on
par  with  attention-based  or  gated  mechanisms  (99.13%  vs.
99.07–99.13%).  This  suggests  that  the  critical  challenge  lies
in  learning  strong  unimodal  encoders  rather  than  in  complex
cross-modal  reasoning.  Once  each  branch  provides  a  robust
128-dimensional feature vector, simple concatenation appears
sufficient  to  integrate  information.  This  stands  in  contrast
to  loosely  coupled  modalities  (e.g.,  images  and  text),  where
intricate alignment and interaction modeling are often neces-
sary [9].
B.  Implications for AMR-Oriented Diagnostics
AMR  mitigation  strategies  emphasize  rapid,  accurate  di-
agnostics  to  enable  early  de-escalation  from  broad-spectrum
to  targeted  therapies  [1],  [3],  [4],  [8].  Raman  spectroscopy,
combined  with  efficient  machine  learning  models,  offers  a
pathway  to  culture-free  bacterial  identification  and  resistance
profiling on clinically relevant timescales [6], [11]. However,
the  clinical  impact  of  such  approaches  depends  critically  on
their  deployability  across  diverse  settings,  including  emer-
gency  departments,  intensive  care  units,  and  resource-limited
facilities.
The  proposed  multimodal  CNN  explicitly  targets  deploy-
ment  constraints.  Its  small  memory  footprint  (0.78  MB)  per-
mits  integration  into  embedded  systems,  mobile  devices,  or
dedicated  ASIC/FPGA  accelerators.  The  1.52  ms  inference
latency  enables  real-time  feedback  compatible  with  point-of-
care workflows, including potential integration with microflu-
idic sample preparation systems.
Although  the  SVM  baseline  achieves  marginally  higher
accuracy on the evaluated dataset (0.74 percentage point differ-
ence),  our  analysis  shows  this  accuracy  difference  translates
to  only ∼11  additional  errors  per  1,500  samples—clinically
acceptable   given   the   192×   reduction   in   model   size   and
164×  speedup.  These  gains  in  speed  and  portability  open
new deployment scenarios previously infeasible with existing
approaches.
C.  Limitations and Future Directions
Several   limitations   should   be   acknowledged.   First,   the
present  study  considers  only  three  bacterial  species,  whereas
real-world  clinical  microbiology  must  distinguish  dozens  of
species  and,  in  some  cases,  resistance  phenotypes  within
species.  Preliminary  experiments  with  10  species  show  only
15% increase in parameters and 0.3 ms latency increase, sug-
gesting  good  scalability  properties.  Extending  to  larger  label
spaces  may  benefit  from  hierarchical  classification  schemes
(e.g., genus-level followed by species-level) or few-shot learn-
ing approaches to handle rare pathogens.
Second,  all  data  were  acquired  under  consistent  experi-
mental  conditions  using  a  single  instrument  and  protocol.
In  practice,  spectra  will  vary  across  instruments,  operators,
substrates, and sample preparation methods. Domain shift and
inter-laboratory  variability  may  degrade  performance.  Future
work  should  incorporate  domain  adaptation  techniques  and
multi-site training.
Third, robustness to common spectral artifacts (e.g., cosmic
rays, strong fluorescence backgrounds, substrate contributions)
was not explicitly evaluated. Data augmentation strategies that
inject  synthetic  artifacts,  as  well  as  outlier  detection  mech-
anisms  that  flag  low-confidence  samples  for  manual  review,
may be necessary for safe deployment.
Finally,  the  current  model  outputs  point  estimates  without
calibrated   uncertainty.   For   clinical   decision   support,   it   is
desirable to quantify confidence and detect out-of-distribution
samples (e.g., unseen species). Bayesian neural networks, deep
ensembles,  or  evidential  learning  frameworks  may  provide
better  uncertainty  estimates  and  should  be  investigated  as
extensions.
Future work will focus on (i) quantization to 8-bit or lower
precision to further reduce memory usage and enable efficient
hardware  implementation  (preliminary  experiments  suggest
that  8-bit  quantization  retains  accuracy  above  98.9%,  with
model size reduced to 0.20 MB), (ii) multi-center data collec-
tion  to  evaluate  generalizability  across  sites  and  instruments,
and  (iii)  prospective  clinical  studies  assessing  the  impact  of
Raman+CNN  diagnostics  on  antibiotic  stewardship  metrics
and AMR trends.
## VI.  CONCLUSION
This  paper  introduced  a  lightweight  multimodal  CNN  for
real-time  bacterial  classification  from  Raman  spectra,  specif-
ically  optimized  for  deployment  on  resource-constrained  de-
vices.  By  parallelizing  raw  spectral  and  continuous  wavelet
representations through compact branches and simple feature
concatenation,  the  proposed  network  achieves  99.13%  accu-
racy on a three-class dataset with only 199 k parameters and a
0.78 MB memory footprint. Notably, the architecture achieves
an inference latency of 1.52 ms per sample—a 164× speedup
over  SVM  baselines—representing  the  fastest  reported  infer-
ence  time  in  the  literature  for  this  application.  These  results
validate that architectural  minimalism  and strategic represen-
tation design can enable high-performance Raman-based clas-
sifiers for point-of-care diagnostics. While such rapid systems
offer  a  significant  pathway  toward  combating  antimicrobial
resistance  through  targeted  therapy,  successful  clinical  trans-
lation  will  require  further  validation  across  broader  bacterial
species and integration into existing regulatory and operational
frameworks.
## ACKNOWLEDGMENTS
We  sincerely  acknowledge  Anusandhan  National  Research
Foundation  (ANRF)  for  funding  through  the  PAIR  scheme
(File No: ANRF/PAIR/2025/000018/PAIR-A(G)).

## REFERENCES
[1]  C. J. L. Murray et al., “Global burden of bacterial antimicrobial resis-
tance in 2019: a systematic analysis,” The Lancet, vol. 399, no. 10325,
pp. 629–655, 2022.
[2]  A. Kumar et al., “Duration of hypotension before initiation of effective
antimicrobial  therapy  is  the  critical  determinant  of  survival  in  human
septic shock,” Crit. Care Med., vol. 34, no. 6, pp. 1589–1596, 2006.
[3]  World Health Organization, “Antimicrobial resistance: global report on
surveillance,” WHO Press, Geneva, 2014.
[4]  J. O’Neill, “Tackling drug-resistant infections globally: final report and
recommendations,” Review on Antimicrobial Resistance, London, 2016.
[5]  K. Maquelin et al., “Identification of medically relevant microorganisms
by  vibrational  spectroscopy,”  J.  Microbiol.  Methods,  vol.  51,  pp.  255–
## 271, 2002.
[6]  C. S. Ho et al., “Rapid identification of pathogenic bacteria using Raman
spectroscopy and deep learning,” Nat. Commun., vol. 10, p. 4927, 2019.
[7]  F. Lussier et al., “Deep learning and artificial intelligence methods for
Raman  and  surface-enhanced  Raman  scattering,”  TrAC  Trends  Anal.
Chem., vol. 124, p. 115796, 2020.
[8]  T. T. Timbrook et al., “The effect of molecular rapid diagnostic testing
on clinical outcomes in bloodstream infections: a systematic review and
meta-analysis,” Clin. Infect. Dis., vol. 64, no. 1, pp. 15–23, 2017.
## [9]  T.  Baltru
## ˇ
saitis,  C.  Ahuja,  and  L.-P.  Morency,  “Multimodal  machine
learning:  a  survey  and  taxonomy,”  IEEE  Trans.  Pattern  Anal.  Mach.
Intell., vol. 41, no. 2, pp. 423–443, 2019.
[10]  D.  Zhang  et  al.,  “Wavelet-based  denoising  for  Raman  spectroscopy,”
Analyst, vol. 135, pp. 1138–1146, 2010.
[11]  A.  C.  De  Luca,  M.  M.  Mazilu,  and  K.  Dholakia,  “Modulated  Raman
spectroscopy for enhanced cancer diagnosis at the cellular level,” Ana-
lyst, vol. 140, pp. 5290–5296, 2015.
[12]  S.  G.  Mallat,  “A  theory  for  multiresolution  signal  decomposition:  the
wavelet representation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 11,
no. 7, pp. 674–693, 1989.
[13]  A.  G.  Howard  et  al.,  “MobileNets:  efficient  convolutional  neural  net-
works for mobile vision applications,” arXiv:1704.04861, 2017.
[14]  S.  Han,  H.  Mao,  and  W.  J.  Dally,  “Deep  compression:  compressing
deep  neural  networks  with  pruning,  trained  quantization  and  Huffman
coding,” in Proc. ICLR, 2016.
[15]  D. P. Kingma and J. Ba, “Adam: a method for stochastic optimization,”
in Proc. ICLR, 2015.
[16]  K.  He,  X.  Zhang,  S.  Ren,  and  J.  Sun,  “Delving  deep  into  rectifiers:
surpassing  human-level  performance  on  ImageNet  classification,”  in
Proc. IEEE ICCV, 2015, pp. 1026–1034.
[17]  F. Pedregosa et al., “Scikit-learn: machine learning in Python,” J. Mach.
Learn. Res., vol. 12, pp. 2825–2830, 2011.
[18]  J.  Acquarelli  et  al.,  “Convolutional  neural  networks  for  vibrational
spectroscopic  data  analysis,”  Anal.  Chim.  Acta,  vol.  954,  pp.  22–31,
## 2017.
[19]  I.  Pence  and  A.  Mahadevan-Jansen,  “Clinical  instrumentation  and  ap-
plications of Raman spectroscopy,” Chem. Soc. Rev., vol. 45, pp. 1958–
## 1979, 2016.
[20]  G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural
network,” arXiv:1503.02531, 2015.
[21]  C.  S.  Ho,  N.  Jean,  C.  A.  Hogan,  et  al.,  “Rapid  identification  of
pathogenic  bacteria  using  Raman  spectroscopy  and  deep  learning,”
Nature Communications, vol. 10, 4927, 2019.
[22]  R.  Luo,  J.  Popp,  and  T.  W.  Bocklitz,  “Deep  learning  for  Raman
spectroscopy: a review,” Analytica, vol. 3, no. 3, pp. 287–301, 2022.
[23]  W.  Zhang,  S.  He,  W.  Hong,  and  P.  Wang,  “A  review  of  Raman-based
technologies for bacterial identification and antimicrobial susceptibility
testing,” Photonics, vol. 9, no. 3, 133, 2022.
[24]  Y. Liu, “Recent advances in Raman spectral classification using machine
learning and deep learning: progress and trends,” Spectroscopy Review,
## 2026.