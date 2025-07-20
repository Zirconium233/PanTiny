### Overview

Codebase for paper "Rethinking Pan-sharpening: Principled Design, Unified Training, and a Universal Loss Surpass Brute-Force Scaling".

![abstract figure](fig/abstract.png)


### Introduction

This is a pan-sharpening research codebase designed for efficient experimentation and reproducible research. It provides a unified framework for training, evaluating, and comparing different pan-sharpening models across multiple satellite datasets using a unified training paradigm. **One model, three datasets**.

#### Key Features

- **Unified Experiment Framework**: Single configuration file to run multiple experiments with automatic result collection
- **Multi-Dataset Support**: Automatic evaluation on WV2, WV3, and GF2 datasets with both reduced and full-resolution testing
- **Comprehensive Metrics**: Automatic calculation of both reference and no-reference metrics
- **Model Zoo**: Extensive collection of implemented pan-sharpening models including PanNet, PNN, MSDCNN, PanFormer, PanMamba, CFDCNet, and our proposed methods
- **Flexible Configuration**: Hierarchical YAML configuration with inheritance and override support
- **Automatic Reporting**: Detailed experiment reports with performance comparisons and model statistics
- **Reproducible Research**: Automatic config saving, seed control, and deterministic training

#### Codebase Architecture

```
newpan_2411/
├── runner.py              # Main experiment runner
├── solver/
│   ├── unisolver.py      # Universal solver for all experiments
│   └── basesolver.py     # Base solver class
├── model/                # Model implementations
│   ├── panrestormer.py   # Our proposed PanRestormer
│   ├── pannet.py         # PanNet implementation
│   ├── panmamba.py       # PanMamba implementation
│   └── ...               # other models
├── data/
│   └── data.py           # Enhanced data loading with multi-dataset support
├── utils/
│   ├── metrics.py        # Comprehensive metrics calculation
│   ├── loss.py           # Various loss functions
│   └── ...               # Other utilities
└── configs/              # Experiment configurations
    ├── example/
    └── 4T/               # Our experiment configs
```

#### Available Models

The codebase includes implementations of major pan-sharpening methods:

- **Classical Methods**: PNN, PanNet, MSDCNN
- **Attention-Based**: PanFormer with various attention mechanisms
- **State-Space Models**: PanMamba (with Mamba blocks)
- **Research Variants**: M1-M6 (our experimental models with different fusion strategies)

#### Environment Setup

**Quick Start**

We recommend using `uv` for environment management.

If you already have `uv`, simply run:
```bash
uv sync
```

Otherwise, you can install `uv` with one of the following commands:
```bash
conda install uv           # install with conda
pip install uv             # install in your environment
curl -LsSf https://astral.sh/uv/0.7.18/install.sh | sh  # official method, installs at user level
```

If you prefer not to use `uv`, you can manually install the dependencies with conda or pip:
```bash
conda env create -n xxx
conda activate xxx
pip install -r requirements.txt  # Note: PyTorch 2.7+ is not available in conda channels
```

Our codebase does not require any specific version of Python, PyTorch, or CUDA. PyTorch 2.1, 2.6, 2.7 and CUDA 11.8, 12.8 have all been tested. In most cases, your local environment should work without additional setup.

**Additional Dependencies**

If you want to reproduce Pan-Mamba results, the following additional dependencies are required:
```bash
git clone https://github.com/hustvl/Vim.git
cd Vim/causal-conv1d
uv pip install -e . # if you use pip, just remove uv
cd ../mamba-1p1p1
uv pip install -e .
```
We have not compiled Vim's packages with PyTorch 2.7 and CUDA 12.8. If you encounter any issues, we suggest using PyTorch 2.1 and CUDA 11.8, which have been tested. Simply edit the `pyproject.toml` file to switch to the older version.

#### Configuration Usage

We recommend reproducing our results using our codebase. If you use other codebases like [Pan-Mamba](https://github.com/alexhe101/Pan-Mamba.git) or [MSDCNN](https://github.com/alexhe101/MSDDN.git), you may need to concatenate datasets manually and test multiple times for each experiment. In our codebase, we have implemented automatic dataset concatenation and multi-run testing. You can focus on model implementation and hyperparameter tuning by simply editing the config file to run a series of experiments.

#### Configuration Structure

Our codebase uses a hierarchical YAML configuration system that supports both single experiments and batch experiments. The configuration consists of two main parts:

1. **Base Configuration**: Common settings shared across all experiments
2. **Experiments List**: Individual experiment configurations that override base settings

Here's the basic structure (see `configs/example/example.yml` for a complete example):

```yaml
# Base configuration - shared by all experiments
base_config:
  algorithm: "panrestormer"  # Model to use
  nEpochs: 200              # Training epochs
  gpu_mode: true
  save_best: true

  # Multi-dataset support
  data_dirs:
    train:
      WV2: ../data/WV2_data/train128
      WV3: ../data/WV3_data/train128
      GF2: ../data/GF2_data/train128
    eval:
      WV2: ../data/WV2_data/test128
      WV3: ../data/WV3_data/test128
      GF2: ../data/GF2_data/test128

  # Training configuration
  data:
    batch_size: 16 # We use 16 as batch size to ensure most methods do not OOM on 8GB VRAM devices. Based on our experiments, with the same epochs, higher batch size yields better performance.
    patch_size: 32
    n_colors: 4

  data_usage:
    # datasets: ["WV2", "WV3", "GF2"]  # Use all three datasets for training
    datasets: ["WV2"] # Choose which datasets to train on
    usage_percent: 1.0 # You can change data usage percentage for ablation studies
    data_seed: 42
    balance_datasets: false

  # Evaluation configuration
  evaluation:
    full_resolution_in_val: false # If you enable this, training may be slow
    full_resolution_in_test: true
    save_all_test_images: true
    test_models: # Which checkpoints to save (and test)
      latest: true
      bestPSNR: true
      bestSSIM: false
      bestQNR: false
    # Evaluation settings
    eval_interval: 20 # Validate once every 20 training epochs
    save_interval: 20
    use_YCbCr: false

  schedule:
    lr: 4e-5
    optimizer: ADAM

# Individual experiments
experiments:
  - name: "baseline"
    description: "Baseline experiment"
    algorithm: "panrestormer"
    model:
      base_filter: 64
      downsample_levels: 0

  - name: "enhanced"
    description: "Enhanced model with attention"
    model:
      base_filter: 64
      downsample_levels: 2
      fusion_type: "attention"
```


#### Serial Experiments and Override Rules

Our system supports **serial experiments** where multiple experiments are run sequentially. Each experiment inherits the base configuration and can override specific parameters:

1. **Inheritance**: Each experiment starts with the complete `base_config`
2. **Override**: Any parameter specified in an experiment overrides the base value
3. **Deep Merge**: Nested dictionaries are merged recursively (e.g., `model.base_filter` overrides only that specific parameter)

**Override Priority** (highest to lowest):
- Experiment-specific parameters
- Base configuration parameters
- Default values in the code

#### Usage Examples

**Example 1: Model Architecture Comparison**
```yaml
base_config:
  algorithm: "panrestormer"
  nEpochs: 200
  data:
    batch_size: 16

experiments:
  - name: "small_model"
    description: "Small model with 32 filters"
    model:
      base_filter: 32
      downsample_levels: 0

  - name: "medium_model"
    description: "Medium model with 64 filters"
    model:
      base_filter: 64
      downsample_levels: 1

  - name: "large_model"
    description: "Large model with 128 filters"
    model:
      base_filter: 128
      downsample_levels: 2
```

**Example 2: Loss Function Ablation Study**
```yaml
base_config:
  algorithm: "panrestormer"
  model:
    base_filter: 64

experiments:
  - name: "l1_only"
    description: "L1 loss only"
    loss:
      L1:
        enabled: true
        weight: 1.0
      MEF_SSIM:
        enabled: false
      focal:
        enabled: false

  - name: "l1_ssim"
    description: "L1 + SSIM loss"
    loss:
      L1:
        enabled: true
        weight: 0.8
      MEF_SSIM:
        enabled: true
        weight: 0.5
      focal:
        enabled: false

  - name: "l1_ssim_focal"
    description: "L1 + SSIM + Focal loss"
    loss:
      L1:
        enabled: true
        weight: 0.8
      MEF_SSIM:
        enabled: true
        weight: 0.5
      focal:
        enabled: true
        weight: 0.4
```

#### Running Experiments

To run your experiments:
```bash
python runner.py --config configs/example/example.yml
```

Or use a specific config file:
```bash
python runner.py configs/experiment17.yml
```

#### Results and Metrics

After running experiments, comprehensive results and metrics are automatically calculated and saved. You don't need to run `test.py` specifically—results are automatically saved in the experiment directory. However, if something goes wrong, you can run `test.py` to test the models again; the results and metrics will be printed but not saved.

```bash
python test.py ../Out/experiment20_loss_deep_exploration_1751443087/
```

#### Metrics Calculated

Our codebase automatically calculates comprehensive metrics for pan-sharpening evaluation:

**Reference Metrics** (with ground truth, on reduced-resolution datasets):
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better)
- **CC**: Correlation Coefficient (higher is better)
- **SAM**: Spectral Angle Mapper (lower is better)
- **ERGAS**: Erreur Relative Globale Adimensionnelle de Synthèse (lower is better)

**No-Reference Metrics** (without ground truth, on full-resolution datasets):
- **D_λ**: Spectral distortion index (lower is better)
- **D_s**: Spatial distortion index (lower is better)
- **QNR**: Quality with No Reference (higher is better, QNR = (1-D_λ)(1-D_s))

#### Result Table

For each experiment, you should get a table like the following:
```
================================================================================
COMPREHENSIVE METRICS RESULTS
================================================================================

REFERENCE METRICS (with ground truth):
------------------------------------------------------------
Dataset         PSNR      SSIM        CC       SAM     ERGAS
------------------------------------------------------------
WV2          36.7224    0.9320    0.9434    0.0408    1.5486
WV3          21.4286    0.5737    0.7376    0.1369    9.4398
GF2          34.7042    0.9265    0.8162    0.0368    1.8020

NO-REFERENCE METRICS (full resolution):
------------------------------------------------------------
Dataset     D_lambda       D_s       QNR
------------------------------------------------------------
WV2           0.0000    0.0000    0.0000
WV3           0.0000    0.0000    0.0000
GF2           0.0000    0.0000    0.0000
```
If some metrics are not applicable (e.g., no full-resolution ground truth available, not enabled, or the method reported an error), they will be displayed as zero.

You will also receive a comprehensive report like the following:
```
====================================================================================================================================================================================
Model                     Algorithm    Params(K)    FLOPs(M)     Size(MB)   Time(h)  WV2_PSNR   WV2_SSIM   WV2_SAM    WV3_PSNR   WV3_SSIM   WV3_SAM    GF2_PSNR   GF2_SSIM   GF2_SAM
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
panflow_orig(16bs)        panflow      87.3         2861.37      0.33       3.29     41.5541    0.9677     0.0234     30.2634    0.9179     0.0775     47.7515    0.9869     0.0106
panflow_bs165eclp4augg    panflow      87.3         2861.37      0.33       4.73     40.7046    0.9629     0.0249     29.8801    0.9127     0.0811     46.5685    0.9834     0.0123
panflow_bs161eclp04       panflow      87.3         2861.37      0.33       4.58     41.1635    0.9654     0.0243     30.0869    0.9153     0.0798     47.0548    0.9850     0.0114
panflow_bs45eclp4         panflow      87.3         2861.37      0.33       9.76     41.4867    0.9673     0.0235     30.1712    0.9171     0.0785     47.2323    0.9854     0.0111
panrestormer_2ds          panrestormer nan          OOM          nan        nan        nan        nan        nan        nan        nan        nan        nan        nan        nan
====================================================================================================================================================================================
```

#### Results Directory Structure

The codebase organizes results in a hierarchical directory structure:

```
../Out/big_exp_name/
├── experiment_name_{timestamp}/           # Experiment name from config
│   └── timestamp/            # Unix timestamp when experiment started
│       ├── checkpoints/      # Model checkpoints
│       │   ├── bestPSNR.pth # Best model by PSNR
│       │   ├── bestSSIM.pth # Best model by SSIM
│       │   └── latest.pth   # Latest checkpoint
│       ├── logs/            # Training logs and tensorboard files
│       │   └── events.out.tfevents.*
│       └── results/         # Experiment results and reports
│           ├── config.yml   # Copy of experiment configuration
│           ├── test_results/ # Test images (if save_images enabled)
│           └── comprehensive_report.txt # Detailed results report
```
If you run a series of experiments, the results will be organized as follows:

```
../Out/
└── experiment_name_timestamp/           # Main experiment batch directory
    ├── comprehensive_report.txt         # Overall comparison report
    ├── bug_report.txt                  # Error logs for failed experiments
    ├── experiment1/                    # Individual experiment 1
    │   └── timestamp/                  # Experiment timestamp
    │       ├── checkpoints/            # Model checkpoints
    │       │   ├── bestPSNR.pth       # Best model by PSNR
    │       │   ├── bestSSIM.pth       # Best model by SSIM
    │       │   └── latest.pth         # Latest checkpoint
    │       ├── logs/                  # Training logs
    │       │   └── events.out.tfevents.*
    │       └── results/               # Individual results
    │           ├── config.yml         # Experiment config
    │           └── test_results/      # Test images
    ├── experiment2/                   # Individual experiment 2
    │   └── timestamp/
    │       ├── checkpoints/
    │       ├── logs/
    │       └── results/
    └── experiment3/                   # Individual experiment 3
        └── timestamp/
            ├── checkpoints/
            ├── logs/
            └── results/
```

The `comprehensive_report.txt` contains a detailed comparison table like this:

```
COMPREHENSIVE EXPERIMENT REPORT
================================================================================
Generated at: 2024-07-02 15:30:45
Total experiment time: 4.25 hours

EXPERIMENT SUMMARY:
Total experiments: 3
Successful: 3
Failed: 0

MAIN COMPARISON TABLE:
====================================================================================================================================================================================
Model                     Algorithm    Params(K)    FLOPs(M)     Size(MB)   Time(h)  WV2_PSNR   WV2_SSIM   WV2_SAM    WV3_PSNR   WV3_SSIM   WV3_SAM    GF2_PSNR   GF2_SSIM   GF2_SAM
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
baseline                  panrestormer 168.6        501.36       0.64       1.20     32.4567    0.9234     3.2145     31.8934    0.9156     3.4567     33.1234    0.9345     2.9876
enhanced_attention        panrestormer 174.2        523.45       0.66       1.35     33.2145    0.9312     3.0234     32.5678    0.9234     3.2345     33.8765    0.9423     2.8234
large_model               panrestormer 674.5        2005.44      2.57       2.45     34.1234    0.9456     2.8765     33.6789    0.9378     3.1234     35.0123    0.9567     2.6789
====================================================================================================================================================================================

INDIVIDUAL EXPERIMENT DETAILS:
--------------------------------------------------

Experiment: baseline
Description: Baseline model with standard configuration
Algorithm: panrestormer
Success: True
Training time: 1.20 hours
Results directory: ../Out/loss_ablation_1672934567/baseline/1672934567
Parameters: 168624
FLOPs: 501360000
------------------------------

Experiment: enhanced_attention
Description: Enhanced model with attention fusion
Algorithm: panrestormer
Success: True
Training time: 1.35 hours
Results directory: ../Out/loss_ablation_1672934567/enhanced_attention/1672934568
Parameters: 174234
FLOPs: 523450000
------------------------------
```

#### Multi-Dataset Evaluation

Our codebase supports automatic evaluation on multiple datasets:

- **WV2**: WorldView-2 satellite data (8 spectral bands → 4 bands for pan-sharpening)
- **WV3**: WorldView-3 satellite data (8 spectral bands → 4 bands for pan-sharpening)
- **GF2**: GaoFen-2 satellite data (4 spectral bands)

Each experiment is evaluated on all configured datasets, and both individual and average metrics are reported.

#### Accessing Results

1. **Console Output**: Real-time metrics during training and final results summary
2. **TensorBoard**: Training curves and validation metrics (`tensorboard --logdir ../Out/experiment_name/timestamp/logs`)
3. **Comprehensive Report**: Detailed text report with all metrics and comparisons
4. **Config Copy**: Exact configuration used for reproducibility
5. **Model Checkpoints**: Best models saved for inference and further analysis

### Running Our Experiments

#### Model Design

**Main Results Table**

This is the main comparison table presented in our paper:

| Model             | Algorithm    | Params(K) | FLOPs(M)  | Size(MB) | Time(h) | WV2_PSNR | WV2_SSIM | WV2_SAM  | WV3_PSNR | WV3_SSIM | WV3_SAM  | GF2_PSNR | GF2_SSIM | GF2_SAM  |
| :---------------- | :----------- | :-------- | :-------- | :------- | :------ | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- |
| pnn_original      | pnn          | 68.9      | 2257.72   | 0.26     | 3.03    | 39.8195  | 0.9540   | 0.0282   | 29.4904  | 0.9005   | 0.0861   | 43.1433  | 0.9667   | 0.0178   |
| pannet_original   | pannet       | 80.3      | 2632.45   | 0.31     | 4.65    | 38.9757  | 0.9468   | 0.0301   | 29.1169  | 0.8927   | 0.0935   | 43.2604  | 0.9668   | 0.0176   |
| msdcnn_original   | msdcnn       | 239.0     | 7831.68   | 0.91     | 6.17    | 40.3074  | 0.9580   | 0.0267   | 29.6261  | 0.9033   | 0.0833   | 43.2111  | 0.9671   | 0.0176   |
| panflow_original  | panflow      | 87.3      | 2861.37   | 0.33     | 5.63    | 41.1051  | 0.9645   | 0.0243   | 30.0416  | 0.9106   | 0.0799   | 46.3574  | 0.9825   | 0.0125   |
| pscinn_orig       | pscinn       | 3321.5    | 108839.04 | 12.67    | 7.75    | 35.5955  | 0.8967   | 0.0336   | 22.6138  | 0.5538   | 0.1115   | 42.6892  | 0.9616   | 0.0181   |
| sfdi_original     | sdfi         | 49.1      | 1609.47   | 0.19     | 3.94    | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      |
| panmamba_orig     | panmamba     | 488.8     | 16018.44  | 1.86     | 23.49   | 41.3900  | 0.9663   | 0.0236   | 30.1715  | 0.9174   | 0.0779   | 43.9777  | 0.9725   | 0.0164   |
| cfdcnet_orig      | cfdcnet   | 1700.8    | 55730.21  | 6.49     | 8.77    | 41.5437  | 0.9667   | 0.0233   | 30.4175  | 0.9155   | 0.0775   | 47.7564  | 0.9866   | 0.0107   |
| ours(small) | panrestormer | 48.3      | 1581.68   | 0.18     | 4.17    | 41.6173  | 0.9685   | 0.0230   | 30.3823  | 0.9216   | 0.0768   | 48.1624  | 0.9884   | 0.0099   |
| ours(big) | pantiny      | 81.7      | 2678.59   | 0.31     | 3.95    | 41.8455  | 0.9696   | 0.0224   | 30.5902  | 0.9238   | 0.0749   | 48.6119  | 0.9894   | 0.0095   |

The data comes from `reproduction.yml` and `model_size_abl.yml`. SFDI encounters `nan` problems when training on three datasets for 500 epochs, which can be fixed by training for 200 epochs or enabling only one dataset. PSCINN is very unstable when trained on three datasets. When data transitions from WV2 to WV3, PSNR sometimes suddenly jumps to negative values, causing the loss to become extreme and resulting in the same `nan` problem as SFDI. The results above represent the highest scores we could achieve.

**Unified Loss Benchmark**

Models with different loss functions are not directly comparable, and these results can only demonstrate that our approach performs better than other methods. To obtain a more comprehensive comparison that proves the superiority of our model architecture, we prepared `model_abl.yml`. The results are shown below:
| Model             | Algorithm    | Params(K) | FLOPs(M)  | Size(MB) | Time(h) | WV2_PSNR  | WV2_SSIM | WV2_SAM  | WV3_PSNR  | WV3_SSIM | WV3_SAM  | GF2_PSNR  | GF2_SSIM | GF2_SAM  |
| :---------------- | :----------- | :-------- | :-------- | :------- | :------ | :-------- | :------- | :------- | :-------- | :------- | :------- | :-------- | :------- | :------- |
| pnn_ourloss       | pnn          | 68.9      | 2257.72   | 0.26     | 13.77   | 40.8362   | 0.9635   | 0.0256   | 29.8171   | 0.9128   | 0.0834   | 43.3978   | 0.9688   | 0.0173   |
| pannet_ourloss    | pannet       | 80.3      | 2632.45   | 0.31     | 3.06    | 40.7938   | 0.9620   | 0.0256   | 29.8205   | 0.9106   | 0.0841   | 43.7551   | 0.9705   | 0.0167   |
| msdcnn_ourloss    | msdcnn       | 239.0     | 7831.68   | 0.91     | 3.47    | 41.4559   | 0.9669   | 0.0236   | 30.1812   | 0.9189   | 0.0786   | 44.0943   | 0.9730   | 0.0161   |
| panflow_ourloss   | panflow      | 87.3      | 2861.37   | 0.33     | 5.53    | 41.6762   | 0.9688   | 0.0229   | 30.2370   | 0.9197   | 0.0785   | 47.4877   | 0.9865   | 0.0109   |
| pscinn_ourloss    | pscinn       | 3321.5    | 108839.04 | 12.67    | 17.93   | -267.6816 | 0.0000   | 1.4138   | -279.4819 | 0.0000   | 1.4124   | -261.0171 | 0.0000   | 1.4125   |
| sdfi_ourloss      | sdfi         | 49.1      | 1609.47   | 0.19     | 4.31    | nan       | nan      | nan      | nan       | nan      | nan      | nan       | nan      | nan      |
| panmamba_ourloss     | panmamba     | 488.8     | 16018.44  | 1.86     | 28.23   | 41.7675   | 0.9691   | 0.0226   | 30.3643   | 0.9215   | 0.0769   | 45.8430   | 0.9811   | 0.0134   |
| cfdcnet_ourloss     | cfdcnet | 1700.8    | 55730.21  | 6.49     | 8.21    | 42.5020   | 0.9729   | 0.0205   | 31.1123   | 0.9298   | 0.0707   | 49.0722   | 0.9903   | 0.0091   |
| deeppnn_large     | deeppnn      | 271.1     | 8882.09   | 1.03     | 6.75    | 41.8946   | 0.9700   | 0.0224   | 30.4300   | 0.9228   | 0.0759   | 47.4505   | 0.9869   | 0.0109   |
| resatten_large    | resatten     | 263.0     | 8616.41   | 1.00     | 3.39    | 41.9658   | 0.9702   | 0.0222   | 30.3966   | 0.9223   | 0.0759   | 47.1377   | 0.9856   | 0.0113   |
| ours(small) | panrestormer | 48.3      | 1581.68   | 0.18     | 4.17    | 41.6173   | 0.9685   | 0.0230   | 30.3823   | 0.9216   | 0.0768   | 48.1624   | 0.9884   | 0.0099   |
| ours(big) | pantiny      | 81.7      | 2678.59   | 0.31     | 3.95    | 41.8455   | 0.9696   | 0.0224   | 30.5902   | 0.9238   | 0.0749   | 48.6119   | 0.9894   | 0.0095   |

This time, all models are trained on the same datasets with the same loss function. The results show that our model still performs better than other methods. Only the newest and largest models can achieve better performance than ours (when compared to our large and huge models, the performance gap is not significant). The 4 methods at the bottom of the table are designed or modified by us. Most methods perform better than their original versions, which indicates that our loss design is also effective.

DeepPNN is an enlarged and deepened version of PNN, which proves that better performance can be achieved simply by enlarging the model. At the same time, it also reflects the superior performance efficiency of our small model.

ResAtten is a ResNet with channel attention layers. The pipeline is very simple and similar to our work, representing an early attempt of ours. However, it still requires a large number of parameters to achieve good performance. This is why we switched to a Restormer-like structure with a shared encoder to reduce parameters.

**Multi-Dataset Training**

| Model        | Dataset Type | WV2_PSNR | WV2_SSIM | WV2_SAM | WV3_PSNR | WV3_SSIM | WV3_SAM | GF2_PSNR | GF2_SSIM | GF2_SAM |
| :----------- | :----------- | :------- | :------- | :------ | :------- | :------- | :------ | :------- | :------- | :------ |
| Pan-Mamba    | all          | 41.3900  | 0.9663   | 0.0236  | 30.1715  | 0.9174   | 0.0779  | 43.9777  | 0.9725   | 0.0164  |
| Pan-Mamba    | separate     | 42.2354  | 0.9729   | 0.0212  | 31.1551  | 0.9299   | 0.0702  | 47.6453  | 0.9894   | 0.0103  |
| CFDCNet      | all          | 41.5437  | 0.9667   | 0.0233  | 30.4175  | 0.9155   | 0.0775  | 47.7564  | 0.9866   | 0.0107  |
| CFDCNet      | separate     | 42.2406  | 0.9733   | 0.0209  | 31.2386  | 0.9327   | 0.0694  | 47.8423  | 0.9902   | 0.0097  |
| PNN          | all          | 39.8195  | 0.9540   | 0.0282  | 29.4904  | 0.9005   | 0.0861  | 43.1433  | 0.9667   | 0.0178  |
| PNN          | separate     | 40.7550  | 0.9624   | 0.0259  | 29.9418  | 0.9121   | 0.0824  | 43.1208  | 0.9704   | 0.0172  |
| PanNet       | all          | 38.9757  | 0.9468   | 0.0301  | 29.1169  | 0.8927   | 0.0935  | 43.2604  | 0.9668   | 0.0176  |
| PanNet       | separate     | 40.8176  | 0.9626   | 0.0257  | 29.6840  | 0.9072   | 0.0851  | 43.0659  | 0.9685   | 0.0178  |
| MSDCNN       | all          | 40.3074  | 0.9580   | 0.0267  | 29.6261  | 0.9033   | 0.0833  | 43.2111  | 0.9671   | 0.0176  |
| MSDCNN       | separate     | 41.3355  | 0.9664   | 0.0242  | 30.3038  | 0.9184   | 0.0782  | 45.6847  | 0.9827   | 0.0135  |
| PanFlow      | all          | 41.1051  | 0.9645   | 0.0243  | 30.0416  | 0.9106   | 0.0799  | 46.3574  | 0.9825   | 0.0125  |
| PanFlow      | separate     | 41.8584  | 0.9712   | 0.0224  | 30.4873  | 0.9221   | 0.0751  | 47.2533  | 0.9884   | 0.0103  |
| PSCINN       | all          | 35.5955  | 0.8967   | 0.0336  | 22.6138  | 0.5538   | 0.1115  | 42.6892  | 0.9616   | 0.0181  |
| PSCINN       | separate     | 41.8520  | 0.9703   | 0.0223  | 30.5599  | 0.9230   | 0.0748  | 47.1100  | 0.9878   | 0.0107  |
| SFDI         | all          | nan      | nan      | nan     | nan      | nan      | nan     | nan      | nan      | nan     |
| SFDI         | separate     | 41.7244  | 0.9725   | 0.0220  | 30.5971  | 0.9236   | 0.0741  | 47.4712  | 0.9901   | 0.0102  |
| Pan-LUT      | separate     | 39.8362  | 0.9555   | 0.0286  | 28.8213  | 0.8936   | 0.0935  | 42.6559  | 0.9642   | 0.0189  |
| Ours         | all          | 41.8455  | 0.9696   | 0.0224  | 30.5902  | 0.9238   | 0.0749  | 48.6119  | 0.9894   | 0.0095  |
| Ours         | separate     | 42.1628  | 0.9711   | 0.0217  | 30.6142  | 0.9245   | 0.0747  | 48.9268  | 0.9900   | 0.0092  |

The table above compares unified model experiments with the original training methods reported in the original papers. We can observe that some SOTA methods experience significant performance drops, such as PSCINN and SFDI, which cannot train stably in our codebase. However, some SOTA methods still perform well, such as PanFlow and PanMamba. Pan-LUT did not release their code, so we cannot reproduce their results in our codebase. We find that on WorldView datasets, most methods drop about 1 PSNR, but on the GaoFen-2 dataset, PanFlow performs best, achieving 46 PSNR with only about 1 PSNR drop. Other methods drop about 2-3 PSNR. However, our method only drops about 0.3 PSNR when switching to all-in-one training.


**Downsampling Levels**

| Model              | Algorithm    | Params(K) | FLOPs(M) | Size(MB) | Time(h) | WV2_PSNR | WV2_SSIM | WV2_SAM  | WV3_PSNR | WV3_SSIM | WV3_SAM  | GF2_PSNR | GF2_SSIM | GF2_SAM  |
| :----------------- | :----------- | :-------- | :------- | :------- | :------ | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- |
| panrestormer_4ds   | panrestormer | 446.7     | 14639.01 | 1.70     | 3.65    | 39.5922  | 0.9543   | 0.0287   | 28.9258  | 0.8926   | 0.0977   | 45.3323  | 0.9791   | 0.0140   |
| panrestormer_2ds   | panrestormer | 121.2     | 3970.99  | 0.46     | 9.96    | 40.7399  | 0.9627   | 0.0255   | 29.5754  | 0.9079   | 0.0856   | 46.7448  | 0.9840   | 0.0118   |
| panrestormer_0ds   | panrestormer | 48.0      | 1572.47  | 0.18     | 10.41   | 40.5795  | 0.9618   | 0.0257   | 29.5796  | 0.9083   | 0.0849   | 46.6407  | 0.9839   | 0.0118   |

The performance in the table above is not high because we used simple L1 loss and the fusion layer is just 1×1 convolution. This experiment demonstrates that the more downsampling levels used, the larger the parameters become, yet the worse the performance gets. Therefore, we finally chose zero downsampling levels, which corresponds to our small model in the model ablation table—the smallest model we have seen in the pan-sharpening field. However, this performance does not achieve SOTA results.

Inspired by Pan-Mamba, we experimented with a larger encoder and fusion module. The parameters are slightly larger but the improvement is worthwhile. Compared to previous SOTA methods like Pan-Mamba and the efficient PanFlow, our model achieves SOTA performance with fewer parameters and FLOPs. (Note: PanFlow is not a one-step model, so the FLOPs may not be accurate. Other papers report the true FLOPs as 5710M, which is the data used in our paper.)

| Model           | Algorithm | Params(K) | FLOPs(M) | Size(MB) | Time(h) | WV2_PSNR | WV2_SSIM | WV2_SAM | WV3_PSNR | WV3_SSIM | WV3_SAM | GF2_PSNR | GF2_SSIM | GF2_SAM |
| :-------------- | :-------- | :-------- | :------- | :------- | :------ | :------- | :------- | :------ | :------- | :------- | :------ | :------- | :------- | :------ |
| ours_chosen    | pantiny      | 81.7      | 2678.59   | 0.31     | 3.95    | 41.8455   | 0.9696   | 0.0224   | 30.5902   | 0.9238   | 0.0749   | 48.6119   | 0.9894   | 0.0095   |
| ours_large_body | pantiny   | 172.4     | 5647.63  | 0.66     | 9.82    | 42.0648  | 0.9708   | 0.0219  | 30.6710  | 0.9248   | 0.0742  | 48.7485  | 0.9896   | 0.0094  |
| ours_huge      | pantiny   | 195.9     | 6419.64  | 0.75     | 10.11   | 42.1236  | 0.9711   | 0.0218  | 30.7382  | 0.9258   | 0.0736  | 48.8507  | 0.9898   | 0.0093  |


| Model                     | Algorithm | Params(K) | FLOPs(M) | Size(MB) | Time(h) | WV2_PSNR | WV2_SSIM | WV2_SAM | WV3_PSNR | WV3_SSIM | WV3_SAM | GF2_PSNR | GF2_SSIM | GF2_SAM |
| :------------------------ | :-------- | :-------- | :------- | :------- | :------ | :------- | :------- | :------ | :------- | :------- | :------ | :------- | :------- | :------ |
| pantiny_conv_1x1          | pantiny   | 68.2      | 2234.52  | 0.26     | 5.81    | 41.7524  | 0.9690   | 0.0227  | 30.4518  | 0.9222   | 0.0761  | 48.3679  | 0.9888   | 0.0098  |
| pantiny_channel_attention | pantiny   | 71.4      | 2338.26  | 0.27     | 5.81    | 41.7150  | 0.9686   | 0.0228  | 30.4404  | 0.9216   | 0.0767  | 48.3406  | 0.9886   | 0.0097  |
| pantiny_gated_conv        | pantiny   | 70.3      | 2303.33  | 0.27     | 3.92    | 41.6551  | 0.9686   | 0.0229  | 30.4417  | 0.9219   | 0.0766  | 48.3242  | 0.9886   | 0.0098  |
| pantiny_deepfusion_5      | pantiny   | 113.6     | 3722.12  | 0.43     | 5.05    | 41.6579  | 0.9684   | 0.0229  | 30.3410  | 0.9206   | 0.0771  | 48.3468  | 0.9887   | 0.0098  |
| pantiny_enhanced_conv     | pantiny   | 81.7      | 2678.59  | 0.31     | 3.95    | 41.8455  | 0.9696   | 0.0224  | 30.5902  | 0.9238   | 0.0749  | 48.6119  | 0.9894   | 0.0095  |

These two tables come from `model_size_abl.yml` and `fusion_abl.yml`. The `fusion_abl.yml` experiment shows that 10+K parameters in the fusion module are sufficient, and convolution performs better than many other methods (they cannot even beat 1×1 convolution). The `model_size_abl.yml` is designed to prove that continuously enlarging the model can achieve higher performance and reach SOTA levels (note the WV2 PSNR reaching 42). The performance of the large version is better, but we did not choose it because we consider this approach to be brute-force scaling. Another reason is that the previous version can be trained on 6GB VRAM devices.


| Model                    | Algorithm | Params(K) | FLOPs(M) | Size(MB) | Time(h) | WV2_PSNR | WV2_SSIM | WV2_SAM  | WV3_PSNR | WV3_SSIM | WV3_SAM  | GF2_PSNR | GF2_SSIM | GF2_SAM  |
| :----------------------- | :-------- | :-------- | :------- | :------- | :------ | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- |
| refine_conv_ours         | pantiny   | 81.7      | 2678.59  | 0.31     | 3.97    | 41.8947  | 0.9697   | 0.0224   | 30.6112  | 0.9240   | 0.0749   | 48.4884  | 0.9891   | 0.0097   |
| refine_channel_attention | pantiny   | 96.4      | 3157.75  | 0.37     | 4.03    | 41.8986  | 0.9698   | 0.0223   | 30.5539  | 0.9230   | 0.0751   | 48.4961  | 0.9891   | 0.0096   |
| refine_large_conv        | pantiny   | 88.8      | 2909.80  | 0.34     | 3.95    | 41.8697  | 0.9696   | 0.0224   | 30.4858  | 0.9225   | 0.0759   | 48.5185  | 0.9891   | 0.0096   |

This table concerns the refinement block. We find that a simple convolution is the best choice. The improvements from other designs, we believe, come from enlarging the model, and sometimes performance even drops.

#### Loss Function Analysis

| Model                    | Algorithm | Params(K) | FLOPs(M) | Size(MB) | Time(h) | WV2_PSNR | WV2_SSIM | WV2_SAM  | WV3_PSNR | WV3_SSIM | WV3_SAM  | GF2_PSNR | GF2_SSIM | GF2_SAM  |
| :----------------------- | :-------- | :-------- | :------- | :------- | :------ | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- |
| loss_l1_only_L1(1.0)     | pantiny   | 70.3      | 2303.33  | 0.27     | 6.27    | 39.7720  | 0.9532   | 0.0285   | 29.1879  | 0.8939   | 0.0953   | 45.4224  | 0.9782   | 0.0141   |
| loss_ssim_only_SSIM(1.0) | pantiny   | 70.3      | 2303.33  | 0.27     | 5.75    | 40.8163  | 0.9648   | 0.0254   | 29.9138  | 0.9158   | 0.0816   | 47.2131  | 0.9865   | 0.0111   |
| loss_focal_only_foc(1.0) | pantiny   | 70.3      | 2303.33  | 0.27     | 4.12    | 39.8677  | 0.9545   | 0.0281   | 29.1835  | 0.8937   | 0.0933   | 44.7996  | 0.9757   | 0.0150   |
| loss_0p8_0p5_0p4         | pantiny   | 70.3      | 2303.33  | 0.27     | 4.22    | 40.9993  | 0.9640   | 0.0248   | 29.9877  | 0.9128   | 0.0832   | 47.3759  | 0.9859   | 0.0110   |
| loss_1_1_1               | pantiny   | 70.3      | 2303.33  | 0.27     | 4.77    | 41.2801  | 0.9659   | 0.0240   | 30.1695  | 0.9170   | 0.0791   | 47.6783  | 0.9869   | 0.0105   |
| loss_3_3_3               | pantiny   | 70.3      | 2303.33  | 0.27     | 7.68    | 41.6821  | 0.9683   | 0.0229   | 30.4851  | 0.9214   | 0.0762   | 48.2967  | 0.9885   | 0.0099   |
| loss_1_3_1               | pantiny   | 70.3      | 2303.33  | 0.27     | 7.95    | 41.5723  | 0.9680   | 0.0232   | 30.3688  | 0.9213   | 0.0771   | 48.1416  | 0.9882   | 0.0100   |
| loss_3_1_1               | pantiny   | 70.3      | 2303.33  | 0.27     | 7.50    | 41.3829  | 0.9663   | 0.0237   | 30.3147  | 0.9186   | 0.0777   | 47.9921  | 0.9877   | 0.0102   |
| loss_1_1_3               | pantiny   | 70.3      | 2303.33  | 0.27     | 12.21   | 41.4104  | 0.9665   | 0.0236   | 30.2799  | 0.9177   | 0.0777   | 47.9169  | 0.9875   | 0.0102   |
| L1(2.0)+SSI(2.0)+foc(2.0)| panrestormer | 48.3      | 1581.68  | 0.18     | 4.83    | 41.5967  | 0.9680   | 0.0231   | 30.3834  | 0.9206   | 0.0768   | 48.1690  | 0.9884   | 0.0099   |
| L1(3.0)+SSI(0.8)+foc(1.0)| panrestormer | 48.3      | 1581.68  | 0.18     | 4.82    | 41.3192  | 0.9661   | 0.0237   | 30.2979  | 0.9180   | 0.0777   | 47.9532  | 0.9876   | 0.0102   |
| L1(0.8)+SSI(0.8)+foc(3.0)| panrestormer | 48.3      | 1581.68  | 0.18     | 4.01    | 41.3451  | 0.9664   | 0.0237   | 30.2694  | 0.9177   | 0.0779   | 48.1254  | 0.9880   | 0.0099   |
| L1(0.8)+SSI(5.0)+foc(1.0)| panrestormer | 48.3      | 1581.68  | 0.18     | 3.26    | 41.6633  | 0.9689   | 0.0228   | 30.3987  | 0.9227   | 0.0767   | 48.2504  | 0.9887   | 0.0099   |
| L1(1.5)+SSI(4.0)+foc(1.5)| panrestormer | 48.3      | 1581.68  | 0.18     | 3.21    | 41.7009  | 0.9689   | 0.0228   | 30.4560  | 0.9225   | 0.0761   | 48.2870  | 0.9887   | 0.0098   |
| L1(1.5)+SSI(3.5)+foc(1.5)| panrestormer | 48.3      | 1581.68  | 0.18     | 3.24    | 41.6437  | 0.9686   | 0.0229   | 30.4165  | 0.9219   | 0.0765   | 48.2798  | 0.9886   | 0.0098   |
| L1(0.8)+SSI(3.0)+foc(1.0)| panrestormer | 48.3      | 1581.68  | 0.18     | 5.38    | 41.5234  | 0.9681   | 0.0232   | 30.3885  | 0.9213   | 0.0768   | 48.0565  | 0.9883   | 0.0101   |
| L1(0.5)+SSI(8.0)+foc(0.5)| panrestormer | 48.3      | 1581.68  | 0.18     | 5.41    | 41.6847  | 0.9694   | 0.0228   | 30.4500  | 0.9233   | 0.0760   | 48.1729  | 0.9887   | 0.0100   |

We conducted extensive experiments in `loss_abl.yml` and `loss_abl_1enc.yml`. Experiments on the 2-encoder model revealed that SSIM loss is the most important, but using SSIM loss alone is insufficient—L1 loss is also necessary. Based on this conclusion, we used the effective single-encoder model to conduct more experiments to find the best loss combination. The combination (1.5, 4.0, 1.5) is the best we found.

#### Generalization Analysis

A series of papers have focused on generalization research. They trained their pan-sharpening models using only one dataset and then tested them on three different datasets, attempting to develop a generalizable model. However, the optimization objectives of these different datasets are quite disparate, and this inter-dataset gap makes it difficult for them to achieve strong generalization performance.

![metrics from a generalization paper](fig/from_DDIF.png)

As shown in the figure below, many of our testing methods underwent one epoch of training on WV2. At the time, running these one-epoch models was for exploring the model's fitting performance and selecting candidate models for subsequent experiments. However, we inadvertently compared the generalization metrics after one epoch of WV2 training with a data table from a generalization training paper. The result was that our model's performance after just one epoch was in no way inferior to the results in those papers. This suggests that their subsequent hundreds of epochs were essentially just further fitting or overfitting to WV2, while making no progress on the untrained WV3 and GF2 datasets.

![one epoch result on WV2](fig/one_epoch.png)

Therefore, we believe the current problem lies with the data, and it is unrealistic to enhance generalization capability by solely improving the model. Furthermore, the "all-in-one" training paradigm with three datasets can significantly boost test performance on WV2 full-resolution data, regardless of the model. Modifying the model can at most improve performance by 0.0x; to truly improve QNR from 0.79 to 0.89, the data is the key factor.


| Metrics | ours_small | msdcnn_original | ours_huge_body | pannet_original | pnn_original | pscinn_orig | ours | panflow_original | ours_large_body |
|---|---|---|---|---|---|---|---|---|---|
| $D_\lambda \downarrow$ | 0.0613 | 0.0487 | 0.0579 | 0.0581 | 0.0504 | 0.0474 | 0.0584 | 0.0514 | 0.0548 |
| $D_S \downarrow$ | 0.0681 | 0.0650 | 0.0640 | 0.0753 | 0.0691 | 0.0715 | 0.0666 | 0.0620 | 0.0624 |
| QNR $\uparrow$ | 0.8751 | 0.8898 | 0.8823 | 0.8726 | 0.8844 | 0.8849 | 0.8793 | 0.8900 | 0.8865 |

![WV2 full test result from other paper](fig/from_PanLUT.png)

### Troubleshooting

1. In the old codebase, `data['upscale']` had a spelling error—it was written as 'upsacle'. We have fixed this, and now you can use either spelling. However, if any bugs related to this occur, try using the alternative spelling.
2. The method `sdfi.py` is actually SFDI. If you want to fix this, simply rename the file and the class name within the file, and update the config file accordingly.
3. The VGG loss and GAN loss may not work properly—we haven't tested them thoroughly. Some users report that they actually lower performance. You can try them if you wish.
4. The naming in the `pantiny` model is misleading: the "refinement" module is actually the latent module, while `advanced_refine` is the true refinement module. We kept the name "refinement" to maintain compatibility with previous weights, which is why the name `advanced_refine` appears for the actual refinement module.
5. `Panrestormer` is our small model version, not a reproduction of the original Restormer. We kept this name to load previous configurations.
6. If you want to perform inference with (1024, 1024) size tensors on Pan-Mamba or CFDCNet, you need to modify the code. They use fixed sizes in their implementation, which we copied directly:
```python
# For Pan-Mamba
ms = global_f.transpose(1, 2).view(B, C, 128, 128) # -> change to 1024 x 1024
```
```python
# For CFDCNet
self.featdim = 1049600 # change to channel * H * W during __init__, should be 33554432
```

<!-- 8. SFDI -->

### Additional Explanations

1. **Why is training time not correlated with parameter count?** The training times provided in the table are not very accurate. Some methods, such as Pan-Mamba, do not actually require 20+ hours to train. Other users were using the same GPU simultaneously, which slowed down our training speed. These times should be considered as rough references only.

2. **Why not simply increase all loss weights?** You might notice in the loss table above that a loss weight combination of (3,3,3) performs better than (1,3,1). We didn't delve into this in detail in the paper, or rather, we avoided this issue. In fact, this is a direction worth exploring further:

---

  * For some models, a weight ratio of (0.8, 3, 1) is better than (1.5, 4, 1.5). For instance, CFDCNet can achieve a PSNR of 42.6 on the WV2 dataset with the former, but for PanTiny, the latter performs better.
  * The loss consistently decreased throughout the training process rather than fluctuating. We believe 500 epochs did not allow the model to converge optimally. Increasing the learning rate (or the weights of all losses) might achieve the effect of more epochs within the same 500-epoch limit.
  * When increasing all weights, the primary improvement came from the increased weight of SSIM, with the other two showing slight but not significant improvements.
  * There is an upper limit to performance improvement from increasing weights. For example, setting the SSIM weight to 8 did not yield further gains compared to 4.
  * Given sufficient epochs, we believe the (3,3,3) configuration would ultimately perform worse than (1,1,1). Therefore, the loss ratio is what truly matters. This is why we believe (1,3,1) is actually better than (3,3,3) in the table above, despite (3,3,3) having seemingly higher metrics.
  * Due to insufficient evidence to discuss this point in the paper and page limitations, we directly presented the conclusion along with reasoning that reviewers would likely appreciate.

### Acknowledgements

Our repository is based on [Pan-Mamba](https://github.com/alexhe101/Pan-Mamba.git). We thank the authors for their excellent work.

### Citation

If you find this codebase useful, please consider citing our paper:
```
placeholder
```