# PatchCore

This repository is one of the PatchCore implementations.

## 🛠Requirements

- python = "^3.10"
- torch = "^2.3.0"
- torchvision = "^0.18.0"
- numpy = "^1.26.4"
- scikit-learn = "^1.5.0"
- opencv-python = "^4.9.0.80"
- scipy = "^1.13.1"
- pillow = "^10.3.0"
- tqdm = "^4.66.4"

## 🌲Directory

<pre>
project_template
├─── data
│       ├── train  : 訓練データ
│       └── test   : pdfデータと変換したpngのデータ(必要な場合)
│
├─── environments  : Dockerfileなどの実行環境
│
├─── models        : 学習済みのモデル
│
├─── notebooks     : 実験用のJupyter Notebook
│
├─── outputs       : 出力結果
│
├─── scripts       : スクリプト
│
└─── src           : ソースコード
</pre>

## ⚙️Installation

Clone this repository.

```bash
git clone https://github.com/kargenk/patchcore.git
```

### Using Poetry

Install Poetry:

```bash
# Install the Poetry dependency management tool, skip if installed
# Reference: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -
```

Create environment with Poetry:

```bash
cd patchcore/src

# Install the project dependencies and Activate
poetry shell
poetry install
```

## 💻Usage

Write usage of this repository

<!-- > [!WARNING]
> This is warnings -->

<!-- > [!IMPORTANT]
> This is importants -->

## 📝Note

<!-- > [!NOTE]
> This is notes -->

### Execution Environments

- OS: Ubuntu 22.04.4 LTS
- CPU: AMD Ryzen 7 5700G with Radeon Graphics (8 Core 16 Threads)
- GPU: GeForce RTX 3080 Ti (12GB)
- Memory: 16GB

<!-- OS: lsb_release -a -->
<!-- CPU: lscpu -->
<!-- GPU: lspci | grep -i nvidia -->
<!-- Memory: sudo dmidecode -t memory -->

### Calculation Time⌛

<!-- - processing X takes N \[sec\] to process each image(H x W px). -->
<!-- ![calculation Time](api/img/calculation_time.png) -->
- Using ResNet50 as Feature Extractor
  - resolution: 256x256
    - It takes N \[hour\] for sampling 1% coreset memorybank. (GPU: A6000, 309 images)
    - It takes N \[sec\] for inference.
  - resolution: 512x512
    - It takes 3 ~ 4 \[hour\] for sampling 1% coreset memorybank. (GPU: A6000, 309 images)
    - It takes 0.3 \[sec\] for inference. (GPU: A6000, 309 images)

## 🚀Updates
**2024.07.31**

- support MVTecAD dataset

**2024.07.29**

- add segmentation

**2024.06.03**

- add kNearestNeighbor
- add Random Projection and Greedy Sampling
- add Memory Bank

## 📧Authors

kargenk a.k.a **gengen**(https://twitter.com/gengen_ml)

## ©License

This repository is free, but I would appreciate it if you could inform the author when you use it.

<!-- ProjectTemplate is under [MIT licence](https://en.wikipedia.org/wiki/MIT_License) -->
