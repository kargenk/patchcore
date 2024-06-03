# project_template

This repository is template for Docker + Python(pyenv) + Poetry.

## 🛠Requirements

- python = "^3.11"

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
git clone https://github.com/kargenk/project_template.git
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
cd project_template/src

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

## 🚀Updates

**yyyy.mm.dd**

- updates

## 📧Authors

kargenk a.k.a **gengen**(https://twitter.com/gengen_ml)

## ©License

This repository is free, but I would appreciate it if you could inform the author when you use it.

<!-- ProjectTemplate is under [MIT licence](https://en.wikipedia.org/wiki/MIT_License) -->
