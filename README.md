# 🧠 HSST-EEG [IEEE Access 2026]
[![Paper](https://img.shields.io/badge/ISBI%202026-Oral-blue)](https://ieeexplore.ieee.org/abstract/document/11533369)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📂 Repository Structure
```
HSST-EEG/
├── models/                 # Model architectures
│   ├── hsst.py               # Main model
│   ├── tsception.py          # Ding, Yi, et al. (2022)
│   ├── eeg_conformer.py      # Song, Yonghao, et al. (2022)
│   ├── sparcnet.py           # Jing, Jin, et al. (2023)
│   ├── biot.py               # Yang, Chaoqi, et al. (2023)
├── dataset.py              # Dataset loading
├── train.py                # Main training script
├── utils.py                # Utilities (e.g., metrics)
├── README.md               # This file
```
### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/emeelkaa/hsst_eeg.git
   cd hsst-eeg
```

2. **Create a virtual environment (recommended)**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install PyTorch** (CUDA 12.1)
```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

4. **Install Mamba**

Follow the official instructions at [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba). Both `mamba-ssm` and `causal-conv1d` must be built for your CUDA version.

5. **Install remaining dependencies**
```bash
pip install -r requirements.txt
```
For dataset preprocessing please refer to: https://github.com/emeelkaa/cgm_eeg
## 📧 Contact

For questions, issues, or collaboration inquiries, please contact:

- **Email**: [emilkim01@pusan.ac.kr](mailto:emilkim01@pusan.ac.kr)
- **Author**: Emil Kim

---
## 📚 Citation

If you find our work helpful, please consider citing the following paper:

```bibtex
@article{kim2026hsst,
  title={HSST-EEG: A Hybrid State-Space and Transformer Architecture for EEG Decoding},
  author={Kim, Emil and Gahm, Jin Kyu},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}
```
