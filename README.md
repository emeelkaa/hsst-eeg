# HSST-EEG [IEEE Access 2026]

This repository contains code for **HSST-EEG**,  
a hybrid mamba-transformer framework for EEG analysis.  
It includes model training scripts, baseline implementations, and dataset handling utilities.  
---

## 📂 Repository Structure
```
CGM-EEG/
├── models/                 # Model architectures
│   ├── hsst.py               # Main model
│   ├── tsception.py          # Ding, Yi, et al. (2022)
│   ├── eeg_conformer.py      # Song, Yonghao, et al. (2022)
│   ├── sparcnet.py           # Jing, Jin, et al. (2023)
│   ├── biot.py               # Yang, Chaoqi, et al. (2023)
├── dataset.py              # Dataset loading
├── train.py                 # Main training script
├── utils.py                # Utilities (e.g., metrics)
├── README.md               # This file
```
### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/emeelkaa/cgm_eeg.git
   cd CGM-EEG
```

2. **Create a virtual environment (recommended)**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```
> **📌 Note:** For Mamba installation, please refer to the official repository:
> [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
## 📧 Contact

For questions, issues, or collaboration inquiries, please contact:

- **Email**: [emilkim01@pusan.ac.kr](mailto:emilkim01@pusan.ac.kr)
- **Author**: Emil Kim

---
