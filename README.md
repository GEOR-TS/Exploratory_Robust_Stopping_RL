Exploratory Robust Stopping with Reinforcement Learning

This repository contains source code for the paper "Robust Exploratory Stopping under Ambiguity in Reinforcement Learning" by _Kyunghyun Park, Hoi Ying Wong and Junyan Ye_.

Basic structure
- Call/
  - PolicyIteration_call.py — Algorithm class for Policy Iteration with DL-based Policy Evaluation (Call)
  - Deep_Backward_BSDE_call.py — Deep Backward BSDE for Reference Values (Call)
  - Implicit_FDM_call.py — Implicit Finite Difference Method for option price (Call)
  - PI_NN_load_call.py — Utilities for loading saved PI models (Call)
  - PolicyIteration_Convergence_call.ipynb — Notebook for PI algorithm Implementation and Convergence Evaluation (Call)
  - PolicyIteration_Robustness_call.ipynb — Notebook for Robustness Evaluation under Dividend Rate Misspecification (Call)
- Put/
  - PolicyIteration_put.py — Algorithm class for Policy Iteration with DL-based Policy Evaluation (Put)
  - Deep_Backward_BSDE_put.py — Deep Backward BSDE for Reference Values (Put)
  - PolicyIteration_Convergence_put.ipynb — for PI algorithm Implementation and Convergence Evaluation (Put)

Minimal requirements
- Python 3.8+ (3.13 recommended)
- Packages:
  - torch
  - numpy
  - scipy
  - pandas
  - matplotlib
  - tqdm
  - jupyter (optional, for notebooks)

Quick install
- pip install torch numpy scipy pandas matplotlib tqdm
- Optional for notebooks: pip install jupyter

License
- This project is licensed under the MIT License.

Citation
  ```bibtex
  @article{PWY2025RobustStoppingRL,
    title={Robust Exploratory Stopping under Ambiguity in Reinforcement Learning},
    author={Park, Kyunghyun and Wong, Hoi Ying and Ye, Junyan},
    journal={Preprint},
    month={October},
    year={2025},
    note={[Add DOI/arXiv link when available]}
  }
  ```
