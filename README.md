# TorchRDIT

`TorchRDIT` is a software package for the paper:

[Inverse Design of Photonic Structures Using Automatic Differentiable Rigorous Diffraction Interface Theory](https://doi.org/10.1364/CLEO_AT.2023.JTu2A.119)


The EM algorithms used in this package are Rigorous Coupled Wave Analysis (RCWA) and Rigorous Diffraction
Interface Theory (R-DIT). The solver is developed on an automatic differentiable
framework (PyTorch) and is fully differentiable for more advanced applications.

## Dependencies

- pytorch>=2.0.0
- torchvision
- numpy
- tensorboard
- matplotlib
- tqdm

## Usage

Please find the demo jupyter noteboks for more information.

- `./Demo-01.ipynb`
- `./Demo-02.ipynb`
- `./Demo-03.ipynb`

## Citation

The package contains the work from multiple publicatinos. Please consider to cithe the following paper for the R-DIT solver:

```
Y. Huang, H. Tang, B. Zheng, Y. Dong, M. Haerinia, V. Podolskiy, and H. Zhang, "Inverse Design of Photonic Structures Using Automatic Differentiable Rigorous Diffraction Interface Theory," in CLEO 2023, Technical Digest Series (Optica Publishing Group, 2023), paper JTu2A.119.
```

```
@inproceedings{Huang:23,
author = {Yi Huang and Hong Tang and Bowen Zheng and Yunxi Dong and Mohammad Haerinia and Viktor A. Podolskiy and Hualiang Zhang},
booktitle = {CLEO 2023},
journal = {CLEO 2023},
keywords = {Diffraction optics; Diffraction theory; Inverse design; Machine learning; Rigorous coupled wave analysis; Waveguide modes},
pages = {JTu2A.119},
publisher = {Optica Publishing Group},
title = {Inverse Design of Photonic Structures Using Automatic Differentiable Rigorous Diffraction Interface Theory},
year = {2023},
url = {https://opg.optica.org/abstract.cfm?URI=CLEO_AT-2023-JTu2A.119},
doi = {10.1364/CLEO_AT.2023.JTu2A.119},
}
```
