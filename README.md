# TorchRDIT

![PyPI - Version](https://img.shields.io/pypi/v/torchrdit)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchrdit)
![PyPI - License](https://img.shields.io/pypi/l/torchrdit)

`TorchRDIT` is a software package for the paper:

- [Eigendecomposition-free inverse design of meta-optics devices](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-8-13986&id=548527)
- [Inverse Design of Photonic Structures Using Automatic Differentiable Rigorous Diffraction Interface Theory](https://doi.org/10.1364/CLEO_AT.2023.JTu2A.119)

--

## Features

The EM algorithms used in this package are Rigorous Coupled Wave Analysis (RCWA) and Rigorous Diffraction
Interface Theory (R-DIT). The solver is developed on an automatic differentiable
framework (PyTorch) and is fully differentiable for more advanced applications.

---

## Installation

You can install the package directly from [PyPI](https://pypi.org/project/torchrdit/):

```bash
pip install torchrdit
```

--

## Dependencies

- pytorch>=2.5.1
- torchvision
- numpy
- tensorboard
- matplotlib
- tqdm
- scikit-image

--

## Usage

Please find the demo jupyter noteboks for more information.

- `examples/Demo-01.ipynb`
- `examples/Demo-02.ipynb`
- `examples/Demo-03.ipynb`

--

## Related Works

- [Eigendecomposition-free inverse design of meta-optics devices](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-8-13986&id=548527)
- [Inverse Design of Photonic Structures Using Automatic Differentiable Rigorous Diffraction Interface Theory](https://opg.optica.org/abstract.cfm?uri=CLEO_AT-2023-JTu2A.119)

--

## Citation

The package contains the work from multiple publicatinos. Please consider to cithe the following paper for the R-DIT solver:

```text
Yi Huang, Ziwei Zhu, Yunxi Dong, Hong Tang, Bowen Zheng, Viktor A. Podolskiy, and Hualiang Zhang, "Eigendecomposition-free inverse design of meta-optics devices," Opt. Express 32, 13986-13997 (2024)
```

```text
@article{Huang:24,
author = {Yi Huang and Ziwei Zhu and Yunxi Dong and Hong Tang and Bowen Zheng and Viktor A. Podolskiy and Hualiang Zhang},
journal = {Opt. Express},
keywords = {Diffraction theory; Finite-difference time-domain method; Inverse design; Machine learning; Neural networks; Refractive index},
number = {8},
pages = {13986--13997},
publisher = {Optica Publishing Group},
title = {Eigendecomposition-free inverse design of meta-optics devices},
volume = {32},
month = {Apr},
year = {2024},
url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-32-8-13986},
doi = {10.1364/OE.514347},
abstract = {The inverse design of meta-optics has received much attention in recent years. In this paper, we propose a GPU-friendly inverse design framework based on improved eigendecomposition-free rigorous diffraction interface theory, which offers up to 16.2\&\#x2009;\&\#x00D7; speedup over the traditional inverse design based on rigorous coupled-wave analysis. We further improve the framework\&\#x2019;s flexibility by introducing a hybrid parameterization combining neural-implicit and traditional shape optimization. We demonstrate the effectiveness of our framework through intricate tasks, including the inverse design of reconfigurable free-form meta-atoms.},
}
```

---

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.en.html) file for details.

---

## Feedback and Support

If you have any questions, issues, or suggestions, please open an [issue](https://github.com/yi-huang-1/torchrdit/issues) or email us at `yi_huang@student.uml.edu`.

---

## Release History

- **0.1.0**
  - Initial release.
