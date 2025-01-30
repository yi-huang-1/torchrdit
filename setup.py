import re
from setuptools import setup, find_packages
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Extract version from __init__.py
def get_version():
    with open("torchrdit/__init__.py", "r") as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

setup(
    name='torchrdit',
    version=get_version(),  # Dynamically set the version
    description='A PyTorch based package for designing and analyzing optical devices, utilzing the Rigorous Diffraction Interface Theory (R-DIT).',
    author='Yi Huang',
    author_email='yi_huang@student.uml.edu',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scipy',
        'scikit-image',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.6',
        project_urls={
        'Documentation': 'https://github.com/yi-huang-1/torchrdit/wiki',
        'Source': 'https://github.com/yi-huang-1/torchrdit',
        'Tracker': 'https://github.com/yi-huang-1/torchrdit/issues',
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)