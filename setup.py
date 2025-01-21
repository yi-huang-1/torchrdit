import re
from setuptools import setup, find_packages

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
    description='A module for TorchRDIT',
    author='Yi Huang',
    author_email='yi_huang@student.uml.edu',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)