from setuptools import setup, find_packages

setup(
    name="medsegmnist",
    version="0.1.0",
    description="Standardised biomedical image segmentation datasets for PyTorch",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MedSegMNIST/MedSegMNIST",
    license="Apache 2.0",
    packages=find_packages(exclude=["examples", "scripts"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchmetrics>=1.0.0",
        "lightning>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "preprocess": [
            "torchio>=1.0.0",
            "nibabel>=5.0.0",
            "Pillow>=10.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medsegmnist=medsegmnist.cli:main",
        ],
    },
)
