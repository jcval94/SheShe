\
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
readme = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="sheshe",
    version="0.1.2",
    description="SheShe: Smart High-dimensional Edge Segmentation & Hyperboundary Explorer",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="JC + ChatGPT",
    license="MIT",
    url="https://example.com/sheshe",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn>=1.1",
        "matplotlib",
    ],
    extras_require={
        "dev": ["pytest", "seaborn", "hnswlib"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
