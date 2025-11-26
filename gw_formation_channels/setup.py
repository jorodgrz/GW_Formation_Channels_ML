"""
Setup script for gw_formation_channels package

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="gw_formation_channels",
    version="0.1.0",
    author="Joseph Rodriguez",
    author_email="",
    description="Physics-Informed Deep Learning for GW Formation Channel Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UCSD-Astronomy/ASTROTHESIS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'generate-compas-ensemble=compas_ensemble.generate_ensemble:main',
            'train-formation-channels=train:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

