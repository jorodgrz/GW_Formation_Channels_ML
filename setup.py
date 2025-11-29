"""
Setup script for the ASTROTHESIS research pipelines package.

Install with: pip install -e .
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "configs" / "infrastructure" / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, encoding="utf-8") as req_file:
        requirements = [
            line.strip()
            for line in req_file
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="astrothesis-pipelines",
    version="0.1.0",
    author="Joseph Rodriguez",
    author_email="",
    description="Physics-informed research pipelines for gravitational-wave formation channel studies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UCSD-Astronomy/ASTROTHESIS",
    packages=find_packages(include=["pipelines", "pipelines.*"]),
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
        "console_scripts": [
            "generate-compas-ensemble=pipelines.ensemble_generation.compas.generate_ensemble:main",
            "generate-cosmic-ensemble=pipelines.ensemble_generation.cosmic.generate_ensemble:main",
            "generate-multi-code-ensemble=pipelines.ensemble_generation.multi_code.unified_generator:main",
            "train-formation-channels=pipelines.inference_and_falsification.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

