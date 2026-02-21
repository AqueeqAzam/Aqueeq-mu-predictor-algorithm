from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mu-predictor",
    version="4.0.0",
    author="Azeeq Azam",
    author_email="txwort@gmail.com",
    description="μ-Predictor Algorithm for Infinite Geometric Nested Radicals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/azeeqazam/mu-predictor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    entry_points={
        "console_scripts": [
            "mu-predictor=mu_predictor.cli:main",
        ],
    },
)
