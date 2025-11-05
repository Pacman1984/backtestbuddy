from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="backtestbuddy",
    version="0.1.10",
    author="Sebastian Pachl",
    description="A flexible backtesting framework for trading and betting strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pacman1984/backtestbuddy",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.13.3",
        "pandas>=0.25",
        "matplotlib>=3.0.0",
        "scipy>=1.0.0",
        "plotly>=4.14.0",
        "scikit-learn>=0.24.0",
        "nbformat>=4.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "black>=21.5b1",
            "isort>=5.9.0",
            "pylint>=2.8.0",
            "tox>=3.24",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "backtestbuddy=backtestbuddy.cli:main",
        ],
    },
    zip_safe=False,
)