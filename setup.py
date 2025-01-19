from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="backtestbuddy",
    version="0.1.4",
    author="Sebastian Pachl",
    description="A flexible backtesting framework for trading and betting strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pacman1984/backtestbuddy",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
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
        ],
    },
    entry_points={
        "console_scripts": [
            "backtestbuddy=backtestbuddy.cli:main",
        ],
    },
    zip_safe=False,
)