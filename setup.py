"""Setup script for portfolio optimization project"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="portfolio-optimization-rl",
    version="0.1.0",
    author="Portfolio Optimization Team",
    author_email="team@portfolio-optimization.com",
    description="Multi-Asset Portfolio Optimization with Reinforcement Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/portfolio-optimization-rl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "gpu": [
            "torch[cuda]>=1.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "portfolio-train=scripts.train:main",
            "portfolio-backtest=scripts.backtest:main",
            "portfolio-collect-data=scripts.collect_data:main",
            "portfolio-dashboard=app.dashboard.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords="portfolio optimization reinforcement learning finance trading",
    project_urls={
        "Bug Reports": "https://github.com/your-username/portfolio-optimization-rl/issues",
        "Source": "https://github.com/your-username/portfolio-optimization-rl",
        "Documentation": "https://portfolio-optimization-rl.readthedocs.io/",
    },
)
