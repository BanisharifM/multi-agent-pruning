"""
Setup script for Multi-Agent LLM Pruning Framework
"""

from setuptools import setup, find_packages
import os

# Read version from version.py
version_file = os.path.join(os.path.dirname(__file__), 'multi_agent_pruning', 'version.py')
with open(version_file) as f:
    exec(f.read())

# Read README for long description
readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_file):
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Multi-Agent LLM Pruning Framework for Neural Network Compression"

# Read requirements
requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
with open(requirements_file, 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="multi-agent-pruning",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=__url__,
    
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    python_requires=">=3.8",
    install_requires=requirements,
    
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "wandb": [
            "wandb>=0.13.0",
        ],
        "full": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
            "wandb>=0.13.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "multi-agent-prune=multi_agent_pruning.scripts.prune:main",
            "multi-agent-train=multi_agent_pruning.scripts.train:main",
            "multi-agent-evaluate=multi_agent_pruning.scripts.evaluate:main",
            "multi-agent-compare=multi_agent_pruning.scripts.compare_baselines:main",
            "multi-agent-setup=multi_agent_pruning.scripts.setup_models:main",
        ],
    },
    
    include_package_data=True,
    package_data={
        "multi_agent_pruning": [
            "configs/**/*.yaml",
            "configs/**/*.yml",
        ],
    },
    
    zip_safe=False,
)

