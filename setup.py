from setuptools import setup, find_packages

setup(
    name="quant-finance-rl",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.7",
        "numpy>=1.19",
        "pandas>=1.2",
        "gym>=0.18",
        "matplotlib>=3.3",
        "ta>=0.7",
    ],
)
