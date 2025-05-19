from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nnetflow",
    version="0.1.0",
    description="A minimal neural network framework with autodiff and NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lewis Njue",
    author_email="lewiskinyuanjue.ke@gmail.com",
    url="https://github.com/lewisnjue/nnetflow",
    packages=find_packages(),
    install_requires=["numpy"],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "neural network",
        "autodiff",
        "deep learning",
        "machine learning",
        "numpy",
        "backpropagation",
        "AI",
        "educational",
    ],
)
