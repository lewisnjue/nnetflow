from setuptools import setup, find_packages

setup(
    name="numpyflow",
    version="0.1.0",
    description="A minimal neural network framework with autodiff",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/numpyflow",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
