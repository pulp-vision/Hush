from setuptools import setup, find_packages

setup(
    name="deepfilternet-se",
    version="1.0.0",
    description="DeepFilterNet Speech Enhancement with Background Speaker Suppression",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="",
    license="Apache-2.0",
    packages=find_packages(include=["model", "model.*", "training", "training.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.1",
        "scipy>=1.10.0",
        "h5py>=3.8.0",
    ],
    extras_require={
        "train": ["tensorboard>=2.12.0", "mlflow>=2.4.0"],
        "eval": ["pesq>=0.0.4", "pystoi>=0.3.3"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
