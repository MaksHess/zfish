#!/usr/bin/env python

"""The setup script."""

from Cython.Build import cythonize
from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Cython",
    "h5py",
    "itk",
    "numpy",
    "xarray",
    "xxhash",
    "polars",
    "pandas",
    "tqdm",
    "toolz",
    "napari[all]",
    "pydantic",
    "pydantic_yaml",
    "spatial_image",
    "multiscale_spatial_image",
    "imageio",
    "zarr",
    "scikit-learn",
    "colorcet",
    "pyarrow",
    "pingouin",
    "numba",
    "python-forge",
    "feature_engine",
    "strenum",
]

test_requirements = []

setup(
    author="Max Timo Hess",
    author_email="max.hess@mls.uzh.ch",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="""Python Boilerplate contains all the boilerplate you need to create a Python package.""",
    entry_points={
        "console_scripts": [
            "zfish=zfish.cli:main",
        ],
    },
    install_requires=requirements,
    ext_modules=cythonize(
        "zfish/features/neighborhood/neighborhood_matrix_parallel.pyx"
    ),
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="zfish",
    name="zfish",
    packages=find_packages(include=["zfish", "zfish.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/MaksHess/zfish",
    version="0.1.0",
    zip_safe=False,
)
