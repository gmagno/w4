#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme: str = readme_file.read()

requirements: list = ["numpy", "scipy"]

test_requirements: list[str] = [
    "pytest>=3",
]

setup(
    author="Gonçalo Magno",
    author_email="goncalo.magno@gmail.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    description="W4 Method for Nonlinear Root Finding",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="w4",
    name="w4",
    packages=find_packages(include=["w4", "w4.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/gmagno/w4",
    version="0.1.2",
    zip_safe=False,
)
