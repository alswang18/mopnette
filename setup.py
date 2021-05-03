from setuptools import setup, find_packages
import codecs
import os


VERSION = '0.0.11'
DESCRIPTION = 'Wrapper for quantizing vision models'
LONG_DESCRIPTION = 'A package that allows to build quantized resnet vision models.'

# Setting up
setup(
    name="mopnette",
    version=VERSION,
    author="Alec Wang, Tricia Cu, Jim Tan",
    author_email="alswang18@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['torch==1.7.1', 'fastai'],
    keywords=['python', 'image', 'quantization', 'pytorch'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)