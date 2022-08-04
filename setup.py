#!/usr/bin/env python
# Adapted from https://github.com/pypa/sampleproject/blob/master/setup.py
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


# VERSION
try:
    from lithium_at import __version__
except:
    __version__ = ''

setup(
    name='lithium_at',
    version=__version__,
    author='Rob Jeffries',
    author_email='r.d.jeffries@keele.ac.uk',
    license='GNU GPLv3',
    description='Estimates ages for a set of stars based on lithium EWs and Teff.',
    long_description=long_description,
    classifiers = [
        'Development Status :: 1 - Planning Development Status ',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python'
    ],
    packages=['lithium_at'],
    install_requires=[ 'numpy', 'pandas'],
    entry_points={
      'console_scripts': [
        'lithium_at=lithium_at.lithium_at:main',
      ],
    },
)

