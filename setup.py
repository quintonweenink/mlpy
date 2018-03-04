#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Quinton Weenink'


from setuptools import setup, find_packages

setup(
    name="mlpy",
    author="Quinton Weenink",
    version="0.3.6",
    description="ML Library",
    license="MIT",
    keywords="PSO NN PSONN",
    url="git://github.com/quintonweenink/mlpy",
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    test_suite='mlpy.nn.test.run_tests',
    install_requires = ["numpy", "matplotlib", "scipy"],
)