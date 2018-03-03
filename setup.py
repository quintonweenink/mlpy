#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Quinton Weenink'


from setuptools import setup, find_packages

setup(
    name="mlpy",
    version="0.3.5",
    description="PSO",
    license="MIT",
    keywords="PSO",
    url="git://github.com/quintonweenink/mlpy",
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    test_suite='mlpy.nn.test.run_tests',
    #package_data={'mlpy': ['rl/environments/ode/models/*.xode']},
    install_requires = ["numpy", "matplotlib", "scipy"],
)