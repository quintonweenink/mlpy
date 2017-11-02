#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Quinton Weenink'


from setuptools import setup, find_packages


setup(
    name="PyBrain",
    version="0.3.3",
    description="PSO",
    license="MIT",
    keywords="PSO",
    url="http://quinton-weenink.herokuapp.com",
    packages=find_packages(exclude=['examples']),
    include_package_data=True,
    test_suite='mlpy.experiments.test.run_tests',
    #package_data={'mlpy': ['rl/environments/ode/models/*.xode']},
    install_requires = ["numpy", "matplotlib", "scipy"],
)