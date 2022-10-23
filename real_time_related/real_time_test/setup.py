#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from pip._internal.req import parse_requirements

with open('README.md') as readme_file:
    readme = readme_file.read()

requirement_f = 'pytorch_scripts/envs/requirement.txt'
reqs = []
req_links = []
with open(requirement_f) as f:
    for req in f.readlines():
        if 'find-links' in req:
            req_links.append(req)
        else:
            reqs.append(req)

setup(
    name='unsup-plas',
    version='0.1.0',
    description="Unsupervised neuronal plasticity test",
    long_description=readme,
    packages=find_packages(exclude=['scripts', 'notebooks', 'unsup_plas']),
    include_package_data=True,
    install_requires=reqs,
    dependency_links=req_links,
    license="MIT license",
    zip_safe=False,
    keywords='unsup-plas',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
)
