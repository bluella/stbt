#!/usr/bin/env python3
"""Module to make package from folder"""
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'stbt'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'latest'
finally:
    del get_distribution, DistributionNotFound
