import os
import pkg_resources
from setuptools import setup, find_packages

setup(
    name="mpr",
    version="1.0",
    description="",
    author="Alex Oesterling, Carol Long",
    author_email="aoesterling@g.harvard.edu",
    py_modules=["mpr"],
    packages=find_packages(exclude=["experiments*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
)