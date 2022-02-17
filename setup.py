import os
from typing import List

from setuptools import find_packages, setup

_repo: str = "uc-timeseries"
_pkg: str = "uc_timeseries"
_version = "0.0.1"


def read(fname) -> str:
    """Read the content of a file.

    You may use this to get the content of, for e.g., requirements.txt, VERSION, etc.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Declare minimal set for installation
required_packages: List[str] = [
    # "pandas==1.3.2",
    # "numpy==1.20.3",
    # "kats==0.1.0",
    # "hvplot==0.7.3",
    # "panel==0.9.7",
    # "bokeh==2.2.0",
    # "jupyter_bokeh",
    # "shap",
    # "streamlit==0.86.0",
    # "streamlit-aggrid==0.2.1",
]

setup(
    name=_pkg,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version=_version,
    description="A short description of the project.",
    long_description=read("README.md"),
    author="yapweiyih",
    url=f"https://github.com/abcd/{_repo}/",
    download_url="",
    project_urls={
        "Bug Tracker": f"https://github.com/abcd/{_repo}/issues/",
        "Documentation": f"https://{_repo}.readthedocs.io/en/stable/",
        "Source Code": f"https://github.com/abcd/{_repo}/",
    },
    license="MIT License",
    platforms=["any"],
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
    python_requires="==3.9.*",
    install_requires=required_packages,
)
