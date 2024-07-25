from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
    name="pyPCG_toolbox",
    version="0.1-a",
    description="A PCG processing toolbox",
    author="Kristóf Müller",
    author_email="muller.kristof@itk.ppke.hu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://dev.itk.ppke.hu/m.kristof-phd/pypcg-toolbox",
    project_urls={"Bug Tracker": "https://dev.itk.ppke.hu/m.kristof-phd/pypcg-toolbox/-/issues",},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    packages=["pyPCG"],
    install_requires=[required],
    python_requires=">=3.10"
)