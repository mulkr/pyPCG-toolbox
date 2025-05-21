from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    
with open("requirements.txt", "r", encoding="utf-8") as f:
    dependencies = f.read().splitlines()

version = "0.1x"
with open('pyproject.toml', 'r', encoding="utf-8") as toml:
    lines = toml.readlines()
    for line in lines:
        if line.startswith("version"):
            parts = line.split("=")
            version = parts[1].strip()[1:-2]

setup(
    name="pyPCG_toolbox",
    version=version,
    description="A PCG processing toolbox",
    author="Kristóf Müller, Janka Hatvani, Miklós Koller, Márton Áron Goda",
    author_email="muller.kristof@itk.ppke.hu, goda.marton.aron@itk.ppke.hu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mulkr/pyPCG-toolbox/",
    project_urls={"Bug Tracker": "https://github.com/mulkr/pyPCG-toolbox/issues",},
    license="GPL-3.0-only",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    packages=["pyPCG"],
    install_requires=dependencies,
    python_requires=">=3.10"
)