from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyPCG_toolbox",
    version="0.1-b",
    description="A PCG processing toolbox",
    author="Kristóf Müller, Janka Hatvani, Miklós Koller, Márton Áron Goda",
    author_email="muller.kristof@itk.ppke.hu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mulkr/pyPCG-toolbox/",
    project_urls={"Bug Tracker": "https://github.com/mulkr/pyPCG-toolbox/issues",},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    packages=["pyPCG"],
    install_requires=[  "scikit-learn>=1.1.3",
                        "scipy>=1.9.3",
                        "PyWavelets>=1.5.0",
                        "pandas>=2.0.0",
                        "matplotlib>=3.6.2",
                        "numpy>=1.25.2",
                        "hsmmlearn @ git+https://github.com/jvkersch/hsmmlearn@master",
                        "emd>=0.6.2",
                        "nolds>=0.5.2",
                        "tqdm>=4.64.1",
                        "pyQt5>=5.15.7"],
    python_requires=">=3.10"
)