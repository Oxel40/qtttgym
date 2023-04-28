import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qtttgym",
    version="0.0.1",
    author="Oxel40",
    description="AI Gymnasium like environment for QTTT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Oxel40/qtttgym",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)

