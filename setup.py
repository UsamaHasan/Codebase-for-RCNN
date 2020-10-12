import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cbrcnn", # Replace with your own username
    version="0.0.1",
    author="Usama Hasan",
    author_email="usamahasan72@gmail.com",
    description="A Code Base to implement Regional Conv Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pucit-waqar/Codebase-of-RCNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

