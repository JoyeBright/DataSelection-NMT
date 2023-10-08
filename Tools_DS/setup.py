import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Tools_DS",
    version="1.1",
    author="Javad Pourmostafa",
    author_email="javad.pourmostafa@gmail.com",
    description="A Python Tool for Selecting Domain-Specific Data in Machine Translation.",
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    url="https://github.com/JoyeBright/DataSelection-NMT",
    packages=setuptools.find_packages(),
    zip_safe=False,
    package_data={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing",
        "Topic :: Utilities"
    ],
)
