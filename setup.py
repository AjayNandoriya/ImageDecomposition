import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imdecomposer",
    version="0.0.1",
    author="Ajay Nandoriya",
    author_email="cartoon.ajay@gmail.com",
    description="Decompose image in different feature domains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AjayNandoriya/ImageDecomposition",
    project_urls={
        "Bug Tracker": "https://github.com/AjayNandoriya/ImageDecomposition/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "imdecomposer"},
    packages=setuptools.find_packages(where="imdecomposer"),
    python_requires=">=3.6",
    install_requires=['tensorflow_gpu==2.5.0',
                      'numpy==1.19.5',
                      'matplotlib==3.4.2',
                      'opencv_python==4.5.3.56',
                      'tensorflow==2.5.0'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite="tests",
)
