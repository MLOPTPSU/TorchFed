import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fedtorch", 
    version="0.1.2",
    author="MM. Kamani",
    author_email="mm.kamani7@gmail.com",
    description="An open-source Python package for distributed and federated training of \
         machine learning models using PyTorch distributed API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MLOPTPSU/FedTorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch >= 1.6.0',
        'torchvision >= 0.8.2',
        'numpy >= 1.18'
    ],
)