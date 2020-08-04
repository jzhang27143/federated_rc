import setuptools

setuptools.setup(
    name="federatedrc",
    version="1.0.0",
    author="Jason Zhang",
    author_email="jzhang27143@gmail.com",
    description="A python package for federated learning under resource constraints in PyTorch",
    url="https://github.com/jzhang27143/federated_rc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=["ifaddr", "torch", "torchvision"],
)
