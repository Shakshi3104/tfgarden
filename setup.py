import setuptools

setuptools.setup(
    name="tfgarden",
    version="0.3.1",
    author="Shakshi3104",
    description="TensorFlow model Garden for HASC",
    url="https://github.com/Shakshi3104/tfgarden",
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires=">=3.6, <4",
    install_requires=["tensorflow"],
    package_dir={'': 'src'},
)
