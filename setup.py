from setuptools import setup

setup(
    name='evaluate-instance-segmentation',
    version='0.1',
    description='Evaluate instance segmentation results (just instances, no semantic classes)',
    url='https://github.com/Kainmueller-Lab/evaluate-instance-segmentation',
    author='Peter Hirsch, Lisa Mais',
    author_email='kainmuellerlab@mdc-berlin.de',
    license='MIT',
    install_requires=[
        'h5py',
        'numpy',
        'scipy',
        'scikit-image',
        'toml',
        'tifffile',
        'zarr',
    ],
    packages=[
        'evaluateInstanceSegmentation',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
