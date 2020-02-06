from setuptools import setup

setup(
        name='evaluate-instance-segmentation',
        version='0.1',
        description='Evaluate instance segmentation (just instances, no semantic classes)',
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
        ]
)
