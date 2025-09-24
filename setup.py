from setuptools import setup, find_packages

setup(
    name='dynamic_bitmap',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy'],
    description='Dynamic Parallel Bitmap para búsquedas y JOINs masivos',
    author='Jesus Alberto Degollado Lopez',
    author_email='jesusdeg5587@gmail.com',
    url='https://github.com/JesusDeg8061/Dynamic_bitmap.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
