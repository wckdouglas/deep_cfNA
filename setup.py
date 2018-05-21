from setuptools import find_packages, setup, Extension
#from distutils.core import setup, Extension
import glob

setup(
    name = 'deep_cfNA',
    version = '0.1',
    description = 'Using deep learning to classify fragments from TGIRT-seq libraries of cell-free genetic materials',
    url = '',
    author = 'Douglas C. Wu',
    author_email = 'wckdouglas@gmail.com',
    license = 'MIT',
    packages = find_packages(),
    scripts = glob.glob('bin/*'),
    zip_safe = False,
    data_files=[('model', glob.glob('model/deep_cfNA_*'))],
)
