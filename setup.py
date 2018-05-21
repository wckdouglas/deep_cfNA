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
    zip_safe = False,
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext}
    package_data={'deep_cfNA/model': ['model/deep_cfNA_*']},
)
