from setuptools import find_packages, setup, Extension
#from distutils.core import setup, Extension
import glob
import numpy as np
import pysam
from Cython.Build import cythonize
from Cython.Distutils import build_ext


include_path = [np.get_include()]
include_path.extend(pysam.get_include())
ext_modules=cythonize([
            Extension('*', ['deep_cfNA/*.pyx'],
                            include_dirs = include_path)
])

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
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext}
)
