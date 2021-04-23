from setuptools import setup, find_packages
import sys
from cbsyst import __version__

if sys.version_info[0] < 3:
    sys.exit(('*********************************************\n' +
              'Sorry - cbsyst does not work with Python 2.x.\n' +
              'Please install Python 3 and try again.\n' +
              '*********************************************\n'))

setup(name='cbsyst',
      version=__version__,
      description='Tools for calculating ocean C and B chemistry.',
      url='https://github.com/oscarbranson/cbsyst',
      author='Oscar Branson',
      author_email='oscarbranson@gmail.com',
      license='MIT',
      packages=find_packages(),
      keywords=['science', 'chemistry', 'oceanography', 'carbon'],
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3 :: Only',
                   ],
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'uncertainties',
                        'tqdm'],
      package_data={'cbsyst': ['test_data/GLODAP_data/*.py',
                               'test_data/Lueker2000/*.csv']},
      zip_safe=True)
