from setuptools import setup, find_packages
import sys

if sys.version_info[0] < 3:
    sys.exit(('*********************************************\n' +
              'Sorry - cbsyst does not work with Python 2.x.\n' +
              'Please install Python 3 and try again.\n' +
              '*********************************************\n'))

setup(name='cbsyst',
      version='0.2-dev',
      description='Tools calculating ocean C and B chemistry.',
      url='https://github.com/oscarbranson/cbsyst',
      author='Oscar Branson',
      author_email='oscarbranson@gmail.com',
      license='MIT',
      packages=find_packages(),
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Scientists',
                   'Programming Language :: Python :: 3',
                   ],
      install_requires=['numpy',
                        'tqdm'],
      zip_safe=False)