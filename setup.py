from distutils.core import setup

setup(name='open_source_i_n',
      version='1.0',
      packages=['open_source_i_n'],
      long_description=open('README.md').read(),
      python_requires='>=3.5',
      setup_requires=['setuptools', 'wheel'],
      install_requires=['matplotlib', 'numpy', 'pandas', 'recordclass', 'scipy', 'skyfield', 'itur'],
      extras_require={
          'dev': [
              'pylint',
              'pytest',
              'yapf',
          ],
      },
      package_dir={'': 'src'})
