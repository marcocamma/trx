from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='trx',
      version='0.3.0',
      description='Tools for time resolved x-ray techniques',
      long_description=readme(),
      url='https://github.com/marcocamma/trx',
      author='marco cammarata',
      author_email='marcocamma@gmail.com',
      license='MIT',
      packages=['trx'],
      install_requires=[
          'numpy',
          'fabio',
          'pyFAI',
          'matplotlib',
          'datastorage',
      ],
      zip_safe=False)
