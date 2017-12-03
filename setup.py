from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='trx',
      version='0.5.12',
      description='tools for (T)ime (R)esolved (X)-ray techniques',
      long_description=readme(),
      url='https://github.com/marcocamma/trx',
      author='marco cammarata',
      author_email='marcocamma@gmail.com',
      license='MIT',
#      packages=['trx'],
      packages=find_packages("."),
      install_requires=[
          'numpy',
          'fabio',
          'pyFAI',
          'matplotlib',
          'datastorage>=0.4.3',
          'statsmodels'
      ],
      zip_safe=False)
