from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='trx',
      version='0.1',
      description='Tools for time resolved x-ray techniques',
      long_description=readme(),
      url='https://git.ipr.univ-rennes1.fr/mcammara/trx',
      author='marco cammarata',
      author_email='marco.cammarata.xray@gmail.com',
      license='MIT',
      packages=['trx'],
      install_requires=[
          'numpy',
          'fabio',
          'pyFAI',
          'matplotlib',
      ],
      zip_safe=False)
