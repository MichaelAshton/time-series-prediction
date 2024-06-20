from setuptools import setup

requirements = [
      'beautifulsoup4==4.7.1',
      'selenium==3.141.0',
      'slackclient==1.3.1',
      'sqlalchemy==1.3.5',
      'mxnet==1.9.1',
      'gluonts==0.3.0',
      'numpy==1.22.0',
      'xlrd==1.2.0',
      'PyYAML==5.4',
      'scikit-learn==1.5.0',
      'configobj>=5.0.7',
      'pytz==2019.2',
      'comet-ml'
      ]

setup(name='eureka254',
      version='0.1',
      description='The first steps towards financial freedom',
      url='https://github.com:MichaelAshton/virtual_betting.git',
      email='ashtonmyk@gmail.com',
      authors='Michael Ashton and Java Black',
      license='MIT',
      install_requires=requirements,
      packages=['eureka254'],
      zip_safe=False)
