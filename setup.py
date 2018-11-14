from setuptools import setup, find_packages

setup(name='titanic',
      version='0.1',
      description='Package to get predictions',
      author='Kshitij Mahur',
      author_email='k.mathur68@gmail.com',
      license='Open-Source',
      zip_safe=False,
      project_urls={
        "Source Code": "https://github.com/Kshitij68/Productionize-ML-model",
      },
      extras_require={
        'server': [
            'Flask==0.12.2',
            'boto3==1.7.80'
        ],
        'testing': [
            'pytest==3.7.2',
            'pytest-cov==2.5.1'
        ]
      },
      install_requires=[
            'pandas==0.23.4'
      ])
