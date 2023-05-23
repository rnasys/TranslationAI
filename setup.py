from setuptools import setup
import io


setup(name='translationai',
      version='1.0',
      description='TranslationAI: An advanced deep learning framework for precise identification'
                  ' of translation start and end sites within mature mRNA sequences',
      long_description=io.open('README.md', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      author='Xiaojuan Fan',
      author_email='xiaojuanfan05@gmail.com',
      packages=['translationai'],
      license='GPLv3',
      install_requires=['keras>=2.0.5',
                        'numpy>=1.14.0',
                        'pandas>=0.24.2'],
      extras_require={'cpu': ['tensorflow>=1.2.0'],
                      'gpu': ['tensorflow-gpu>=1.2.0']},
      package_data={'translationai': ['models/translationAI_2000_l1.h5',
                                 'models/translationAI_2000_l2.h5',
                                 'models/translationAI_2000_l3.h5',
                                 'models/translationAI_2000_l4.h5',
                                 'models/translationAI_2000_l5.h5']},
      entry_points={'console_scripts': ['translationai=translationai.__main__:main']})
