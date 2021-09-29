from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8'
]

setup(
    name='prevh',
    version='0.0.1',
    description='A data analysis library for data mining.',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='',
    author='Júlio César Guimarães Costa',
    author_email='juliocesargcosta123@gmail.com',
    license='MIT License',
    classifiers=classifiers,
    keywords='Data Mining',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn']
)
