import setuptools

import versioneer

#with open('README.rst') as f:
#    long_description = f.read()

setuptools.setup(
    name="h5schemaesqe",
    version=versioneer.get_version(),
    packages = ["h5schemaesqe"],
    install_requires = ["h5py"],
    author = "James Tocknell",
    author_email = "aragilar@gmail.com",
    #description = "A bunch of tools for using venvs (and virtualenvs) from python.",
    #long_description = long_description,
    license = "BSD",
    #keywords = "virtualenv venv",
    #url = "http://venv_tools.rtfd.org",
    #classifiers=[
    #    'Development Status :: 3 - Alpha',
    #    'Intended Audience :: Developers',
    #    "Topic :: Software Development :: Libraries :: Python Modules",
    #    "Topic :: System :: Shells",
    #    'License :: OSI Approved :: BSD License',
    #    'Programming Language :: Python :: 2',
    #    'Programming Language :: Python :: 2.6',
    #    'Programming Language :: Python :: 2.7',
    #    'Programming Language :: Python :: 3',
    #    'Programming Language :: Python :: 3.1',
    #    'Programming Language :: Python :: 3.2',
    #    'Programming Language :: Python :: 3.3',
    #],
    cmdclass=versioneer.get_cmdclass(),
)
