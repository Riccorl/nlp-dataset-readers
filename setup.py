"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py
To create the package for pypi.
1. Run `make pre-release` (or `make pre-patch` for a patch release) then run `make fix-copies` to fix the index of the
   documentation.
2. Run Tests for Amazon Sagemaker. The documentation is located in `./tests/sagemaker/README.md`, otherwise @philschmid.
3. Unpin specific versions from setup.py that use a git install.
4. Commit these changes with the message: "Release: VERSION"
5. Add a tag in git to mark the release: "git tag VERSION -m 'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master
6. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).
   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.
7. Check that everything looks correct by uploading the package to the pypi test server:
   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers
8. Upload the final version to actual pypi:
   twine upload dist/* -r pypi
9. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.
10. Run `make post-release` (or `make post-patch` for a patch release).
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ["conllu", "rich"]

extras_require = {}

setuptools.setup(
    name="nlp-dataset-readers",
    version="0.1.7",
    author="Riccardo Orlando",
    author_email="orlandoricc@gmail.com",
    description="Dataset Readers for NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Riccorl/nlp-dataset-readers",
    keywords="NLP Sapienza sapienzanlp deep learning transformer pytorch datasets data reader readers dataset",
    packages=setuptools.find_packages(),
    include_package_data=True,
    license="Apache",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.6",
)
