# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

project = 'TorchSparse'
copyright = f'{datetime.datetime.now().year}, MIT HAN Lab.\nDocument author: Xiuyu Li. Modified by: Haotian Tang and Shang Yang.'
author = 'Haotian Tang, Shang Yang, Zhijian Liu, Xiuyu Li, Ke Hong, Zhongming Yu, Yujun Lin, Guohao Dai, Yu Wang, Song Han'
version_file = '../torchsparse/version.py'
with open(version_file) as f:
    exec(compile(f.read(), version_file, 'exec'))
__version__ = locals()['__version__']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
    'myst_parser',
]

add_module_names = False
autodoc_member_order = 'bysource'
autodoc_mock_imports = ['torchvision', 'numpy', 'tqdm', 'typing-extensions']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'torch': ('https://pytorch.org/docs/master', None),
}

templates_path = ['_templates']
exclude_patterns = []  # type: ignore

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'sidebar/scroll-start.html',
        'sidebar/brand.html',
        'sidebar/search.html',
        'sidebar/navigation.html',
        'sidebar/ethical-ads.html',
        'sidebar/scroll-end.html',
    ]
}
html_baseurl = "https://torchsparse-docs.github.io"
