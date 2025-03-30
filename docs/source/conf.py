# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import sphinx_rtd_theme

project = 'SSVEP-ML'
copyright = '2025, Brynhildr Wu'
author = 'Brynhildr Wu'
release = '0.1'

sys.path.insert(0, os.path.abspath('../../programs'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax'
]

templates_path = ['_templates']
exclude_patterns = [
    'algos_dd.rst',
    'offline_analysis.rst',
    'srca_cpu.rst',
    'srca_test.rst',
    'ssta_latest.rst',
    'stda_cls.rst',
    'tca_cpu.rst',
    'test_algo.rst',
    'toolbox.rst',
    'trca_cuda.rst',
    'utils_cuda.rst'
]

language = 'zh-CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'style_nav_header_background': '#248067',
}
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
master_doc = 'index'
