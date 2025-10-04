# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'migo'
copyright = '2024-2025, games-research-komaba'
author = 'games-research-komaba'
release = 'r0-1-4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'breathe',
]

breathe_projects = {
    "cygo_cc": '../xml',
}
breathe_default_project = "cygo_cc"
breathe_domain_by_extension = {
    "hpp": "cpp",
}
breathe_show_enumvalue_initializer = True

doctest_global_setup = 'import migo, migo.sgfutils, cygo, migo.features, migo.drawing, migo.record, io'

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'agogo'  # 'alabaster'
html_static_path = ['_static']
