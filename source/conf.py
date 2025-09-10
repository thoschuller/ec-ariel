# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Ariel'
copyright = '2025, CI Group'
author = 'CI Group'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", 
              "sphinx.ext.apidoc", 
              "sphinx.ext.napoleon", 
              "sphinx.ext.viewcode",
              "sphinx.ext.doctest",
              "sphinx.ext.duration",
              "sphinx.ext.autosectionlabel",
              "myst_parser",]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_logo = "../resources/ariel_logo_1.png"
html_favicon = "../resources/ariel_logo_1.png"
html_static_path = ['_static']
