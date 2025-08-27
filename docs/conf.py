"""Sphinx configuration."""

project = "Ariel"
author = "jmdm"
copyright = "2025, jmdm"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "shibuya"
