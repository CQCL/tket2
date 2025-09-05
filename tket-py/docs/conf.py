# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "tket-py"
copyright = "2025, Quantinuum compiler team"
author = "Quantinuum compiler team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_nb",
]

autosummary_ignore_module_all = False  # Respect __all__ if specified
autosummary_generate = True

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


exclude_patterns = ["jupyter_execute/**"]

suppress_warnings = [
    "misc.highlighting_failure",
]


nb_execution_mode = "off"
