# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'track2trajectory'
copyright = '2022, Thejasvi Beleyur'
author = 'Thejasvi Beleyur'

# The full version, including alpha/beta/rc tags
release = '0.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon',
				'sphinx.ext.viewcode','recommonmark','sphinx.ext.mathjax',
				'sphinx_gallery.gen_gallery']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# Keep the original order of functions in each module 
autodoc_member_order = 'bysource'   

# Keep the original order of functions in each module 
autodoc_member_order = 'bysource'

# also document the __init__ of classes
autodoc_special_members = ['__init__']


# MOCK THE INSTALLATION OF THESE PACKAGES  becaue RTD doesn't allow C packages
autodoc_mock_imports = ["cv2"]


# sphinx gallery
from sphinx_gallery.sorting import FileNameSortKey
sphinx_gallery_conf = {
     'gallery_dirs' : ['gallery_examples'],
     'examples_dirs': ['../../examples'],  # path to where to save gallery generated output
	 'image_scrapers': ('matplotlib'),
	 'within_subsection_order': FileNameSortKey
						}