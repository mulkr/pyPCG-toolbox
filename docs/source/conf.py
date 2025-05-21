from pyPCG import __version__

project = 'pyPCG'
copyright = '2024, Kristóf Müller'
author = 'Kristóf Müller'
release = __version__

extensions = ['sphinx.ext.autodoc','sphinx.ext.napoleon','sphinx.ext.viewcode','sphinx_toolbox.more_autodoc.autotypeddict','enum_tools.autoenum']
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_context = {
    "display_github": True,
    "github_user": "mulkr",
    "github_repo": "pyPCG-toolbox",
    "github_version": "main",
    "conf_py_path": "/docs/source/"
}

html_theme_options = {
    'logo_only': False,
    'vcs_pageview_mode': ''
}

import os
if os.environ.get('READTHEDOCS', None) == 'True':
    import sys
    from unittest.mock import MagicMock
    class Mock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
            return MagicMock()

    MOCK_MODULES = ['hsmmlearn.base']
    for mod_name in MOCK_MODULES:
        sys.modules[mod_name] = Mock()