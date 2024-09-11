project = 'pyPCG'
copyright = '2024, Kristóf Müller'
author = 'Kristóf Müller'
release = '0.1b2'

extensions = ['sphinx.ext.autodoc','sphinx.ext.napoleon','sphinx_toolbox.more_autodoc.autotypeddict','enum_tools.autoenum']
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

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