   Custom class template to make sphinx-autosummary list the full API doc after
   the summary. See https://github.com/sphinx-doc/sphinx/issues/7912

{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :special-members:

   {% block methods %}
   {% if methods %}
   {% endif %}
   {% endblock %}
