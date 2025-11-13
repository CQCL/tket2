{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :exclude-members: __annotate_func__
   :undoc-members:
   :special-members:

   {% block methods %}
   {% if methods %}
   {% endif %}
   {% endblock %}
