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
