Basic Convex Shape Optimization
===============================

This tutorial demonstrates how to optimize a convex shape
using a learned convex diffeomorphism.

We assume basic familiarity with PyTorch.

--------------------------------
Problem setup
--------------------------------

We want to learn a convex domain Ω by optimizing a gauge-based
diffeomorphism.

--------------------------------
Code
--------------------------------

.. literalinclude:: ../../examples/isoperimetric.py
   :language: python
   :linenos:


-------------------------------
Adding symmetries
-------------------------------

.. literalinclude:: ../../examples/isoperimetric_symmetries.py
   :language: python
   :linenos:
