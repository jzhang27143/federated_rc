.. Resource Constrained Federated Learning documentation master file, created by
   sphinx-quickstart on Wed Oct  7 21:05:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Resource Constrained Federated Learning
=======================================

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Charts
======

The following images show results of a federated simulation with four clients using aggressive parameter thresholding (with parameters thresholded at 1e-1) with i.i.d. data distribution. Parameter thresholding maintains an accuracy above 95% while reducing model size by over 80%.

.. figure:: _static/param_thresh_iid_1.jpg
   :scale: 85 %
   :alt: Server RX during sample run using parameter thresholding (i.i.d. distribution).

   Fig 1. Server RX during sample run using parameter thresholding (i.i.d. distribution).

.. figure:: _static/param_thresh_iid_2.jpg
   :scale: 85 %
   :alt: Loss and accuracy during sample run using parameter thresholding (i.i.d. distribution).

   Fig 2. Loss and accuracy during sample run using parameter thresholding (i.i.d. distribution)

.. figure:: _static/param_thresh_iid_3.jpg
   :scale: 85 %
   :alt: Loss and accuracy during sample run using parameter thresholding (i.i.d. distribution)

   Fig 3. Loss and accuracy during sample run using parameter thresholding (i.i.d. distribution)

.. figure:: _static/param_thresh_iid_4.jpg
   :scale: 85 %
   :alt: Loss and accuracy during sample run using parameter thresholding (i.i.d. distribution)

   Fig 4. Loss and accuracy during sample run using parameter thresholding (i.i.d. distribution)

.. figure:: _static/param_thresh_iid_5.jpg
   :scale: 85 %
   :alt: Loss and accuracy during sample run using parameter thresholding (i.i.d. distribution)

   Fig 5. Loss and accuracy during sample run using parameter thresholding (i.i.d. distribution)

The following plots show the transmission and training histories of four clients (two trained with 4x more digits from 0-4 than 5-9 and two trained with 4x more digits from 5-9 than 0-4). The client models are all able to maintain at least 95% accuracy. This simulation shows that the system is robust enough to effectively aggregate parameter vectors learned from completely different data.

.. figure:: _static/param_thresh_non_iid_1.jpg
   :scale: 85 %
   :alt: Server RX during sample run using parameter thresholding (non i.i.d. distribution).

   Fig 6. Server RX during sample run using parameter thresholding (non i.i.d. distribution).

.. figure:: _static/param_thresh_non_iid_2.jpg
   :scale: 85 %
   :alt: Loss and accuracy during sample run using parameter thresholding (non i.i.d. distribution).

   Fig 7. Loss and accuracy during sample run using parameter thresholding (non i.i.d. distribution)

.. figure:: _static/param_thresh_non_iid_3.jpg
   :scale: 85 %
   :alt: Loss and accuracy during sample run using parameter thresholding (non i.i.d. distribution)

   Fig 8. Loss and accuracy during sample run using parameter thresholding (non i.i.d. distribution)

.. figure:: _static/param_thresh_non_iid_4.jpg
   :scale: 85 %
   :alt: Loss and accuracy during sample run using parameter thresholding (non i.i.d. distribution)

   Fig 9. Loss and accuracy during sample run using parameter thresholding (non i.i.d. distribution)

.. figure:: _static/param_thresh_non_iid_5.jpg
   :scale: 85 %
   :alt: Loss and accuracy during sample run using parameter thresholding (non i.i.d. distribution)

   Fig 10. Loss and accuracy during sample run using parameter thresholding (non i.i.d. distribution)