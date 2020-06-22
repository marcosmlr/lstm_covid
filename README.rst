LSTM COVID-19
========================

This project contains application source code to predict daily cases of COVID-19 using Long Short-Term Models. Basically, the package has a main module (core) and submodules (lstm_covid and forecasting) with functions to tunning models and predict daily cases using time series data of COVID pandemic.

Release Notes
-------------

- Support data files of Comma Separated Values (CSV) with collums int this order (Region, COD, Date, Daily Cases)

Installation
------------

**Dependencies**

    Python 3.7.X, Numpy, Pandas, Keras, Scikit-Learn and Matplotlib
    

Build Steps
-----------

**Setup Conda Environment** 

With Conda installed [#]_, run::

  $ git clone  https://github.com/marcosmlr/lstm_covid.git
  $ cd lstm_covid
  $ make install
  $ source activate lstm

.. [#] If you are using a git server inside a private network and are using a self-signed certificate or a certificate over an IP address ; you may also simply use the git global config to disable the ssl checks::

  git config --global http.sslverify "false"


Usage
-----

See core.py --help for command line details.


Data Processing Requirements
----------------------------

This version of the application requires the input files to be in the Comma Separated Values (CSV) format.


Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The software has not received final approval by the National Institute for Space Research (INPE). No warranty, expressed or implied, is made by the INPE or the Brazil Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. The software is provided on the condition that neither the INPE nor the Brazil Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.


Licence
-------

MIT License

Copyright (c) 2020 Luis Ricardo and Marcos Rodrigues

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Authors
-------

`lstm_covid` was written by `Luis Ricardo <luisricardoengcomp@gmail.com>`_ and `Marcos Rodrigues <marcos.rodrigues@inpe.br>`_.
