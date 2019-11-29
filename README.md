Searching for MACHOs using FRB Spectra
=======================================================================
Andrey Katz, Joachim Kopp, Sergey Sibiryakov, Wei Xue
arXiv:1912.XXXXX

This repository contains the Python codes used in the above mentioned paper/

Simulation code (frb.ipynb)
----------------------------------------------------------------------

This worksheet simulates scintillation in the IGM/ISM as well as gravitational
lensing by ray-tracing the signal from the source to the observer across an
arbitrary number of scintillation screens (limited by available memory and CPU
power).

Examples for how to use it are given in section "Results".

Further information is given in the first section of the worksheet

Analysis routines (directory data_analysis)
----------------------------------------------------------------------

A. analyze_event -- the main analysis routine. Mandatory input: number of event
in the sequence (integer), the events (python dictionary) and the metadata of the 
events (pandas DataFrame). The routine plots the power spectrum, reconstructed
transfer function, transfered function averaged over a certain amount of bins 
(convolution) and the Fourier transforms thereof.The output of the routine 
is the statistical measure of the peak within 15% from the origin, 
and its distance from the origin for the FFT of the power spectrum, reconstructed 
transfer functions and the convolutions respectively. The last two output are the 
mass of the BH (in solar masses) and log of the diffractive radius (log(cm)).
Switches:
a) smearing -- number of bins to smear over. Python dictionary. The final 
plot will contain the number of curves equal to the length of the dictionary. 
Default: [10]
b) peak width -- integer. Default: 2
c) sigma_cut -- the minimal statistical measure to consider in the peak finding 
procedure. Float, default = 2.
d)produce_plot -- boolean, default -- True
e) output -- weather to save the plot. Default -- False
f) sample -- integer or float, the name of the outputted plot. Default - 1.
Ignored if produce_plot = False. 
g) a_input -- regularization parameter for polynomial regression in peak finding 
routine. Python list. Default [1e-10]. If length of the list bigger than one, 
the statistical measure outputted will be an average of all the values.
h) dim_input -- number of dimensions for polynomial regression in peak finding. 
Python list, default = [15]
i) strategy -- 'ridge', 'lasso' or 'None'. Regularization type in peak finding. 
Default -- 'ridge'
j) verbose -- Default -- True. 
k) subtraction_scheme -- integer, Default is 0. Which procedure to use for 
reconstructing the transfer function  


Licensing and citation information
----------------------------------------------------------------------

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2019
Andrey Katz, Joachim Kopp, Wei Xue
Contact address: jkopp@cern.ch


If you use these codes in a scientific publication, please cite the
following reference:

arXiv:1912.XXXXX


