# Sequence Charge Decoration Matrix (SCDM) calculation

#### Defined in: Sawle and Ghosh. *JCP* 2015. [DOI](http://dx.doi.org/10.1063/1.4929391)

#### Also see: Ghosh et al. *Annu. Rev. Biophys.* 2022. 51:355–76. [DOI](https://doi.org/10.1146/annurev-biophys-120221-095357)

## Main files:

* `SCDMmap` - CPU: calculates SCDM using given sequence of amino acids, allows for saving output array and/or plotting
* `SCDM_opencl` - GPU: calculates SCDM with package `pyOpenCL`

## Support file:

* `SCDM_program` - Kernel for application in GPU via pyOpenCL (C-based).

## Sequence CSV file (examples to try):

* `SVset` - simple EK sequences as examples, from: Das and Pappu. *PNAS* 2013. [DOI](www.pnas.org/cgi/doi/10.1073/pnas.1304749110). also: Das et al. *PCCP* 2018. [DOI](https://doi.org/10.1039/c8cp05095c))


# See INSTRUCTIONS for usage notes!