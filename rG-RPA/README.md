# renormalized-Gaussian Random Phase Approximation (rG-RPA) for Liquid-Liquid Phase Separation of Intrinsically Disordered Proteins / Regions (IDPs/IDRs)

#### For an overview of the model and its advantages, see: Lin et al. *JCP* 2020. [DOI](https://doi.org/10.1063/1.5139661)

## Core Python files (best to leave these alone):

* `Sequence` - class for loading and characterizing a sequence of aminos
* `Model_standard` - class defining Free Energy and derivatives, with parameter settings
* `Xfuncs_standard` - support functions in dealing with renormalization factor 'x'
* `Solver` - class with solving tools like critical point and spinodal/binodal


## Calculation Python files (you'll likely want to adjust settings there each time):

* `crit_checker` - finds critical point for given sequence (from CSV file) and parameter setting
* `crit_microphos` - finds MANY critical points, for every possible combination of phosphorylatable sites in base (WT) sequence
* `phase_checker` - constructs phase diagram (spinodal, binodal) for given sequence and parameters

## Sequence CSV files (examples to try):

* `Fisher_seqs` - wild-type and phosphorylated seequences from Dan Fisher group
* `Yamazaki` - sequences from Yamazaki paper  (Yamazaki et al. *Nature Cell Biology* 2022. [DOI](https://doi.org/10.1038/s41556-022-00903-1))
* `rG_tests` - a few sequences for testing


# See INSTRUCTIONS for usage notes!