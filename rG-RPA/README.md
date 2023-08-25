# renormalized-Gaussian Random Phase Approximation for Liquid-Liquid Phase Separation of IDPs/IDRs

Core files:
-
* Sequence - class for loading and characterizing a sequence of aminos
* Model - class defining Free Energy and derivatives, with parameter settings
* Xfuncs - support functions in dealing with renormalization factor 'x'
* Solver - class with solving tools like critical point and spinodal/binodal

Evaluation files:
-
- crit_checker - finds critical point for given sequence (from CSV file) and parameter setting
- phase_checker - constructs phase diagram (spinodal, binodal) for given sequence, parameters

Sequence files:
-
- Fisher_seqs - wild-type and phosphorylated seequences from Dan Fisher
- Yamazaki - sequences from Yamazaki paper
- rG_tests - a few sequences for testing
