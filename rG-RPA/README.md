# renormalized-Gaussian Random Phase Approximation for Liquid-Liquid Phase Separation of IDPs/IDRs

Core files (probably don't mess with these):
-
* Sequence - class for loading and characterizing a sequence of aminos
* Model - class defining Free Energy and derivatives, with parameter settings
* Xfuncs - support functions in dealing with renormalization factor 'x'
* Solver - class with solving tools like critical point and spinodal/binodal

Evaluation files (you'll likely want to adjust settings each time):
-
- crit_checker - finds critical point for given sequence (from CSV file) and parameter setting
- crit_microphos - finds MANY critical points, for every possible combination of phosphorylatable sites in base (WT) sequence
- phase_checker - constructs phase diagram (spinodal, binodal) for given sequence and parameters

Sequence files (examples to try, or can use your own CSV formatted similarly):
-
- Fisher_seqs - wild-type and phosphorylated seequences from Dan Fisher
- Yamazaki - sequences from Yamazaki paper
- rG_tests - a few sequences for testing
