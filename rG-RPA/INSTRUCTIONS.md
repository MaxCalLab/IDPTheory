# How to use this code

Quick Start:
_
* crit_checker
  - point to correct file for sequences
  - select sequence and set file headings
  - set parameters
    * Dictionary with optional keys (described below), i.e. pars={"mode":"fixed-G", ...} \
        Na: length of molecule type A (small) [solvent]
        cions: choice of counter-ions (valence charge 'zc', or 'None')
        salt: choice of salt concentration, dimension-less
        zs: choice of salt valence charge (only if 'salt' is given)
        v2: volume exclusion, dimensionless (units of Kuhn length cube)
        epsa: Flory-Huggins interaction piece, i.e. chi = epsa  (zero or positive)
        epsb: Flory-Huggins interaction piece, i.e. chi = epsb / (l/lB)  (zero or positive)
        chi3: 3-body Flory-Huggins interaction piece (constant)
        potential: choice of electrostatic potential (among ions & polymer); either 'coulomb' or 'short'
        ionsize: effective size of ions (salt & counterions), either 'point' [point-like] or 'smear' [Gaussian smearing]
        mode: either 'fG' (fixed Gaussian, i.e. factor x=1), or 'rG' (renormalized Gaussian, i.e. use solver for x)
        Xthr: threshold value to end iteration early (i.e. convergence)
        mean-field: boolean flag for using mean field electrostatics contribution (from net charge of sequence)
  - run!

* phase_checker
  - similar settings to above
  - run!

Details:
_
