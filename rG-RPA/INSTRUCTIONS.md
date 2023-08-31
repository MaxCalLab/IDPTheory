# How to use this code

### Do Not Modify Core Files:

`Sequence`, `Model_standard`, `Xfuncs_standard`, `Solver`


## To calculate something, use these files:

* `crit_checker` - finds critical point for given sequence and model settings  
* `crit_microphos` - finds MANY critical points, for every possible combination of phosphorylatable sites in base sequence (WT)  
* `phase_checker` - calculates phase diagram (spinodal, binodal) for given sequence and model settings


### For any calculation, you should modify the code to specify:

* `afile` - CSV file where sequences are kept (containing a header row, followed by reference names and amino acid sequences)
* `seqname` - Name of the sequence of interest (as listed in the CSV file)
* `head_name` & `head_seq` - Heading labels correponding to the CSV file, for 'name' and 'sequence'
* `SALT` - Salt concentration, in milli-Molar
* `pars` - Any other model settings / parameter choices, such as 'renormalized-Gaussian' mode  __(see below for details)__
* `Fisher_choice` - (Optional) Only if using a sequence from 'Fisher_phos.csv' -- choice of sequence modification, 'WT' (wild-type), 'Phos' (phosphorylated), 'PM' (phospho-mimic); otherwise 'None'
* `SaveDir` - (Optional) Directory for saving results, if desired


### Command line arguments:

Some settings listed above can be given from the command line!  
Example for order of arguments: `python crit_checker.py 'afile' 'seqname' 'Fisher_choice' 'SaveDir'`  
Note that other settings (CSV heading labels, SALT and other model parameters) must always be set within the Python file.


### See notes on model parameters and file-specific settings below!

---

## Model settings / parameters you may want to adjust

#### Set `pars` as a dictionary with as many settings as you desire. Anything not specified will use a default value.

The settings you likely want to adjust are listed here, with default values.  
(The full list of settings is described in the last section of this document.)

* `salt`: _Salt concentration, in dimensionless form._ __[default: 0]__ Can be zero or positive. __Recommended:__ within code, set the variable `SALT` to your concentration in milli-Molar, and let the function `M.phi_s` convert to dimensionless form.
* `cions`: _Valence charge of counter-ions._ __[default: None]__ Can be any number, most commonly '1'. Use '0' or 'None' to disable counter-ions.
* `mode`: _Gaussian chain setting within RPA._ __[default: fG]__ Use 'fixed-Gaussian' or 'fG' for simple Gaussian chain (Kuhn length factor x=1). Use 'renormalized-Gaussian' or 'rG' for renormalized Gaussian chain (self-consistently determined x≠1).
* `v2`: _Volume exclusion for monomers, in units of cubic Kuhn length._ __[default: 0]__ Can be zero or positive. __Important:__ __MUST be set to a positive value if using 'renormalized-Gaussian' mode!__ Typically '0' with 'fixed-Gaussian' mode, '4.189' (4π/3, hard sphere) with 'renormalized-Gaussian' mode.
* `epsb`: _Flory-Huggins interaction piece._ __[default: 0]__ Should be zero or positive. Introduces a typical Flory-Huggins interaction term, chi = epsb / (l/lB) ~ epsb / T.


## File-specific settings you should check

__The above settings apply to all calculation files (`crit_checker`, `crit_microphos`, `phase_checker`).__  
__However, you should at least check the values of the following file-specific settings; you may want to adjust them to suit your needs.__

* __`crit_checker`__
  - __Generally: if you've addressed all points above, you should be good to go!__
  - `UBRACKET` - In some exceptional cases, you may need to adjust this bracket. It tells the solver two points of inverse temperature value (u=1/t=l/lB) defining the region where you expect to find points of zero curvature in Free Energy. If you make this very extreme, like `(1e-8, 1e8)`, and still cannot find your critical point, then it probably doesn't exist, i.e. the system never phase separates!
 
* __`crit_microphos`__
  - `phos_sites` - Number of phosphorylation sites to use. If zero, use the unmodified (WT) sequence. Otherwise, all combinations of the number of sites are calculated.
  - `phos_residue` - Residue to substitute in place of a phosphorylation site. This can be a singlular, such as 'D', or multiple, as 'DD'. The typical choice is 'X', which models a single residue of charge -2.
  - `phos_choices` - List of phosphorylatable sites in your sequence (indexed from 0). Should only be modified if you are analyzing a sequence other than 'KI-67_cons', or you want to include/exclude some sites for specific investigation.
  - `UBRACKET` - May require adjustment in exceptional cases, as noted above.

* __`phase_checker`__
  - `SHOW` - Set to 'True' if you want to see a plot of resulting spinodal and binodal with critical point. __Note: if this is 'False' and `SaveDir` is 'None', you will lose results!__ Best practice is to always set `SaveDir` so that results are saved somewhere, and can always be plotted later.
  - `t_min_frac` - Fraction defining the minimum teperature to calculate. Value of '0.5' means spinodal and binodal curves will be calculated from the critical temperature down to 0.5 times the critical temperature. This is often adjusted once the critical temperature and general phase behavior are known, so that you stay reasonably close to the critical point.
  - `t_points` - Number of points calculated for spinodal and binodal. __Each point can take considerable time!__ For benchmarking, a reasonable number is 5. For final results, a typical number is 30.
  - `UNevenT` - Fraction of temperature region near critical point where half of the points are calculated. Value of '0.20' means that half of your `t_points` (rounded down) are placed in the top 20% of temperature space, defined by `[t_min_frac*tc, tc]`. The purpose is to allow greater detail in the vicinity of the critical point, where curvature is greatest. Can set to '0' or 'None' to keep points evenly spaced in temperature.
  - `pc` & `tc` - (Optional) Critical point volume fraction (phi) and temperature (l/lB), if known. By default, the critical point will be calculated before spinodal and binodal are calculated.
  - `UBRACKET` - May require adjustment in exceptional cases, as noted above.

---

## Description of all available parameter settings

#### This is the full list of parameters you can set in the `pars` dictionary.

Some are redundant (noted above, under "parameters you may want to adjust").

__You probably don't want these details, but they are available to play with.__  
__Also, you will see the complete list of settings when running any calculation code, e.g.__

        PAR             VALUE
        cions           1
        salt            0.0033044
        v2              4.1888
        epsa            0
        epsb            2
        chi3            0
        potential       coulomb
        ionsize         smear
        mode            rG
        mean-field      0

* `cions`: _Valence charge of counter-ions._ __[default: None]__ Can be any number, most commonly '1'. Use '0' or 'None' to disable counter-ions.
* `salt`: _Salt concentration, in dimensionless form._ __[default: 0]__ Can be zero or positive. __Recommended:__ within code, set the variable `SALT` to your concentration in milli-Molar, and let the function `M.phi_s` convert to dimensionless form.
* `v2`: _Volume exclusion for monomers, in units of cubic Kuhn length._ __[default: 0]__ Can be zero or positive. __Important:__ __MUST be set to a positive value if using 'renormalized-Gaussian' mode!__ Typically '0' with 'fixed-Gaussian' mode, '4.189' (4π/3, hard sphere) with 'renormalized-Gaussian' mode.
* `epsa`: _Flory-Huggins interaction, constant offset_. __[default: 0]__ Should be zero or positive. Introduces a constant Flory-Huggins interaction term, chi = epsa.
* `epsb`:  _Flory-Huggins interaction, temperature-dependent._ __[default: 0]__ Should be zero or positive. Introduces a typical Flory-Huggins interaction term, chi = epsb / (l/lB) ~ epsb / T.  Total interaction is: CHI = epsa + epsb/(l/lB).
* `chi3`: _Three-body Flory-Huggins interaction, constant._ __[default: 0]__ Should be greater than -1/6. Introduces a Free Energy term, chi3*phi^3.
* `potential`: _Form of electrostatic potential._ __[default: coulomb]__ Can be either 'coulomb' or 'short'. Choice 'coulomb' is standard coulomb interaction with screening from ions (salt and counter-ions). Choice 'short' dresses coulomb with a short-range cutoff (cf. Lin et al. *PRL* 2016. [DOI](https://doi.org/10.1103/PhysRevLett.117.178101)).
* `ionsize`: _Effective size of ions (salt and counter-ions)_. __[default: smear]__ Can be either 'point' or 'smear'. Use 'point' to treat ions as true points (no spatial extent). Use 'smear' to apply Gaussian smearing, giving the ions a finite size. This choice impacts only the ionic contribution to Free Energy, and has rather small effects on calculations.
* `mode`: _Gaussian chain setting within RPA._ __[default: fG]__ Use 'fixed-Gaussian' or 'fG' for simple Gaussian chain (Kuhn length factor x=1). Use 'renormalized-Gaussian' or 'rG' for renormalized Gaussian chain (self-consistently determined x≠1).
* `mean-field`: _Inclusion of mean-field electrostatics._ __[default: False]__ Boolean choice, True/False (or 1/0). If True (or 1), an additional term is added to the Free Energy (~phi^2), which captures the polymer-polymer interaction due to its net charge. Only applicable if using Coulomb potential. Also, it only has a meaningful effect at nonzero salt.
