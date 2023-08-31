##  Mike Phillips, 6/15/2022
##  File for checking phase curves (spinodal / binodal),
##      in the vicinity of expected critical point.
##  > Accepts command line inputs:
##      1. Path to CSV file with sequence information (names and aminos)
##      2. Sequence name (as written in CSV file)
##      3. Fisher choice (WT or Phos or PM, or None)
##      4. Directory (path) for saving critical point result

import Sequence as S, Model_standard as M, Solver as V
import numpy as np
from os import path
import sys


## IMPORTANT SETTINGS ##

afile = "./RG_tests.csv"    # file with sequence(s)         [override with first argument at command line]

seqname = "IP5"             # name of sequence in file      [override with second argument at command line]


head_name = "NAME"          # heading in CSV file for sequence name
head_seq = "SEQUENCE"       # heading in CSV file for sequence aminos


SALT = 100    # salt concentration (milli-Molar)

pars = {'cions': 1, 'salt': M.phi_s(SALT), 'v2': 4.1887902047863905, 'epsb': 2.0, 'mode': 'rG'}    # DICTIONARY TO SET PARAMETERS

#pars = {'cions': 1, 'salt': M.phi_s(SALT), 'v2': 0, 'epsb': 0, 'mode': 'fG'}    # fG example


## EXTRA SETTINGS ##

# Show a simple plot of resulting phase diagram?
SHOW = True

# For construction of temperature space
t_min_frac = 0.5    # fraction of 'tc' used as minimum temperature
t_points = 5        # number of points in temperature-space for binodal (and spinodal)      EACH POINT WILL TAKE SOME TIME!
UNevenT = 0.20      # fraction of space near 'tc' with 50% of points    [revert to even spacing with 0 or None]

# If you have pre-calculated critical point, enter (pc,tc)
pc, tc = None, None
#pc, tc = 0.063941, 0.99463

# Fisher sequences : wild-type or phosphorylated?   ['None' if not using Fisher seq.]
#Fisher_choice = "WT"
#Fisher_choice = "Phos"
#Fisher_choice = "PM"
Fisher_choice = None            # [override with third argument at command line]

# Directory for saving results
SaveDir = None                  # [override with fourth argument at command line]


## ADJUSTMENTS FROM COMMAND LINE ##

args = sys.argv
if len(args) > 1:
    afile = args.pop(1)
if len(args) > 1:
    seqname = args.pop(1)
if len(args) > 1:
    Fisher_choice = args.pop(1)
    if "none" in Fisher_choice.lower():
        Fisher_choice = None
if len(args) > 1:
    SaveDir = args.pop(1)


## SPECIAL CASE ADJUSTMENTS ##

# bracket for inverse temperature, u=1/t
UBRACKET = (1e-6,1e4)

# formatted filename for saving results
file_lab = f"{seqname}_{Fisher_choice}_{pars}" if Fisher_choice else f"{seqname}_{pars}"        # common label form
crit_file = f"{file_lab}_crit.npy"      # for critical point
Tlist_file = f"{file_lab}_Tlist.npy"    # for temperature space
spino_file = f"{file_lab}_spino.npy"    # for spinodal results
bino_file = f"{file_lab}_bino.npy"      # for binodal results

# Fisher fix
if type(Fisher_choice) == str and len(Fisher_choice)> 1:
    if "wt" in Fisher_choice.lower():
        head_seq = "WT Seq"
        alias = "WT"
    elif "phos" in Fisher_choice.lower():
        head_seq = "Phosphorylated Seq"
        alias = "Phos"
    elif "pm" in Fisher_choice.lower():
        head_seq = "PhosMimic Seq"
        alias = "PM"
    else:
        alias = None
elif "yamazaki" in afile:
    head_seq = "Seq"
    alias = None
else:
    alias=None


## CALCULATIONS ##

# get sequence object
seq = S.Sequence(seqname, file=afile, alias=alias, headName=head_name, headSeq=head_seq)

# get model object (rG-RPA)
mod = M.RPA(seq, pars)
mod.info()      # show model information (parameter selection)

# binodal calculation method -- best to leave as is  (Maxwell construction, iterated 1D solvers)
bino_method = {"max":"1d"}

# get solver object (phase coexistence), specifying temperature space
if pc and tc:
    # minimum temperature defined at outset
    co = V.Coex(mod, spars={"t_min":(tc*t_min_frac), "t_points":t_points, "thr":1e-6}, methods=bino_method)
    # store pre-defined critical point in the solver _manually_
    (co.pcrit, co.tcrit) = (pc, tc)
else:
    # aribtrary 't_min': minimum temperature is updated by 'find_crit' according to 't_min_frac'
    co = V.Coex(mod, spars={"t_min":(0.1), "t_points":t_points, "thr":1e-6}, methods=bino_method)
    # find and store critical point
    co.find_crit(phi_bracket=None, u_bracket=(1e-6,1e4), u_bracket_notes=False, min_frac=t_min_frac)

print("\n\t(phi_c, T_c) = ({:5.5g}, {:5.5g})".format(co.pcrit, co.tcrit))
print("-  -    "*5)

# evaluate spinodal / binodal solvers
co.evaluate(UNevenTspace=UNevenT)

# save results (if desired)
if SaveDir:
    crit_out = path.join(SaveDir, crit_file)
    Tlist_out = path.join(SaveDir, Tlist_file)
    spino_out = path.join(SaveDir, spino_file)
    bino_out = path.join(SaveDir, bino_file)
    np.save(crit_out, np.array(co.get_crit()), allow_pickle=True)
    np.save(Tlist_out, np.array(co.get_Teff()), allow_pickle=True)
    np.save(spino_out, np.array(co.get_spino()), allow_pickle=True)
    np.save(bino_out, np.array(co.get_bino()), allow_pickle=True)
    print(f"\n>> Critical point saved:\n'{crit_out:}'")
    print(f"\n>> Temperature space saved:\n'{Tlist_out:}'")
    print(f"\n>> Spinodal results saved:\n'{spino_out:}'")
    print(f"\n>> Binodal results saved:\n'{bino_out:}'")

print(" " + ("- "*20) + "\n")

# plot results (if desired)
if SHOW:
    co.plot()

