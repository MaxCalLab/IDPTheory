##  Mike Phillips, 6/15/2022
##  File for checking phase curves (spinodal / binodal),
##      in the vicinity of expected critical point.
##  > Accepts command line inputs:
##      1. Sequence
##      2. Parameter string
##      3. Fisher choice (WT or Phos)
##      4. Phi crit. (predetermined)    - optional
##      5. T crit. (predetermined)      - optional

import Sequence as S, Model_standard as M, Solver as V
import numpy as np
from os import path as P
import sys

SHOW = True         # actually show plot?
SAVE_RES = False    # directory for saving results (if desired)
#SAVE_RES = "./out/"

# Fisher sequences : wild-type or phosphorylated?   ['None' if not using Fisher seq.]
#Fisher_choice = "WT"
#Fisher_choice = "Phos"
Fisher_choice = None

NO_POS = False       # neglect any positive charges in sequence
#NO_POS = True
if NO_POS:
    S.aminos.update({"R":0, "K":0})

# defaults
head_name = "NAME"
head_seq = "SEQUENCE"
#head_seq = "Seq"

afile = "./RG_tests.csv"

seqname = "IP5"

pars = "cions+v2_rg"

#SALT = 0
#pars = {"salt":M.phi_s(SALT), "mode":"fg"}

bino_method = {"max":"1d"}

#pc, tc = 0.1, 1.0   # default critical point

t_min_frac = 0.5    # fraction of 'tc' used as minimum temperature
t_points = 5      # number of points in temperature-space for binodal (and spinodal)

# command line inputs
args = sys.argv
if len(args) > 1:
    seqname = args.pop(1)
if len(args) > 1:
    pars = args.pop(1)
if len(args) > 1:
    Fisher_choice = args.pop(1)
if len(args) > 2:
    (pc, tc) = tuple(args[1:])
    (pc, tc) = (float(pc), float(tc))


# Fisher fix
if Fisher_choice:
    if "wt" in Fisher_choice.lower():
        head_seq = "WT Seq"
        alias = "WT"
    elif "phos" in Fisher_choice.lower():
        head_seq = "Phosphorylated Seq"
        alias = "Phos"
    else:
        alias = None
else:
    alias=None


# get sequence object
try:
    seq = S.Sequence(seqname, file=afile, alias=alias, headSeq=head_seq)
except NameError:
    afile = "../rG-RPA_coex/sequences/RG_tests.csv"
    seq = S.Sequence(seqname, file=afile, alias=alias, headSeq=head_seq)


# get model object (rG-RPA)
mod = M.RPA(seq, pars)
mod.info()      # show model information (parameter selection)

try:
    # get solver object (coexistence) -- specify temperature-space (and other) parameters if desired
    co = V.Coex(mod, spars={"t_min":(tc*t_min_frac), "t_points":t_points, "thr":1e-6}, methods=bino_method)
    # save pre-defined critical points in the solver _manually_
    (co.pcrit, co.tcrit) = (pc, tc)
except NameError:
    # get solver object (coexistence) -- specify parameters if desired ['t_min' gets updated from 'find_crit']
    co = V.Coex(mod, spars={"t_min":(0.1), "t_points":t_points, "thr":1e-6}, methods=bino_method)
    co.find_crit(phi_bracket=None, u_bracket=(1e-6,1e4), u_bracket_notes=False, min_frac=t_min_frac)

print("\n\t(pc, tc) = ({:.6e}, {:.6f})".format(co.pcrit, co.tcrit))
print("-  -    "*5)

# evaluate spinodal / binodal solvers
print("\n\nUsing Binodal Method  {}.".format(co.methods))
co.evaluate(UNevenTspace=0.20)

# save results (if desired)
if SAVE_RES:
    file_lab = "_".join((seqname, str(Fisher_choice), str(pars)))
    np.save(P.join(SAVE_RES, file_lab + "_crit.npy"), np.array(co.get_crit()), allow_pickle=True)
    np.save(P.join(SAVE_RES, file_lab + "_Tlist.npy"), np.array(co.get_Teff()), allow_pickle=True)
    np.save(P.join(SAVE_RES, file_lab + "_spino.npy"), np.array(co.get_spino()), allow_pickle=True)
    np.save(P.join(SAVE_RES, file_lab + "_bino.npy"), np.array(co.get_bino()), allow_pickle=True)

# plot results (if desired)
if SHOW:
    co.plot()

