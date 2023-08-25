##  Mike Phillips, 8/2/2022
##  Examination of critical points:
##    checking details of phosphorylation site(s), i.e. which specific 'microstate'
##    of 'n' out of 'm' sites gives largest difference in critical temperature?
##  -> check all possibilities ('n' choose 'm'), save highest & lowest results
##  Command Line args:
##   (0) file path to sequence spreadsheet (CSV)
##   (1) number of phosphorylation sites to use
##   (2) directory for saving outputs: list of micro-phos., list of corresponding critical points

import Sequence as S, Solver as V, Model_standard as M
import numpy as np
import itertools as itools
from os import path
import sys

# defaults
head_name = "NAME"
head_seq = "SEQUENCE"
#head_seq = "Seq"
afile = "../rG-RPA_coex/sequences/Fisher_phos_scd.csv"
#afile = "../rG-RPA_coex/sequences/RG_tests.csv"
#afile = "../rG-RPA_coex/sequences/yamazaki.csv"

seqname = "KI-67_cons"
#pars = "cions+v2_fg"
pars = {'cions': 1, 'v2': 4.1887902047863905, 'eps0': 2.0, 'mode': 'rG'}

phos_residue = "X"  # choose amino acid / residue(s) for phosphorylation replacement ('X' or 'DD')
phos_choices = (4, 12, 42, 47, 49, 66, 76, 78, 80, 106, 119)    # list of all choices for phos. sites (indexed at 0)

# command line inputs
args = sys.argv

if len(args) > 1:
    input_file = args.pop(1)
    if len(input_file) > 2:
        afile = input_file

phos_sites = 0
if len(args) > 1:
    phos_sites = int(args.pop(1))

micro_file = f"{seqname}_{phos_sites}_micro.npy"
crits_file = f"{seqname}_{phos_sites}_crits.npy"
SAVE = None
if len(args) > 1:
    SAVE = args.pop(1)

# Fisher fix
if "Fisher" in afile:
    head_seq = "WT Seq"
    alias = None
elif "yamazaki" in afile:
    head_seq = "Seq"
    alias = None
else:
    alias=None

# adjustment for salt
SALT = 0      # salt concentration, in milli-Molar
phi_salt = SALT * (5.5**3) * (6.022e-4) * (1e-3)
if SALT:
    if type(pars) == dict:
        pars.update({"salt":phi_salt})
    else:
        pars = {"cions":1, "v2":(4*M.PI/3), "salt":phi_salt, "eps0":2.0, "mode":"rG"}

## check for dictionary format of parameters
#try:
#    pars = eval(pars)
#except NameError:
#    pars = pars


# common functions to use, given a sequence (object)
def modify_seq(seq, phos):
    aminos = list(seq.aminos)       # get string of aminos
    for p in phos:
        aminos[p] = phos_residue    # modify with each phos. site
    seq.aminos = "".join(aminos)    # re-join as single string
    seq.sigma = seq.translate(seq.aminos)   # get new charge list
    seq.seqAlias = seqname + f"_{phos_sites}.{counter}"
    seq.characterize()      # store important characterizations (N, Qtot, SCD, etc.)
    return seq

def get_crit(seq):
    # get model object (rG-RPA)
    mod = M.RPA(seq, pars)
    mod.info()      # show model information (parameter selection)

    # get solver object (coexistence)
    co = V.Coex(mod, spars={"thr":1e-6})

    # use method to find crit.
    print("Minimizing inverse: u=1/t.\n")
    co.find_crit(phi_bracket=None, u_bracket=(1e-6,1e4), u_bracket_notes=False)

    # below: use 'negative temperature' approach (with brent bracket)
#    print("Minimizing negative: u=-t.\n")
#    co.find_crit_NT(phi_bracket=None, u_bracket=(-1e6,-1e-4), u_bracket_notes=False)

    # store & print
    (pc, tc) = co.get_crit()
    print("\t(phi_c, T_c) = ({:5.5g}, {:5.5g})".format(pc,tc))

    # check that it's really a solution
    print("\nChecking that d2F and d3F are indeed close to zero at (phi_c,T_c):")

    if mod.pars["mode"][0].lower() == "r":
        x = mod.find_x(pc, tc)
        intD = mod.XintD(pc, tc, x)
        dx = mod.find_dx(pc, tc, x, intD)
        d2x = mod.find_d2x(pc, tc, x, dx, intD)
        d3x = mod.find_d3x(pc, tc, x, dx, d2x, intD)
    else:
        x = 1
        dx, d2x, d3x = 0, 0, 0

    d2F = mod.d2F(pc, tc, x, dx, d2x)
    d3F = mod.d3F(pc, tc, x, dx, d2x, d3x)
    print("\td2F(pc,tc) = {:4.3g} ,  d3F(pc,tc) = {:4.3g}\n".format(d2F, d3F))

    # checking values of d2F, d3F _below_ critical point
    Tfac = 4/5
    print("\nChecking d2F and d3F at (phi_c,{:.2f}*Tc):".format(Tfac))

    if mod.pars["mode"][0].lower() == "r":
        x = mod.find_x(pc, Tfac*tc)
        intD = mod.XintD(pc, Tfac*tc, x)
        dx = mod.find_dx(pc, Tfac*tc, x, intD)
        d2x = mod.find_d2x(pc, Tfac*tc, x, dx, intD)
        d3x = mod.find_d3x(pc, Tfac*tc, x, dx, d2x, intD)
    else:
        x = 1
        dx, d2x, d3x = 0, 0, 0

    d2F = mod.d2F(pc, Tfac*tc, x, dx, d2x)
    d3F = mod.d3F(pc, Tfac*tc, x, dx, d2x, d3x)
    print("\td2F(pc,{0:.2f}*tc) = {1:4.3g} ,  d3F(pc,{0:.2f}*tc) = {2:4.3g}\n".format(Tfac, d2F, d3F))
    print(" " + ("- "*20) + "\n")
 
    return (pc,tc)



# get main / starter sequence object (to be modified with given phosphorylations)
main_seq = S.Sequence(seqname, file=afile, alias=alias, headSeq=head_seq, info=False)

# for each phosphorylation micro-state: modify sequence, show info, get critical point, save in list
all_micro_phos = list(itools.combinations(phos_choices, phos_sites))
all_crits = []

counter = 0     # for counting & labelling phos. variants

for phos in all_micro_phos:
    counter += 1
    seq = modify_seq(main_seq.copy(), phos)
    seq.info(showSeq=True)
    (pc,tc) = get_crit(seq)
    all_crits.append((pc,tc))
#    print(seq.aminos)
#    print(main_seq.aminos)


if SAVE:
    np.save(path.join(SAVE, micro_file), all_micro_phos, allow_pickle=True)
    np.save(path.join(SAVE, crits_file), all_crits, allow_pickle=True)
