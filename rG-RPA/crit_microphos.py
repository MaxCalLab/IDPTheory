##  Mike Phillips, 8/2/2022
##  Examination of critical points:
##    checking details of phosphorylation site(s), i.e. which specific 'microstate'
##    of 'n' out of 'm' sites gives largest difference in critical temperature?
##  -> check all possibilities ('n' choose 'm'), save highest & lowest results
##  Command Line args:
##      1. Path to CSV file with sequence information (names and aminos)
##      2. Sequence name (as written in CSV file)
##      3. Fisher choice (WT or Phos or PM, or None)
##      4. Directory (path) for saving critical point result

import Sequence as S, Solver as V, Model_standard as M
import numpy as np
import itertools as itools
from os import path
import sys


## IMPORTANT SETTINGS ##

afile = "./Fisher_phos.csv"     # file with sequence(s)     [override with first argument at command line]

seqname = "KI-67_cons"      # name of sequence in file      [override with second argument at command line]


head_name = "NAME"          # heading in CSV file for sequence name
head_seq = "SEQUENCE"       # heading in CSV file for sequence aminos


SALT = 100    # salt concentration (milli-Molar)

pars = {'cions': 1, 'salt': M.phi_s(SALT), 'v2': 4.1887902047863905, 'epsb': 2.0, 'mode': 'rG'}    # DICTIONARY TO SET PARAMETERS

#pars = {'cions': 1, 'salt': M.phi_s(SALT), 'v2': 0, 'epsb': 0, 'mode': 'fG'}    # fG example


## EXTRA SETTINGS ##

# How to handle phosphorylation
phos_sites = 1      # number of sites to phosphorylate
phos_residue = "X"  # choose amino acid / residue(s) for phosphorylation replacement ('X' or 'DD')
phos_choices = (4, 12, 42, 47, 49, 66, 76, 78, 80, 106, 119)    # list of all choices for phos. sites (indexed from 0)

# Fisher sequences : wild-type or phosphorylated?   ['None' if not using Fisher seq.]
Fisher_choice = "WT"            # you probably want 'WT' in this case       [override with third argument at command line]
#Fisher_choice = "Phos"
#Fisher_choice = "PM"
#Fisher_choice = None

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

# formatted filenames for saving results
micro_file = f"{seqname}_{phos_sites}_micro.npy"        # for list of all phospho-site combinations
crits_file = f"{seqname}_{phos_sites}_crits.npy"        # for list of corresponding critical points as (phi_c,T_c)

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

# function to modify sequence with given phosphorylation site(s)
def modify_seq(seq, phos):
    aminos = list(seq.aminos)       # get string of aminos
    for p in phos:
        aminos[p] = phos_residue    # modify with each phos. site
    seq.aminos = "".join(aminos)    # re-join as single string
    seq.sigma = seq.translate(seq.aminos)   # get new charge list
    seq.seqAlias = seqname + f"_{phos_sites}.{counter}"
    seq.characterize()      # store important characterizations (N, Qtot, SCD, etc.)
    return seq

# function for finding a critical point of a given sequence object
def find_crit(seq):
    # get model object (rG-RPA)
    mod = M.RPA(seq, pars)
    mod.info()      # show model information (parameter selection)

    # get solver object (phase coexistence)
    co = V.Coex(mod, spars={"thr":1e-6})

    # use method to find crit.
    co.find_crit(phi_bracket=None, u_bracket=UBRACKET, u_bracket_notes=False)

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
main_seq = S.Sequence(seqname, file=afile, alias=alias, headName=head_name, headSeq=head_seq, info=False)

# for each phosphorylation micro-state: modify sequence, show info, get critical point, save in list
all_micro_phos = list(itools.combinations(phos_choices, phos_sites))
all_crits = []

counter = 0     # for counting & labelling phos. variants

for phos in all_micro_phos:
    counter += 1
    seq = modify_seq(main_seq.copy(), phos)
    seq.info(showSeq=True)
    (pc,tc) = find_crit(seq)
    all_crits.append((pc,tc))


if SaveDir:
    micro_out = path.join(SaveDir, micro_file)
    crits_out = path.join(SaveDir, crits_file)
    np.save(micro_out, all_micro_phos, allow_pickle=True)
    np.save(crits_out, all_crits, allow_pickle=True)
    print(f"\n>> Micro-phos site combinations saved:\n'{micro_out:}'")
    print(f"\n>> Micro-phos critical points saved:\n'{crits_out:}'")

print(" " + ("- "*20) + "\n")

