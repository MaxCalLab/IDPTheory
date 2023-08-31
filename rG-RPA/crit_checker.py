##  Mike Phillips, 6/10/2022
##  File for checking critical point calculations.
##  > Accepts command line inputs:
##      1. Path to CSV file with sequence information (names and aminos)
##      2. Sequence name (as written in CSV file)
##      3. Fisher choice (WT or Phos or PM, or None)
##      4. Directory (path) for saving critical point result
##  > Note: you can expect some adjustment of 'phi_bracket' and 'u_bracket' may be required for each case.

import Sequence as S, Solver as V, Model_standard as M
import sys
from os import path


## IMPORTANT SETTINGS ##

afile = "./RG_tests.csv"    # file with sequence(s)         [override with first argument at command line]
#afile = "./Fisher_phos.csv"

seqname = "IP5"             # name of sequence in file      [override with second argument at command line]


head_name = "NAME"          # heading in CSV file for sequence name
head_seq = "SEQUENCE"       # heading in CSV file for sequence aminos


SALT = 100    # salt concentration (milli-Molar)

pars = {'cions': 1, 'salt': M.phi_s(SALT), 'v2': 4.1887902047863905, 'epsb': 2.0, 'mode': 'rG'}    # DICTIONARY TO SET PARAMETERS

#pars = {'cions': 1, 'salt': M.phi_s(SALT), 'v2': 0, 'epsb': 0, 'mode': 'fG'}    # fG example


## EXTRA SETTINGS ##

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

# formatted filename for saving result
crit_file = f"{seqname}_{Fisher_choice}_crit.npy" if Fisher_choice else f"{seqname}_crit.npy"

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

# get solver object (phase coexistence)
co = V.Coex(mod, spars={"thr":1e-6})

# find critical point
co.find_crit(phi_bracket=None, u_bracket=UBRACKET, u_bracket_notes=False)

(pc, tc) = co.get_crit()
print("\t(phi_c, T_c) = ({:5.5g}, {:5.5g})\n".format(pc,tc))
#print("\t(T_c, phi_c) = ({:5.5g}, {:5.5g})\n".format(tc,pc))

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

# enable saving of critical point as npy array
if SaveDir:
    crit_out = path.join(SaveDir, crit_file)
    V.np.save(crit_out, V.np.asarray(co.get_crit()))
    print(f"\n>> Critical point saved:\n'{crit_out:}'")

print(" " + ("- "*20) + "\n")

