##  Mike Phillips, 6/10/2022
##  File for checking critical point calculations.
##  > Accepts command line inputs:
##      1. Sequence Name from the CSV file (RG_tests.csv in this case)
##      2. Fisher choice (WT or Phos) Type none if not applicable
##      3. Parameter string
##      4. which model ('standard' or 'selfderivs')
##      5. a file (path to sequence file; CSV)
##      6. directory path for saving critical point
##  > Note: you can expect some adjustment of 'phi_bracket' and 'u_bracket' may be required for each case.

import Sequence as S, Solver as V, Model_standard as M
import sys
import os

# just print info, no real calculations? --> you may want to get info as a first test!
#INFO_ONLY = True
INFO_ONLY = False

# Fisher sequences : wild-type or phosphorylated?   ['None' if not using Fisher seq.]
#Fisher_choice = "WT"
#Fisher_choice = "Phos"
Fisher_choice = None

NO_POS = False       # neglect any positive charges in sequence?
#NO_POS = True
if NO_POS:
    S.aminos.update({"R":0, "K":0})


# defaults
afile = "./RG_tests.csv"        # file with sequence(s)
#afile = "./Fisher_phos.csv"

head_name = "NAME"        # heading in CSV file for sequence name
head_seq = "SEQUENCE"     # heading in CSV file for sequence aminos

seqname = "IP5"    # name of sequence in file (command line overwrites protein name)

#pars = "cions+salt+coulomb_fg"        # using code-string to set parameters

SALT = 100    # salt concentration (milli-Molar)
pars = {'cions': 1, 'salt':M.phi_s(SALT), 'v2': 4.1887902047863905, 'eps0': 2.0, 'mode': 'rG', 'mean-field':False}    # dictionary to set parameters

#pars = {'cions': 1, 'v2': 0, 'eps0': 0, 'mode': 'fG', 'ionsize': 'point', 'potential': 'short'}    # other example


# command line inputs
args = sys.argv
if len(args) > 1:
    seqname = args.pop(1)
if len(args) > 1:
    Fisher_choice = args.pop(1)
    if "none" in Fisher_choice.lower():
        Fisher_choice = None
if len(args) > 1:
    pars = args.pop(1)
if len(args) > 1:
    which_model = args.pop(1)
if len(args) > 1:
    afile = args.pop(1)
if len(args) > 1:
    SaveDir = args.pop(1)
else:
    SaveDir = None


# Fisher fix
if type(Fisher_choice) == str and len(Fisher_choice)> 1:
#    print("TESTING FISHER CHOICE  :  " + Fisher_choice)
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


# check for string-dictionary format of parameters
if type(pars) == str:
    try:
        pars = eval(pars)
    except NameError:
        pars = pars


# get sequence object
seq = S.Sequence(seqname, file=afile, alias=alias, headName=head_name, headSeq=head_seq)

# get model object (rG-RPA)
mod = M.RPA(seq, pars)
mod.info()      # show model information (parameter selection)

# get solver object (coexistence)
co = V.Coex(mod, spars={"thr":1e-6})

# exit now if all you wanted was sequence information
if INFO_ONLY:
    sys.exit()

# find critical point
print("Minimizing inverse: u=1/t.\n")
co.find_crit(phi_bracket=None, u_bracket=(1e-6,1e4), u_bracket_notes=False)

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
    if Fisher_choice:
        outfile = seqname + "_" + Fisher_choice + "_crit.npy"
    else:
        outfile = seqname + "_crit.npy"
    crit_out = os.path.join(SaveDir, outfile)
    V.np.save(crit_out, V.np.asarray(co.get_crit()))
    print("\n>> Crit. saved!\n")

print(" " + ("- "*20) + "\n")

