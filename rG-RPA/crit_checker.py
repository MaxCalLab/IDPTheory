##  Mike Phillips, 6/10/2022
##  File for checking critical point calculations.
##  > Accepts command line inputs:
##      1. Sequence
##      2. Fisher choice (WT or Phos)
##      3. Parameter string
##      4. which model ('standard' or 'selfderivs')
##      5. a file (path to sequence file; CSV)
##      6. directory path for saving critical point
##  > Note: you can expect some adjustment of 'phi_bracket' and 'u_bracket' is required for each case.

# Make sure to import the intended 'Model' !!  [standard self-energy corr., better corr., or 'a priori' approach]
import Sequence as S, Solver as V   #, Model_standard as M
import sys
import os

# just print info, no real calculations? --> you may want to get info as a first test!
#INFO_ONLY = True
INFO_ONLY = False

# Fisher sequences : wild-type or phosphorylated?   ['None' if not using Fisher seq.]
#Fisher_choice = "WT"
#Fisher_choice = "Phos"
Fisher_choice = None

NO_POS = False       # neglect any positive charges in sequence
#NO_POS = True
if NO_POS:
    S.aminos.update({"R":0, "K":0})

OLD_CRIT = False    # use _old_ critical point finder?  [simultaneous zero of d2F and d3F]
#OLD_CRIT = True

# defaults
head_name = "NAME"
head_seq = "SEQUENCE"
#head_seq = "Seq"
#afile = "../rG-RPA_coex/sequences/Fisher_phos_scd.csv"
#afile = "../RPA_coex/sequences/Fisher_phos_scd_NEW-FIX.csv"
afile = "../RPA_coex/sequences/Fisher_phos_scd_NEW-FINAL.csv"
#afile = "../rG-RPA_coex/sequences/Fisher_KI67_repeats_xcheck.csv"
#afile = "../rG-RPA_coex/sequences/RG_tests.csv"
#afile = "../rG-RPA_coex/sequences/yamazaki.csv"
seqname = "KI-67_cons"
#pars = "cions+v2_fg"
pars = {'cions': 1, 'v2': 4.1887902047863905, 'eps0': 2.0, 'mode': 'rG', 'mean-field':False}
#pars = {'cions': 2, 'v2': 0, 'eps0': 0, 'mode': 'fG', 'ionsize': 'point', 'potential': 'short'}
#pars = "cions+short+point_fg"
which_model = "standard"


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

#print("TESTING HEADING  :  " + head_seq)

# load proper model
which_model = which_model.lower()
if which_model == "standard":
    import Model_standard as M
elif which_model == "selfderivs":
    import Model_selfderivs as M
elif which_model == "noself":
    import Model_noself as M

# adjustment for salt
#SALT = 100      # salt concentration, in milli-Molar
SALT = 0
#phi_salt = SALT * (5.5**3) * (6.022e-4) * (1e-3)        # using Kuhn length
phi_salt = SALT * (3.8**3) * (6.022e-4) * (1e-3)        # using bond length
if SALT:
    pars = str({"cions":1, "v2":(4*M.PI/3), "salt":phi_salt, "eps0":2.0, "mode":"rG"})

# check for dictionary format of parameters
if type(pars) == str:
    try:
        pars = eval(pars)
    except NameError:
        pars = pars

#t1 = M.perf_counter()
# get sequence object
seq = S.Sequence(seqname, file=afile, alias=alias, headName=head_name, headSeq=head_seq)
#seq.info(showSeq=True)
#t2 = M.perf_counter()
#print("\nSEQUENCE LOADING TIMER : {}\n".format(t2-t1))

if INFO_ONLY:
#    alpha = 1.18        # hydrophobics and aromatics
#    beta = 19.78        #
    alpha , beta = 2.36 , 0     # just hydrophobics
    e = alpha*seq.frac_hp + beta*seq.fracp*seq.frac_ar
    print("-> EXPECT eps0 = {:.5f}\n\n".format(e))
    sys.exit()

# get model object (rG-RPA)
mod = M.RPA(seq, pars)
mod.info()      # show model information (parameter selection)

# get solver object (coexistence)
co = V.Coex(mod, spars={"thr":1e-6})

# exit now if all you wanted was sequence information
if INFO_ONLY:
#    k2 = 2
#    print("\nCorrelation / Structure values (at k2={:.3f}):".format(k2))
#    print("\tG0 -> {:.5f}".format(mod.realG(k2, 1, mod.range0)))
#    print("\tg0 -> {:.5f}".format(mod.realMinig(k2, 1, mod.range0)))
#    print("\tZ0 -> {:.5f}".format(mod.realZ(k2, 1, mod.range0)))
#    print("\tG2 -> {:.5f}".format(mod.realG(k2, 1, mod.range2)))
#    print("\tg2 -> {:.5f}".format(mod.realMinig(k2, 1, mod.range2)))
#    print("\tZ2 -> {:.5f}".format(mod.realZ(k2, 1, mod.range2)))
#    print("\n")
    sys.exit()

# find critical point
if OLD_CRIT:
#    co.OLD_find_crit()
    co.OLD_find_crit(XintKW={"big":100.})
else:
#    PBrack = (1e-2/(seq.N**1.6),0.01/(seq.N**0.4), (0.1+0*(seq.N**(-0.08)))/(seq.N**0.08))
#    PBrack = (M.np.exp(-seq.N**0.45),0.01/(seq.N**0.5), (0.1+0*(seq.N**(-0.08)))/(seq.N**0.08))
#    PBrack = (1e-3, 4.9e-3, 1e-2)
#    co.find_crit(phi_bracket=PBrack, u_bracket=(1e-4,1e4), u_bracket_notes=False)
    print("Minimizing inverse: u=1/t.\n")
    co.find_crit(phi_bracket=None, u_bracket=(1e-6,1e4), u_bracket_notes=False)
    # below: use 'negative temperature' approach (with brent bracket)
#    print("Minimizing negative: u=-t.\n")
#    co.find_crit_NT(phi_bracket=None, u_bracket=(-1e6,-1e-4), u_bracket_notes=False)

(pc, tc) = co.get_crit()
print("\t(phi_c, T_c) = ({:5.5g}, {:5.5g})\n".format(pc,tc))
#print("\t(T_c, phi_c) = ({:5.5g}, {:5.5g})\n".format(tc,pc))

# check that it's really a solution
print("\nChecking that d2F and d3F are indeed close to zero at (phi_c,T_c):")

#mod.info()

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

