##  Mike Phillips, 6/25/2024
##  Single-chain model for end-to-end factor 'x', cf. Ree^2 = N*l*b*x
##   alongside degrees of ionization 'alp' 'alm'  (fraction of charge on +/- ions).
##  Class / Object definition - encapsulating all functions
##  Each Model object holds a Sequence.
##      > with capability to calculate Free Energy, minimization for triplet ('alp','alm','x'), etc.
##      > holds all necessary parameters, and enables (re-)setting them as desired


import numpy as np
from scipy.special import erfcx as SCIerfcx     # SCALED complimentary error function: erfcx(u)=exp(u^2)*erfc(u)
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.pyplot import cm            # colormap shortcut
import myPlotOptions as mpo
from time import perf_counter       # for timing


##  Global Data Type
DTYPE = np.float64
PI = DTYPE(np.pi)       # appropriate value for 'pi'
erfcx = lambda u: SCIerfcx(u, dtype=DTYPE)

# GLOBAL settings for intrinsic polars
Pint_def = 3.8*(0.5)*(0.5)      # default dipole moment for intrinsic polars: half charges separated by half bond length
#polars = {"B":1}       # fictitious polar (neutral charge) amino acid
#polars = {"S":1, "Y":1, "Q":1, "N":1, "T":1}      # yes: ser tyr gln asn ; maybe possible: cys gly thr , trp his
polars = {}       # blank dictionary - to neglect intrinsically polar residues

pol_msg = "_NOT_" if (len(polars) == 0) else "_INDEED_"
pol_secondl = "" if (len(polars) == 0) else "\tpolars = {polars:},  Pint = {Pintrinsic:}\n"
print(f"\nFROM 'xModel' MODULE:\n\tyou are {pol_msg:} including intrinsic polars!")
print(pol_secondl)


# dictionary of _all_ default parameter settings; any/all can be overridden as desired with 'setPars'
#   > if including intrinsic polars: must specify the intrinsic dipole moment
#   > Pint = d*q  ['d' in units matching 'l', 'q' in units of elementary charge]
#   > concentrations 'cs' (salt) and 'cp' (protein/polymer) should be in units 1/[l^3]  : if milli-Molar, use *6.022e-7 for cubic Angstrom
#   > dipole size 'p' involves pair separation distance and ion pair charge  [units of 'l', and elementary charge]
#   > dielectric mismatch 'delta' is dimensionless by definition : ratio of (dim.less) dielectric constants, eps_water/eps_local
default_pars = {'l':3.8, 'lB':7.12, 'cs':0, 'cp':6e-7, 'w2':0, 'w3':0.1, 'p':1.9, 'delta':1.3, 'pH':None, 'pKex':(), 'Pint':Pint_def, \
                'dipoleD2factor':True, 'F4factor':True, 'F4override':False, 'F4screen':False, 'kill_kappa':False, 'ignoreBases':False}


#####   Class Definition : object for encapsulating, evaluating 'degree of ionization' isolated IDP model #####
class doiModel:
    def __init__(self, seq, info=False, OBfile=None):
        self.seq = seq      # sequence object is a necessary input
        self.seq.polars = self.translate_polar()    # include sequence as polars
        if info:
            self.seq.info()
        # load 2- and 3-body sums if 'OBfile' specified, otherwise calculate immediately
        self.OBfile = OBfile
        if OBfile:
            OBarr = np.load(OBfile)
            try:
                OBind = np.where(OBarr[:,0]==self.seq.N)[0][0]
                self.Onon, self.B = OBarr[OBind, 1:]
            except IndexError:
                OBfile = False
        if not OBfile:
            Onon = 1
            B = 0
            for l in range(2,self.seq.N):
                Onon += np.power(l, -0.5)
                for m in range(1,l):
                    Onon += np.power(l-m, -0.5)
                    for n in range(0,m):
                        B += (l-n) * np.power((l-m)*(m-n), -1.5)
            self.Onon = DTYPE( Onon / self.seq.N )
            self.B = DTYPE( B / self.seq.N )
        # simple functions for charge-dipole & dipole-dipole weights
        self.wcd = lambda l,lB,p,kl: - (PI/3) * np.square(lB*p/(l*l)) * np.exp(-2*kl, dtype=DTYPE) * (2+kl)
        self.wdd = lambda l,lB,p,kl: - (PI/9) * np.square(lB*p*p/(l*l*l)) * np.exp(-2*kl, dtype=DTYPE) * (4+8*kl+4*kl*kl+kl*kl*kl)
        # load in baseline parameters
        self.allpars = {}   # blank initialization to load defaults
        self.setPars()
    #   #   #   #   #   #

    #   translate polars (only relevant if polars are included in theory, cf. 'polars' dictionary)
    def translate_polar(self):
        lst = [(polars[c] if (c in polars) else 0) for c in self.seq.aminos]
        return tuple(lst)

    #   set basic parameters / methods
    def setPars(self, pars={}, pH_seqinfo=True):
        pdict = self.allpars.copy() if self.allpars else default_pars
        pdict.update(pars)
        self.allpars = pdict        # store dictionary of ALL parameters
        self.pars = {k:pdict[k] for k in pdict if k not in ('pH','pKex')}       # all except pH pars (for Free Energy arguments)
        if pdict['pH']:
            self.seq.charges = self.seq.PHtranslate(pH=pdict['pH'], pKexclude=pdict['pKex'])
            self.seq.characterize()
            if pH_seqinfo:
                self.seq.info()
        return pdict

    #   print table of parameters
    def parInfo(self, exclude=('Pint', 'F4factor', 'F4override', 'kill_kappa', 'ignoreBases')):
        print("\nxModel parameters:")
        print("\n\tPAR\tVALUE")
        includedKeys = [k for k in self.allpars.keys() if k not in exclude]
        for par in includedKeys:
            try:
                print("\t{:5}\t{:1.5g}".format(par, self.allpars[par]))
            except:
                print("\t{:5}\t{:6}".format(par, str(self.allpars[par])))
        print("")
        return

    #   function for Debye screening  (dim.less, kappa*l)
    def kapl(self, alP, alM, rh, cs, l, lB):
        if alP < 0 or alM < 0:
            return 0
        else:
            return ( l * np.sqrt( 4*PI*lB* ((self.fracp*alP+self.fracm*alM)*rh + 2*cs ) ) )

    ## Free Energy Terms : always 'beta F / N' -> some overall factors of density '1/rho'
    #   combinatorial entropy - placement of counterions on polymer chain
    def F1(self, alP, alM):
        res = 0
        if 0.0 < alP < 1.0:
            res += self.fracp * alP*np.log(alP, dtype=DTYPE)
            res += self.fracp * (1-alP)*np.log(1-alP, dtype=DTYPE)
        if 0.0 < alM < 1.0:
            res += self.fracm * alM*np.log(alM, dtype=DTYPE)
            res += self.fracm * (1-alM)*np.log(1-alM, dtype=DTYPE)
        return DTYPE(res)

    #   translational entropy - wandering counterions (~plasma)
    def F2(self, alP, alM, rh, cs, l):
        l3 = np.power(l,3)
        rh, cs = rh*l3, cs*l3
        res = 0
        quantP = self.fracp*alP*rh + cs
        quantM = self.fracm*alM*rh + cs
        if quantP > 0.0:
            res += quantP * ( np.log(quantP, dtype=DTYPE) - 1 )
        if quantM > 0.0:
            res += quantM * ( np.log(quantM, dtype=DTYPE) - 1 )
        return DTYPE(res / rh)

    #   ion density fluctuations - from correlation length xi^-3 (xi~1/kappa)
    def F3(self, kl, rh, l):
        res = kl*kl*kl       # ionic fluctuations simply give kappa^3
        return DTYPE( (-1/(12*PI)) * res / (rh*l*l*l) )

    #   ion pair energy - reduction in energy from attraction of opposite charges
    def F4(self, alP, alM, l, lB, p, dlt, kl, extraFac=True, screen=False):
        if np.isclose(dlt,0.0) or np.isclose(p,0.0):
            return DTYPE(0)
        dlt_extra = 0.5 if extrafac else 0      # correction factor from effective dipole form 1/d to 1/p  (i.e. delta*(1+1/(2*delta)) = delta+0.5)
        res = - ( self.fracp*(1-alP) + self.fracm*(1-alM) )
        # explicit 'p' -> interpretation of 'delta' as just ratio of dielectrics: eps_bulk/eps_local
        res *= lB * (dlt + dlt_extra) / p
        if screen:
            res *= np.exp(-kl*p/l, dtype=DTYPE)
        return DTYPE(res)

####    CONTINUE EDITING BELOW  (starting with F5 and Omega, esp. Ocddd and Ocddd_int  [cf. seqXij])        #####

    #   polymer free energy - chain entropy, intra-chain interactions (potential energy)
    def F5(self, alP, alM, x, l, lB, p, dlt, kl, w2, w3, dipoleD2factor=True, ignoreBases=False):
        if x < 0.0:
            return 0
        D2 = (dlt*dlt) if dipoleD2factor else 1
        wcd = self.wcd(lB,p,kl) * D2
        wdd = self.wdd(lB,p,kl) * D2
        om, ocd, odd = self.Omega(alP,alM, ...)
        res = 1.5 * (x - np.log(x, dtype=DTYPE))
        res += w3 * ((3/(2*PI*x))**(3)) * self.B / 2
        res += ((3/(2*PI*x))**(1.5)) * self.Omega(alP, alM, l, lB, p, dlt, kl, ignoreBases, dipoleD2factor)
        res += (2*lB/(PI*l)) * self.Q(alP, alM, x, kl)
        return DTYPE(res / DTYPE(self.seq.N))

    #   2-body volume exclusions (including effective c-d & d-d interactions)
    def Omega(self, alP, alM, l, lB, p, dlt, kl, dipoleD2factor=True, ignoreBases=False):
        return self.Onon, self.Ocd, self.Odd
#        return DTYPE(w2*self.Onon + D2* ( wcd*self.Ocd(alP,alM,includeBases) + wdd*self.Odd(alP,alM,includeBases) ) + \
#                                    D2* ( self.Ocd_int(alP,alM,kl,l,lB) + self.Odd_int(kl,l,lB) ) )
    #   charge-dipole short-range interaction
    def Ocd(self, alP, alM, includeBases=True):
        N = self.N
        pseq = self.sequence
        res = 0
        for m in range(1,N):
            if pseq[m] == 0:
                continue
            for n in range(0,m):
                if pseq[n] == 0:
                    continue
                if pseq[m] > 0:
                    bm = abs(pseq[m])*alP
                    if includeBases:
                        dm = abs(pseq[m])*(1-alP)
                    else:
                        dm = 0
                else:
                    bm = abs(pseq[m])*alM
                    dm = abs(pseq[m])*(1-alM)
                if pseq[n] > 0:
                    if includeBases:
                        bn = abs(pseq[n])*(1-alP)
                    else:
                        bn = 0
                    dn = abs(pseq[n])*alP
                else:
                    bn = abs(pseq[n])*(1-alM)
                    dn = abs(pseq[n])*alM
#                if pseq[m] == 1:
#                    bm = alP
#                    if includeBases:
#                        dm = 1-alP
#                    else:
#                        dm = 0
#                else:
#                    bm = alM
#                    dm = (1-alM)
#                if pseq[n] == 1:
#                    if includeBases:
#                        bn = 1-alP
#                    else:
#                        bn = 0
#                    dn = alP
#                else:
#                    bn = (1-alM)
#                    dn = alM
                res += (bm*bn + dm*dn) * ((m-n)**(-0.5))
        return ( res / DTYPE(N) )
    #   full version
    def Ocd_full(self, alP, alM, kl, lB):
        includeBases, dipoleD2factor = self.pars["includeBases"], self.pars["dipoleD2factor"]
        l, p = self.pars["l"], self.pars["p"]
        D2 = (self.pars["delta"]**2) if dipoleD2factor else 1
        wcd = - (PI/3) * ((lB*p/(l**2))**2) * (2 + kl) * np.exp(-2*kl, dtype=DTYPE)
        return ( D2* wcd*self.Ocd(alP,alM,includeBases) )
    #   dipole-dipole short-range interaction
    def Odd(self, alP, alM, includeBases=True):
        N = self.N
        pseq = self.sequence
        res = 0
        for m in range(1,N):
            if pseq[m] == 0:
                continue
            elif (pseq[m] == 1) and (not includeBases):
                continue
            for n in range(0,m):
                if pseq[n] == 0:
                    continue
                elif (pseq[n] == 1) and (not includeBases):
                    continue
                if pseq[m] > 0:
                    gm = abs(pseq[m]) * (1-alP)
                else:
                    gm = abs(pseq[m]) * (1-alM)
                if pseq[n] > 0:
                    gn = abs(pseq[n]) * (1-alP)
                else:
                    gn = abs(pseq[n]) * (1-alM)
#                if pseq[m] == 1:
#                    gm = 1-alP
#                else:
#                    gm = (1-alM)
#                if pseq[n] == 1:
#                    gn = 1-alP
#                else:
#                    gn = (1-alM)
                res += gm*gn * ((m-n)**(-0.5))
        return ( res / DTYPE(N) )
    #   full version
    def Odd_full(self, alP, alM, kl, lB):
        includeBases, dipoleD2factor = self.pars["includeBases"], self.pars["dipoleD2factor"]
        l, p = self.pars["l"], self.pars["p"]
        D2 = (self.pars["delta"]**2) if dipoleD2factor else 1
        wdd = - (PI/9) * ((lB*p*p/(l**3))**2)
        wdd *= (4 + 8*kl + 4*((kl)**2) + ((kl)**3)) * np.exp(-2*kl, dtype=DTYPE)
        return ( D2* wdd*self.Odd(alP,alM,includeBases) )
    #   intrinsic charge-dipole short-range interaction
    def Ocd_int(self, alP, alM, kl, l, lB):
        p = Pintrinsic
#        DLT = self.pars["delta"] if DIPOLE_DELTA_FAC else 1
        wcd = - (PI/3) * ((lB*p/(l**2))**2) * (2 + kl) * np.exp(-2*kl, dtype=DTYPE)
        N = self.N
        pseq = self.sequence
        polarseq = self.polars
        res = 0
        for m in range(1,N):
            for n in range(0,m):
                if polarseq[m] > 0:
                    dm = 1
                    if pseq[n] > 0:
                        dn = abs(pseq[n]) * alP
                    elif pseq[n] < 0:
                        dn = abs(pseq[n]) * alM
                    else:
                        dn = 0
#                    if pseq[n] > 0:
#                        dn = alP
#                    elif pseq[n] < 0:
#                        dn = alM
#                    else:
#                        dn = 0
                else:
                    dm, dn = 0, 0
                if polarseq[n] > 0:
                    bn = 1
                    if pseq[m] > 0:
                        bm = abs(pseq[m]) * alP
                    elif pseq[m] < 0:
                        bm = abs(pseq[m]) * alM
                    else:
                        bm = 0
#                    if pseq[m] > 0:
#                        bm = alP
#                    elif pseq[m] < 0:
#                        bm = alM
#                    else:
#                        bm = 0
                else:
                    bm, bn = 0, 0
                res += (bm*bn + dm*dn) * ((m-n)**(-0.5))
        return ( DTYPE(res * wcd) / DTYPE(N) )
    #   intrinsic dipole-dipole short-range interaction
    def Odd_int(self, kl, l, lB):
        p = Pintrinsic
#        DLT = self.pars["delta"] if DIPOLE_DELTA_FAC else 1
        wdd = - (PI/9) * ((lB*p*p/(l**3))**2)
        wdd *= (4 + 8*kl + 4*((kl)**2) + ((kl)**3)) * np.exp(-2*kl, dtype=DTYPE)
        N = self.N
        polarseq = self.polars
        res = 0
        for m in range(1,N):
            for n in range(0,m):
                gm = polarseq[m]
                gn = polarseq[n]
                res += gm*gn * ((m-n)**(-0.5))
        return ( DTYPE(res * wdd) / DTYPE(N) )


    #   electrostatic attractions among chain monomers
    def Q(self, alP, alM, x, kl=0, derivative=False):
        N = self.N
        pseq = self.sequence
        if derivative:
            Afunc = self.derA
        else:
            Afunc = self.A
        total = 0
        for m in range(1,N):
            if pseq[m] == 0:
                continue
            for n in range(0,m):
                if pseq[n] == 0:
                    continue
                if pseq[m] > 0:
                    qm = pseq[m] * alP
                else:
                    qm = pseq[m] * alM
                if pseq[n] > 0:
                    qn = pseq[n] * alP
                else:
                    qn = pseq[n] * alM
#                if pseq[m] == 1:
#                    qm = alP
#                elif pseq[m] == -1:
#                    qm = -alM           # SIGN INTRODUCED to capture charge from D.o.I.
#                elif pseq[m] == -2:     # allowing for doubly charged residues, 'X', with q=-2
#                    qm = -2*alM
#                if pseq[n] == 1:
#                    qn = alP
#                elif pseq[n] == -1:
#                    qn = -alM           # sign fix
#                elif pseq[n] == -2:
#                    qn = -2*alM         # 'X' residue
                total += qm * qn * ((m-n)**2) * Afunc(m,n,x,kl)
        return DTYPE(total / DTYPE(N))
    #   function with details, screening, etc.
    def A(self, m, n, x, kl):
        res = 0.5 * ( (6*PI/x)**(0.5) ) * ( (m-n)**(-1.5) )
        res += - kl * (0.5*PI/((m-n))) * self.Xerfc( kl * ((x*(m-n)/6)**(0.5)) )
        return DTYPE(res)
    #   derivative of above function (d/dx) -> for (quasi-)analytic solution of 'w2' given some data point
    def derA(self, m, n, x, kl):
        res = - (1/(4*x)) * ( (6*PI/x)**(0.5) ) * ( (m-n)**(-1.5) )
        res += (1/12) * (kl**2) * ( (6*PI/x)**(0.5) ) * ( (m-n)**(-0.5) )
        res += - (PI/12) * (kl**3) * self.Xerfc( kl * ((x*(m-n)/6)**(0.5)) )
        return DTYPE(res)
    #   shortcut for SCD check
    def SCD(self, alP=1, alM=1, x=1, kl=0):
        # check for necessary parameters first
        try:
            self.choice
        except AttributeError:
            self.setPars()
        fac = 0.5 * ( (6*PI)**(0.5) )
        return ( self.Q(alP,alM,x,kl) / fac )
    #   detailed SCD function (for asymmetry, etc)
    def ogSCD(self, mode="all", low_salt=False):
        seq = self.sequence
        N = len(seq)
        tot = 0
        for m in range(1,N):
            for n in range(m):
                qm = seq[m]
                qn = seq[n]
                if mode == "++":
                    if qm < 0 or qn < 0:
                        continue
                elif mode == "--":
                    if qm > 0 or qn > 0:
                        continue
                elif mode in ("+-", "-+"):
                    if round(np.sign(qm)) == round(np.sign(qn)):
                        continue
                # use power = 1 for 'low_salt', power = 0.5 for regular
                if low_salt:
                    tot += qm * qn * ( (m-n) )
                else:
                    tot += qm * qn * ( (m-n)**(0.5) )
        return (tot/N)

    #   electrostatic interactions for polyelectrolyte - Muthukumar (from Edwards Hamiltonian)
    def Theta(self, a):
        Xerfc = self.Xerfc
        res = 1/(3*a) + 2/(a**2) - (PI**(0.5))/(a**(2.5)) - 0.5*(PI**(0.5))/(a**(1.5))
        res += 0.5*(PI**(0.5)) * (2/(a**(2.5)) - (a**(-1.5))) * self.Xerfc(a**(0.5))
        return res

    #   FULL FREE ENERGY - use list/tuple/array for optimization variables
    def Ftot(self, triplet, rho=5e-4, cs=0.0, l=1, lB=2, p=0, delta=1, w2=0, w3=0,
                F4factor=1, F4screen=True, F4override=False, kill_kappa=False, includeBases=True, dipoleD2factor=False):
        dipole_F4 = p   # dipole length actually used in F4 (ion pair formation screening)
        if F4override:  # and (p < 0.01):     # option: force finite value of dipole length in F4 (screening)
            dipole_F4 = float(F4override) if type(F4override)!=type(True) else 0.5
#            dipole_F4 = float(F4override) if type(F4override)==type(0.) else 0.5
#            dipole_F4 = 1       #
#            dipole_F4 = 0.5     #
#        print(f" F4 override set to '{F4override}'; p set to '{dipole_F4}'\n")
#        dipole_F4 = 0.5    # artificially keep ion-pair F4 with its own value of p
        dipole_F4 = DTYPE(dipole_F4)
        if self.choice == "muthu":
            pname = self.seqName
            if "+" in pname:
                (alP, x) = triplet  # only a doublet for MUTHU case
                alM = 0
            elif "-" in pname:
                (alM, x) = triplet  # different doublet if negative polyelectrolyte
                alP = 0
        else:
            (alP, alM, x) = triplet
        (alP, alM, x, rho, cs, l, lB, p, delta, w2, w3) = ( DTYPE(alP), DTYPE(alM), DTYPE(x), DTYPE(rho),
                DTYPE(cs), DTYPE(l), DTYPE(lB), DTYPE(p), DTYPE(delta), DTYPE(w2), DTYPE(w3) )   # ensure data types
        kl = DTYPE(0) if kill_kappa else DTYPE(self.kapl(alP,alM,rho,cs,l,lB))
#        kl = DTYPE( self.kapl(alP,alM,rho,cs,l,lB) )   # get new value for screening upon each call
#        kl = DTYPE( self.kapl(0,0,rho,cs,l,lB) )   # using salt-only screening (no counter-ions included in kappa)
        res = self.F1(alP,alM)
        res += self.F2(alP,alM,rho,cs)
#        res += self.F3(alP,alM,rho,cs,l,lB)
#        print(f"\nKL test:\t{kl}\n")
        res += self.F3(kl,rho)
        res += self.F4(alP,alM,kl,l,lB,dipole_F4,delta,F4screen,F4factor)
        res += self.F5(alP,alM,x,kl,l,lB,p,w2,w3,includeBases,dipoleD2factor)
        return DTYPE(res)

    #   testing function - set some parameters, get some total Free Energy values
    def test(self, parset="Ghosh", KL="full", whichXerfc=myXerfc, triplets=((1,1,1),(0.5,0.5,1)),
                seqVals=True, seqList=False, parVals=True, pauseEvery=False, customFunc=None):
        self.setPars(pset=parset, KL=KL, whichXerfc=whichXerfc)
        if seqVals:
            self.seqInfo(showSeqList=seqList)
        if parVals:
            self.parInfo()
        for trip in triplets:
            (P, M, X) = trip
            print("\n\ttriplet (alP,alM,x) = " + str(trip))
            print("\t-> Gives: Ftot = %2.8g" % self.Ftot((P,M,X), **self.pars))
            if customFunc:
                print("\t-> Custom Func = %2.8g" % customFunc((P,M,X)))
            print("")
            if pauseEvery:
                pause()
        return

    #   optimization for given set of parameters (should be set prior to calling this)
    def optimize(self, parset="ghosh", choice="ghosh", KL="full", Xerfc=erfcx, method="NM-TNC", #"Powell",
                    alBounds=(1e-8,1.0), xBounds=(1e-3,4), ref=(0.5,0.5,0.5), init=(0.2,0.2,0.8), init_2=None,
                    showPars=("lB","cs","delta"), SCALE=1, TOL=1e-9,
                    messages=False, pauseFail=False, perturbTNC=False): #, dip_F4_override=False):
        # set parameters now
        self.setPars(pset=parset, choice=choice, KL=KL, whichXerfc=Xerfc)
        # dictionary (for printing, optionally)
        pdict = self.pars
        # update dictionary with F4 override option
#        pdict.update({"F4override":dip_F4_override})
#        print(f" F4 override set to '{dip_F4_override}'; p set to '{pdict['p']}'\n")
        # handle variable sets of different lengths
        if self.choice == "muthu":
            var_bounds = (alBounds, xBounds)        # just 2 variables in Muthu / PE case
            print_fmt = "(al,x) = (%1.5g, %1.5g)"   #
        else:
            var_bounds = (alBounds, alBounds, xBounds)          # variable bounds
            print_fmt = "(alP,alM,x) = (%1.5g, %1.5g, %1.5g)"   # format for printing results
        # handle choice of printing selected parameters
        if len(showPars) > 0:
            pinfo = "for (" + (", ".join(showPars)) + ") = ("
            pinfo += (", ".join(["%1.5g" % pdict[p] for p in showPars])) + ") "
        else:
            pinfo = ""
        # intro message (optional)
        if messages:
            print("\nOPTIMIZING %s..." % pinfo)
        t1 = perf_counter()     # begin timer
        Fref = self.Ftot(ref, **pdict)     # reference value of Free Energy
        minfunc = lambda trip: ( ( self.Ftot(trip, **pdict) - Fref ) * SCALE )
        if method.lower() == "nelder-mead":
            result = minimize(minfunc, init,
                        method="Nelder-Mead", bounds=var_bounds,    # tol=TOL,
                        options={"maxiter":30000, "xatol":TOL, "fatol":TOL, "adaptive":True})
        elif method.lower() == "nm-tnc":    # hybrid method: Nelder-Mead to find basin, TNC to hone in
            if messages:
                print("\t entering Nelder-Mead algorithm ...\n")
            res0 = minimize(minfunc, init,
                        method="Nelder-Mead", bounds=var_bounds,    # tol=TOL,
                        options={"maxiter":30000, "xatol":TOL, "fatol":TOL, "adaptive":True})
            new0 = res0.x       # N-M solution used as 'seed' (initial pt.) for TNC
            if messages:
                print("\t passing result " + (print_fmt % tuple(new0)) + " to TNC for refinement ...\n")
            res1x = self.optimize(parset=parset, choice=choice, KL=KL, Xerfc=Xerfc, method="TNC",
                alBounds=alBounds, xBounds=xBounds, ref=ref, init=new0, showPars=showPars,
                SCALE=SCALE, TOL=TOL, messages=messages, pauseFail=pauseFail,
                perturbTNC=perturbTNC)  #, dip_F4_override=dip_F4_override)
            # construct representative 'result' object
            result = res0
            result.x = res1x
        elif method.lower() == "nm-tnc-2":  # two-step approach with hybrid method - compare basins explicitly
            if init_2:
                init1 = init
                init2 = init_2
            else:
                temp_al = min(alBounds[1], init[2])     # ensure reflection is acceptable
                temp_x = max(xBounds[0], init[0])       #
                if init[2] < init[0]:
                    init1 = init        # use given 'init' as 'init1' if it lands ~ in basin 1
#                    init2 = (0.2, 0.2, 0.6)       # basin 2 : condensed & expanded
                    init2 = (temp_al, temp_al, temp_x)      # reflect initial point about line defining basins (al=x)
                else:
#                    init1 = (0.5, 0.5, 0.2)       # basin 1 default : ionized & contracted
#                    init1 = (init[2], init[2], init[0])     # reflect
                    init1 = (temp_al, temp_al, temp_x)      # reflect initial point about line defining basins (al=x)
                    init2 = init        # use given 'init' as 'init2' if it lands ~ in basin 2
            res1 = self.optimize(parset=parset, choice=choice, KL=KL, Xerfc=Xerfc, method="NM-TNC",
                alBounds=alBounds, xBounds=xBounds, ref=ref, init=init1, showPars=showPars,
                SCALE=SCALE, TOL=TOL, messages=messages, pauseFail=pauseFail,
                perturbTNC=perturbTNC)  #, dip_F4_override=dip_F4_override)
            res2 = self.optimize(parset=parset, choice=choice, KL=KL, Xerfc=Xerfc, method="NM-TNC",
                alBounds=alBounds, xBounds=xBounds, ref=ref, init=init2, showPars=showPars,
                SCALE=SCALE, TOL=TOL, messages=messages, pauseFail=pauseFail,
                perturbTNC=perturbTNC)  #, dip_F4_override=dip_F4_override)
            FE1 = self.Ftot(res1, **pdict)
            FE2 = self.Ftot(res2, **pdict)
            if FE1 < FE2:
                real_res = res1
            else:
                real_res = res2
            if messages:
                print("[FE1 = %2.5g, FE2 = %2.5g]" % (FE1, FE2))
                print("  >> FINAL CHOICE : " + print_fmt % real_res + "\n")
            return real_res
        elif method.lower() == "l-bfgs-b":
            result = minimize(minfunc, init,
                        method="L-BFGS-B", bounds=var_bounds,   # tol=TOL,
                        options={"maxiter":25000, "maxfun":30000, "ftol":TOL, "maxls":30})
        elif method.lower() == "slsqp":
            result = minimize(minfunc, init,
                        method="SLSQP", bounds=var_bounds,    # tol=TOL,
                        options={"eps":TOL, "ftol":TOL, "maxiter":30000})
        elif method.lower() == "powell":
            result = minimize(minfunc, init,
                        method="Powell", bounds=var_bounds,    # tol=TOL,
                        options={"xtol":TOL, "ftol":TOL, "maxiter":30000})
        elif method.lower() == "trust-constr":
            result = minimize(minfunc, init,
                        method="trust-constr", bounds=var_bounds,    # tol=TOL,
                        options={"xtol":TOL, "gtol":TOL, "maxiter":30000})
        elif method.lower() == "tnc":
            bad = True      # track bad convergence
            TNC_eval = 10   # max number of retries
            count = 0       # current number of retries
            while bad and count < TNC_eval:
                if count > 0:
                    perturb = (0.04*rand(), 0.04*rand(), 0.08*rand())
#                    perturb = (-0.04, -0.02, 0.08)
                    for i in range(len(init)):
                        new = init[i] + perturb[i]
                        if i in (0, 1):
                            init[i] = self.boundUpdate(new, alBounds)
                        elif i == 2:
                            init[i] = self.boundUpdate(new, xBounds)
                    if messages:
                        print("\n" + result.message + " -> " + "Perturbing initial point...\n")
                        print("[new = (%1.4g, %1.4g, %1.4g)]" % tuple(init))
                if messages:
                    print("\nUSING INITIAL POINT : (%1.4g, %1.4g, %1.4g)" % tuple(init))
                result = minimize(minfunc, init,
                            method="tnc", bounds=var_bounds,    # tol=TOL,
                            options={"xtol":TOL, "ftol":TOL, "gtol":TOL, "maxiter":30000})
#                if "linear search failed" in result.message.lower():
                bad = not result.success
                count += 1
                if not perturbTNC:
                    break
        else:
            print("\n\nERROR: given method '%s' is unsupported.\n\n")
            return
        t2 = perf_counter()     # end timer
        # results messages (optional)
        if messages:
            print("\nDONE - elapsed time:\t%2.5f" % (t2-t1))
            print("\n[Extra Message: '%s']" % result.message)
            if result.success:
                print("\n\tSUCCESSFULLY found:\t" + print_fmt % tuple(result.x) + "\n")
            else:
                print("\n\tFAILED to find minimum;\t" + print_fmt % tuple(result.x) + "\n")
                if pauseFail:
                    pause()
#            print("\n[Extra Message: '%s']\n" % result.message)
        else:
            if not result.success:
                if pauseFail:
                    print("\n\t**FAILED to find minimum;\t" + print_fmt % tuple(result.x))
                    print("\t  [Extra Message: '%s']" % result.message)
                    pause()
        return tuple(result.x)

    #   optimize repeatedly for some varying parameter (pass 'optArgs' as keyword-arg dictionary to 'optimize')
    def multiOpt(self, multiPar="cs", parVals=[0], parset="ghosh", optArgs={}, seedNext=True):
        # use pre-set parameter arguments, update if further specification
        if type(parset) == str:
            self.setPars(pset=parset, choice=parset)
        else:
            try:
                self.setPars(pset=parset, choice=self.choice)
            except AttributeError:
                self.setPars(pset=parset)
        pars = self.pars
        # all keyword arguments to 'optimize' function ['parset' is useless to specify here!]
        args = {}
        args.update(optArgs)
        # prepare result list
        results = [0]*len(parVals)
        res_i = 0
        for pval in parVals:
#            print(f"ARGS passed to 'optimize' function :  {args}\n")
            pars.update({multiPar:pval})
            if (multiPar == "delta") and ("F4factor" in pars):
                if pars["F4factor"] > 1.001:
                    f4_fac = (1+1/(2*pval))
                    pars.update({"F4factor":f4_fac})
            args.update({"parset":pars})
            if "SCALE" not in args:
                scale = abs(self.SCD())
                if multiPar == "lB":
                    scale *= pval
                args.update({"SCALE":scale})
            pres = self.optimize(**args)
            results[res_i] = pres
            res_i += 1
            if seedNext:
                args.update( {"init":pres} )
                # if also re-setting reference point
#                args.update( {"ref":pres} )
        return results

    #   optimize repeatedly while varying Temperature, entering through one or more model parameters (pass 'optArgs' as keyword-arg dictionary to 'optimize')
    def multiOpt_T(self, Tvals=[273], Tfuncs={'lB':(lambda T:1)}, parset="ghosh", optArgs={}, seedNext=True):
        # use pre-set parameter arguments, update if further specification
        if type(parset) == str:
            self.setPars(pset=parset, choice=parset)
        else:
            try:
                self.setPars(pset=parset, choice=self.choice)
            except AttributeError:
                self.setPars(pset=parset)
        pars = self.pars
        # all keyword arguments to 'optimize' function ['parset' is useless to specify here!]
        args = {}
        args.update(optArgs)
        # prepare result list
        results = [0]*len(Tvals)
        res_i = 0
        for t in Tvals:
#            print(f"ARGS passed to 'optimize' function :  {args}\n")
            for p in Tfuncs:
                pval = Tfuncs[p](t)
                pars.update({p:pval})
                if (p == "delta") and ("F4factor" in pars):
                    if pars["F4factor"] > 1.001:
                        f4_fac = (1+1/(2*pval))
                        pars.update({"F4factor":f4_fac})
            args.update({"parset":pars})
            if "SCALE" not in args:
                scale = abs(self.SCD())
                if "lB" in Tfuncs:
                    scale *= Tfuncs["lB"](t)
                args.update({"SCALE":scale})
            pres = self.optimize(**args)
            results[res_i] = pres
            res_i += 1
            if seedNext:
                args.update( {"init":pres} )
                # if also re-setting reference point
#                args.update( {"ref":pres} )
        return results

    #   create plot(s) of optimized variable(s) for some varying parameter
    def plotOpt(self, optVars="all", multiPar="cs", parVals=[0,0.01], parset="ghosh", optArgs={}, savePlots="",
                    SIZE=7, COLORS={}, LABELS={}, TITLES={}, gridDashes=(2,5),
                    showPars=True, showEval=True, PMtogether=True):
        # save all chosen plots -> set directory 'savePlots'
        # plot colors
#        COLORS.update( {"alp":"lightcoral"} )
#        COLORS.update( {"alm":"lightblue"} )
#        COLORS.update( {"x":"lightgreen"} )
        COLORS.update( {"alp":"coral"} )
        COLORS.update( {"alm":"blue"} )
        COLORS.update( {"x":"green"} )
        COLORS.update( {"Qterm":"plum"} )
        # vertical axis labels
        LABELS.update( {"alp":r"$\alpha_+$"} )
        LABELS.update( {"alm":r"$\alpha_-$"} )
        LABELS.update( {"x":r"$x$"} )
        LABELS.update( {"Qterm":r"$2\, l_B\, Q' / \pi\, l$"} )
        # plot titles
        TITLES.update( {"alp":r"Degree of Ionization"} )
        TITLES.update( {"alm":TITLES["alp"]} )
        TITLES.update( {"x":r"Length Factor"} )
        TITLES.update( {"Qterm":r"Electrostatic Term, $\propto Q'$"} )
        # typeset axis labels
        if multiPar == "cs":
            XLAB = r"$\tilde{c}_s$"
            nonpar = "lB"
        elif multiPar == "lB":
            XLAB = r"$\tilde{l}_B$"
            nonpar = "cs"
        else:
            XLAB = r"$%s$" % multiPar
        # optimized variable choice(s)
        if optVars in ("all", "triplet"):
            if self.choice != "muthu":
                selVars = ("alp","alm","x")
            else:
                selVars = ("alp","x")
        elif optVars in ("withQ", "allQ", "Qterm"):
            if self.choice != "muthu":
                selVars = ("alp","alm","x","Qterm")
            else:
                selVars = ("alp","x","Qterm")
        else:
            selVars = optVars
#        # positions in each result tuple
#        POS = {"alp":0, "alm":1, "x":2}
        # position dictionary for each result tuple
        POS = {}
        for var in selVars:
            POS.update({var:selVars.index(var)})
        # calculate list of results
        if showEval:
            print("\nCALCULATING set of optimized variables...\n")
            t1 = perf_counter()
        allres = self.multiOpt(multiPar, parVals, parset, optArgs)
        if showEval:
            t2 = perf_counter()
            print("\n  -> Elapsed Time:\t%2.4f\n" % (t2-t1))
        # optionally show relevant parameters, after calculation
        if showPars:
            self.parInfo(showKL=True, exclude=(multiPar,))
        # independent variable array
        x = np.array(parVals, dtype=DTYPE)
        for var in selVars:
            # make array of y-values for selected variable
            if var == "Qterm":
                if multiPar == "cs":
                    klfunc = lambda P,M,cs: self.kapl(P,M,self.pars["rho"],cs,self.pars["l"],self.pars["lB"])
                    Qfunc = lambda P,M,X,cs: (2*self.pars["lB"]/(PI*self.pars["l"])) * self.Q(P,M,X,klfunc(P,M,cs))
                elif multiPar == "lB":
                    klfunc = lambda P,M,lB: self.kapl(P,M,self.pars["rho"],self.pars["cs"],self.pars["l"],lB)
                    Qfunc = lambda P,M,X,lB: (2*lB/(PI*self.pars["l"])) * self.Q(P,M,X,klfunc(P,M,lB))
                else:
                    print("\nERROR: selected parameter (%s) is not supported for the 'Qterm' list.\n" % multiPar)
                y = np.zeros(len(x), dtype=DTYPE)
                for i in range(len(x)):
                    (P,M,X) = allres[i]
                    y[i] = Qfunc(P,M,X,x[i])
            else:
                y = np.array( [ res[POS[var]] for res in allres ] , dtype=DTYPE)
            if not (PMtogether and var == "alm"):
                fig = plt.figure("optimized variable plot", (1.2*SIZE, SIZE))
                ax = fig.add_subplot()
                YLAB = LABELS[var]
            else:
                YLAB = LABELS["alp"] + "  ,  " + LABELS["alm"]
            ax.plot(x, y, '-', color=COLORS[var], label=LABELS[var])
            ax.set_xlabel(XLAB)
            if 0 < y.min() < 0.15:
                ax.set_ylim(-0.01, 1.02*y.max())
            ax.set_ylabel(YLAB)
            ax.set_title(TITLES[var] + "  (seq. '%s')" % self.seqName)
            ax.grid(True, dashes=gridDashes)
            ax.minorticks_on()
            fig.tight_layout()
            # save plots to specified directory, if given
            if savePlots:
                # file prefix -> quantity
                if var == "Qterm":
                    filePre = "Q"
                else:
                    filePre = var.upper()
                # file middle -> sequence
                fileMid = self.seqName
                # file suffix -> important value(s) of other par(s)
                nonval = "%g" % self.pars[nonpar]
                if "." in nonval:
                    nonval = nonval[nonval.index(".")+1:]
                fileSuf = nonpar + nonval + ".png"
                fileFull = (filePre + "_" + fileMid + "-" + fileSuf)
                fig.savefig(savePlots + fileFull, format="png")
                print("\n * FIGURE SAVED as '%s', in location '%s'. *\n" % (fileFull, savePlots))
            if PMtogether and var == "alm":
                ax.legend(loc="best", edgecolor="inherit")
            if PMtogether and var == "alp":
                continue
            plt.show()
            plt.close()
        return allres

    #   create heatmap of Free Energy for some pair of variables
    def mapFE(self, Hoptions={"cmap":cm.Spectral, "vmax":0}, SIZE=7, pair=("al","x"), ALM_fac=1, AL_val=1,
                Npts=50, bounds={}, showPars=True, shiftMax=10, xval=1, R_trans=False, make_3d=False, contour=False, SHOW=True):
        """
        Make a color map of the Free Energy landscape.
        Hoptions : dictionary of options passed into 'pcolor'
        SIZE : overall size (height) of figure; width from 4:3 ratio
        pair : variable pair for plotting, in form (x,y)
                > (al,x) or (alP,x) or (alM,x) or (alP,alM) or (alM,alP)
        ALM_fac : custom factor for 'alM' from 'alP', when using (al,x) pair    [default: 1, i.e. enforcing alM=1*alP]
        AL_val : custom value for other degree of ionization, when using (alP,x) or (alM,x)     [i.e. fix alM=1 while varying alP]
        Npts : number of points long each axis  [total number of points in grid = Npts^2]
        bounds : dictionary of bounds for each variable choice  (x, al, alP, alM)
        showPars : boolean setting for showing parameter set before calculation
        shiftMax : amount to shift maximum FE range, relative to minimum    [i.e. (FE max) = (FE min) + (shift)]
        xval : value on 'x' when plotting just charge, pair (alP,alM) or (alM,alP)
        R_trans : boolean for translating 'x' back to 'Ree' in final plot
        make_3d : use 3D rendering for Free Energy surface, with projection of map on bottom
        contour : use 'contourf' to make colorful contours in place of smooth 'pcolor' (for 2d mode)
        SHOW : boolean setting for actually showing the plot; disable in order to annotate before showing
        """
        # dictionary for bounds
        bound_d = {"al":(0,1), "x":(1e-2,2), "alP":(0,1), "alM":(0,1)}
        bound_d.update(bounds)
        # optionally show parameter choices
        if showPars:
            self.parInfo(showKL=True, exclude=())
        pdict = self.pars   # grab parameter dictionary
        # make table of Free Enenrgy values
        NumPoints = Npts
        var1 = np.linspace(bound_d[pair[0]][0], bound_d[pair[0]][1], NumPoints, dtype=DTYPE)
        var2 = np.linspace(bound_d[pair[1]][0], bound_d[pair[1]][1], NumPoints, dtype=DTYPE)
        mesh1, mesh2 = np.meshgrid(var1, var2)
        shp = mesh1.shape
        Fmesh = np.zeros(shp, dtype=DTYPE)
        if pair == ("al", "x"):
            Xlab = r"$\alpha$"  #  (degree of ionization)"
            Ylab = r"$x$  (length factor)"
            if self.choice == "muthu":
                Ffunc = lambda i,j: self.Ftot(triplet=(mesh1[i][j],mesh2[i][j]), **pdict)
                titlestr = "polyelectrolyte"
            else:
                Ffunc = lambda i,j: self.Ftot(triplet=(mesh1[i][j],ALM_fac*mesh1[i][j],mesh2[i][j]), **pdict)
                if ALM_fac < 1:
                    titlestr = r"with $\alpha_-=$" + f"{ALM_fac:.3g}" + r"$\alpha_+$"
                else:
                    titlestr = r"with $\alpha_-=\alpha_+$"
        elif pair == ("alP", "x"):
            Xlab = r"$\alpha_+$"    #  (degree of ionization)"
            Ylab = r"$x$  (length factor)"
            if self.choice == "muthu":
                Ffunc = lambda i,j: self.Ftot(triplet=(mesh1[i][j],mesh2[i][j]), **pdict)
                titlestr = "polyelectrolyte"
            else:
                Ffunc = lambda i,j: self.Ftot(triplet=(mesh1[i][j],AL_val,mesh2[i][j]), **pdict)
                titlestr = r"with $\alpha_-=$" + f"{AL_val:.3g}"
        elif pair == ("alM", "x"):
            Xlab = r"$\alpha_-$("    #  (degree of ionization)"
#            Xlab = r"$\alpha_-$ (degree of ionization)"
            Ylab = r"$x$  (length factor)"
            if self.choice == "muthu":
                Ffunc = lambda i,j: self.Ftot(triplet=(mesh1[i][j],mesh2[i][j]), **pdict)
                titlestr = "polyelectrolyte"
            else:
                Ffunc = lambda i,j: self.Ftot(triplet=(AL_val,mesh1[i][j],mesh2[i][j]), **pdict)
                titlestr = r"with $\alpha_+=$" + f"{AL_val:.3g}"
        elif pair == ("alP", "alM"):
            Xlab = r"$\alpha_+$"    #  (degree of ionization)"
            Ylab = r"$\alpha_-$"    #  (degree of ionization)"
            Ffunc = lambda i,j: self.Ftot(triplet=(mesh1[i][j],mesh2[i][j],xval), **pdict)
            titlestr = r"with $x=$%1.5g" % xval
        elif pair == ("alM", "alP"):
            Xlab = r"$\alpha_-$"    #  (degree of ionization)"
            Ylab = r"$\alpha_+$"    #  (degree of ionization)"
            Ffunc = lambda i,j: self.Ftot(triplet=(mesh2[i][j],mesh1[i][j],xval), **pdict)
            titlestr = r"with $x=$%1.5g" % xval
        else:
            print("\nERROR: given parameter pair (%s,%s) is not supported.\n" % pair)
        for i in range(shp[0]):
            for j in range(shp[1]):
                Fmesh[i][j] = Ffunc(i,j)
        if shiftMax:
            Hoptions.update({"vmax":(Fmesh.min()+shiftMax)})
        # translate back to Ree
        if R_trans:
            R_func = lambda x: np.sqrt(self.N*3.8*8*x)/10       # to [nm]
            if pair[0] == 'x':
                var1 = R_func(var1)
                mesh1 = R_func(mesh1)
                Xlab = r"$R_{ee}$   [nm]"
            elif pair[1] == 'x':
                var2 = R_func(var2)
                mesh2 = R_func(mesh2)
                Ylab = r"$R_{ee}$   [nm]"
        # prepare axes
        fig = plt.figure("Visualizing Free Energy", (1.3*SIZE,SIZE))
        if make_3d:
            ax = fig.add_subplot(111, projection='3d', elev=37, azim=-94, computed_zorder=False)
            # necessary adjustments / additional info
            zrange = shiftMax if shiftMax else (Fmesh.max()-Fmesh.min())
            zmin = Fmesh.min() - zrange/5
            zmax = (Fmesh.min() + 1.3*shiftMax) if shiftMax else (Fmesh.max() + zrange/5)
            Hoptions.update({"vmin":Fmesh.min(), "vmax":zmax})
            # plot distribution as 3d surface with map projection
            mask_test = Fmesh > zmax
            Fmasked = np.ma.masked_where(mask_test, Fmesh)
#            surf = ax.plot_surface(mesh1, mesh2, Fmasked, color='royalblue', shade=True, lightsource=LightSource(azdeg=315,altdeg=45), \
#                    lw=0.5, rstride=1, cstride=1, alpha=0.7, zorder=5)
            surf = ax.plot_trisurf(mesh1[~mask_test], mesh2[~mask_test], Fmasked[~mask_test], zorder=5, \
                    lw=0.1, ec='royalblue', antialiased=True, \
                    color='deepskyblue', shade=True, lightsource=LightSource(azdeg=215,altdeg=45), alpha=0.7) #**Hoptions)
            Ftrim = Fmesh.copy()
            Ftrim[mask_test] = zmax
            img = ax.contourf(mesh1, mesh2, Ftrim, zdir='z', offset=zmin, zorder=3, **Hoptions)
#            img = ax.contourf(mesh1, mesh2, Fmasked, zdir='z', offset=zmin, zorder=3, **Hoptions)
            ax.set_zlim(zmin, zmax)
#            ax.set_zlabel(r'Free Energy  ($\beta F / N$)', labelpad=38)
            ax.set_zlabel(r'Free Energy', labelpad=45)
        else:
            ax = fig.add_subplot(111)
            # plot distribution as 2d map
            if contour:
                zmax = (Fmesh.min() + 1.3*shiftMax) if shiftMax else (Fmesh.max() + zrange/5)
                Hoptions.update({"vmin":Fmesh.min(), "vmax":zmax})
                mask_test = Fmesh > zmax
                Ftrim = Fmesh.copy()
                Ftrim[mask_test] = zmax
                img = ax.contourf( mesh1, mesh2, Ftrim, **Hoptions)
            else:
                img = ax.pcolor( mesh1, mesh2, Fmesh , **Hoptions)
            lp = None
        # display options
        ax.set_xlim(var1.min(), var1.max())       # padding on low & high ends (just visual)
        ax.set_ylim(var2.min(), var2.max())       #
        ax.set_xlabel(Xlab, labelpad=14)
        ax.set_ylabel(Ylab, labelpad=18)
        ax.set_title("Total Free Energy for sequence '%s' \n(%s)" % (self.seqName, titlestr))
#        divider = make_axes_locatable(ax)
#        cax = divider.append_axes("right", size="4%", pad=0.05)
#        fig.colorbar(img, cax=cax)
        fig.colorbar(img)
        if SHOW:
            # show
            fig.tight_layout()
            plt.show()
            plt.close()
        return ax

    #   function for updating values and staying within bounds -> 'wrap' around inteval boundary [for 'perturbTNC']
    def boundUpdate(self, new, bounds):
        # recursive: adjust bounds by 'wrapping' around problem boundary,
        #   then pass back through to ensure new result is within bounds; re-wrap repeatedly as necessary
        if new < bounds[0]:
            diff = bounds[0] - new
            return (self.boundUpdate(bounds[0]+diff, bounds))
        elif new > bounds[1]:
            diff = new - bounds[1]
            return (self.boundUpdate(bounds[1]-diff, bounds))
        else:
            return new

    #   polymer free energy - Simple Model, 'Xonly' formulation, basically F5 alone
    def F_sm(self, alP, alM, x, cs=0, l=1, lB=2, p=0, w2=0, w3=0.1, includeBases=True, dipoleD2factor=True, kill_kappa=False):
        kl = DTYPE(0) if kill_kappa else DTYPE(self.kapl(alP,alM,0,cs,l,lB))
        return self.F5(alP, alM, x, kl=kl, l=l, lB=lB, p=p, w2=w2, w3=w3, includeBases=includeBases, dipoleD2factor=dipoleD2factor)

    #   polymer free energy - effective, from Higgs-Joanny formulation [JCP 1991]
    def F_hj(self, alP, alM, x, cs=0, l=1, lB=2, w2=0, w3=0):
        total = alP*self.fracp + alM*self.fracm
        diff = alP*self.fracp - alM*self.fracm
        kl = self.kapl(alP, alM, 0, cs, l, lB)
        amph = - PI*(lB*lB)*(total*total)/kl
        elec = 4*PI*lB*(diff*diff)/(kl*kl)
        v_star = w2 + amph + elec
#        om = (4/3)*np.sqrt(self.N)      # continuum limit
        om = self.Onon
        o3 = w3 * np.power(2*PI*x/3, -3) * self.B / 2       # including 3-body term
        return ( (3/2)*(x-np.log(x)) + np.power(2*PI*x/3, -1.5) * v_star * om + o3)

    #   minimize free energy for _simple models_ : either 'F5' alone, or HJ formulation
    def opt_sm(self, alP, alM, F5orHJ='F5', xinit=0.5, xinit2=None, x_bound=(1e-3,35), thr=1e-6,  info=False):
        if F5orHJ == 'F5':
            if 'includeBases' not in self.pars:
                self.pars.update({'includeBases':True})
            if 'dipoleD2factor' not in self.pars:
                self.pars.update({'dipoleD2factor':True})
            if 'kill_kappa' not in self.pars:
                self.pars.update({'kill_kappa':False})
            if info:
                self.parInfo()
            pdt = {k:self.pars[k] for k in ('cs', 'l', 'lB', 'p', 'w2', 'w3', 'includeBases', 'dipoleD2factor', 'kill_kappa')}
            func = lambda xl: self.F_sm(alP, alM, xl[0], **pdt)
        else:
            pdt = {k:self.pars[k] for k in ('cs', 'l', 'lB', 'w2', 'w3')}
            func = lambda xl: self.F_hj(alP, alM, xl[0], **pdt)
        if not xinit:
            xlist = np.linspace(x_bound[0], x_bound[1], round(10*x_bound[1]))
            flist = [func([xv]) for xv in xlist]
            ind = np.argmin(flist)
            xinit = xlist[ind]
        opt = minimize(func, (xinit,), method="Nelder-Mead", bounds=(x_bound,), tol=thr)
        xres = (opt.x)[0]
        fres = opt.fun
        if xinit2:
            opt2 = minimize(func, (xinit2,), method="Nelder-Mead", bounds=(x_bound,), tol=thr)
            if opt2.fun < opt.fun:
                xres = (opt2.x)[0]
                fres = opt2.fun
        return xres, fres

    #   finding two-body exclusion 'w2' (or 'v') under Higgs-Joanny model, given 'x' (at some salt)
    def findW2fromHJ(self, alP=1, alM=1, x=1, cs=0, l=1, lB=2, w3=0):
        total = alP*self.fracp + alM*self.fracm
        diff = alP*self.fracp - alM*self.fracm
        kl = self.kapl(alP, alM, 0, cs, l, lB)
        amph = - PI*(lB*lB)*(total*total)/kl
        elec = 4*PI*lB*(diff*diff)/(kl*kl)
#        om = (4/3)*np.sqrt(self.N)      # continuum limit
        om = self.Onon
        res = np.power(x, 2.5) - np.power(x, 1.5)
        res -= w3 * np.power(2*PI*np.sqrt(x)/3, -3) * self.B       # 3-body term ~ derivative
        res *= np.power(2*PI/3, 1.5) / om
        return (res - amph - elec)

    #   using x-only formulation (i.e. just F5) to solve for 2-body term
    def findW2fromF5(self, alP=1, alM=1, x=1, cs=0, rho=5e-4, w3=0.1, l=1, lB=2, p=0, xinit=0.5, xinit2=None, x_thr=1e-4, x_bound=(1e-3,35)):
        # handle screening
        kl = self.kapl(alP,alM,rho,cs,l,lB)
        # numerator
        num = 1.5*(1-1/x)       # Gaussian chain term
        num += - w3 * (3/x) * ((3/(2*PI*x))**3) * self.B / 2      # 3-body term # NEW FIX!! division by '2'
        num += (2*lB/(PI*l)) * self.Q(alP,alM,x,kl, derivative=True)      # electrostatics _derivative_ (d/dx)
        # denominator
        den = (3/(2*x)) * ((3/(2*PI*x))**(1.5)) # factor on 2-body term(s)
        # divide
        res = num / den
        # subtract other 2-body terms (i.e. dipoles) : just full 'Omega' term, with w2=0
        res += - self.Omega(alP,alM,kl,l,lB,0,p)
        # divide by 2-body volume sum factor (akin to 3-body 'B') to get final result
        res /= self.Onon
        # use result of 'w2' to optimize w/r/t 'x' only, to confirm self-consistency
        x_func = lambda xvar: self.F5(alP, alM, xvar, kl=kl, l=l, lB=lB, p=p, w3=w3, w2=res)
        minfunc = lambda xvar: x_func(xvar) - x_func(1.0)
        if not xinit:
            xlist = np.linspace(x_bound[0], x_bound[1], round(10*x_bound[1]))
            flist = [x_func(xv) for xv in xlist]
            ind = np.argmin(flist)
            xinit = xlist[ind]
        x_opt = minimize(minfunc, x0=(xinit,), method="Nelder-Mead", bounds=(x_bound,))
        x_sol = x_opt.x[0]
        if xinit2:
            x_opt2 = minimize(minfunc, x0=(xinit2,), method="Nelder-Mead", bounds=(x_bound,))
            if x_opt2.fun < x_opt.fun:
                x_sol = x_opt2.x[0]
        if abs(x - x_sol) > x_thr:
            print("WARNING: 2-body solution w2=%1.5f did _not_ reproduce expected minimum x=%2.4f!!" % (res,x))
            print("\t[instead gave x_opt=%2.4f]\n" % x_sol)
            res = None
        return res

    #   using x-only formulation (i.e. just F5) to solve for 2-body term _numerically_
    def num_findW2fromF5(self, alP=1, alM=1, x=1, cs=0, rho=5e-4, w3=0.1, l=1, lB=2, p=0, x_thr=1e-6, notes=False):
        # given 'w2' used as seed / initial point
        try:
            w2_init = pars["w2"]
        except NameError:
            w2_init = 0.5
        # function for difference in 'x' only, to be minimized for w2
        def w2Func(w2):
            x_func = lambda xvar: self.F_sm(alP, alM, xvar, cs=cs, l=l, lB=lB, p=p, w3=w3, w2=w2)
            minfunc = lambda xvar: x_func(xvar) - x_func(1.0)
            x_opt = minimize(x_func, x0=(0.5,), method="Nelder-Mead", bounds=((1e-3,35),))
            x_sol = x_opt.x[0]
#            print(f"\n\t >  w2={w2:.5g}  gives  x={x_sol:.5g}")
            return (x - x_sol)
        if notes:
            t1 = perf_counter()
            print("\nFinding 'w2' by iterating full optimization...")
        # now use SciPy to iterate -> find 'w2' at which the given metric returns zero
        res = root_scalar(w2Func, x0=w2_init, bracket=(-30,30), xtol=x_thr, method="brentq")
#        print(res)
        w2_res = res.root
        if notes:
            t2 = perf_counter()
            print("DONE :\tw2=%2.4f  (elapsed time: %2.3f)" % (w2_res, (t2-t1)))
            print(f"converged\t'{res.converged:}'\nflag\t'{res.flag:}'")
        return w2_res

    #   solve for W2 _numerically_ : calibrate datapoint 'x' at a particular salt value 'cs'  [dimensionless!]
    def findW2(self, x_calib, cs_calib, x_thr=1e-8, diff_metric=(lambda x,y: (y-x)), notes=False,
            optArgs={"method":"NM-TNC", "xBounds":(5e-3, 15), "ref":(0.5,0.5,1.5), "init":(0.8,0.8,2.5)}):
        # grab parameters as they are
        pars = self.pars.copy()
        # ensure use of given salt value
        pars.update( {"cs":cs_calib} )
        # given 'w2' used as seed / initial point
        w2_init = pars["w2"]
        # make function of 'w2' only, which will be numerically solved for zero by repeated optimization
        def w2Func(w2):
            # update parameter set first
            pars.update( {"w2":w2} )
            # optimize at that point
            (alP, alM, x) = self.optimize(parset=pars, **optArgs)
            # check difference (from some metric)
            return diff_metric(x, x_calib)
        if notes:
            t1 = perf_counter()
            print("\nFinding 'w2' by iterating full optimization...")
        # now use SciPy to iterate -> find 'w2' at which the given metric returns zero
#        res = root_scalar(w2Func, x0=w2_init, x1=-w2_init/2, xtol=x_thr, rtol=x_thr, method="secant")
        res = root_scalar(w2Func, x0=w2_init, bracket=(-10,10), xtol=x_thr, method="brentq")
        w2_res = res.root
        if notes:
            t2 = perf_counter()
            print("DONE :\tw2=%2.4f  (elapsed time: %2.3f)" % (w2_res, (t2-t1)))
            print(f"converged\t'{res.converged:}'\nflag\t'{res.flag:}'")
        return w2_res

    #   solve for 'delta' and 'w2' _simultaneously_ : calibrate using datapoints (cs_dlt, x_dlt) & (cs_w2, x_w2)
    def findDandW2(self, x_calibs=[], cs_calibs=[], thr=1e-6,  diff_metric=(lambda x,y: (y-x)), notes=False,
            optArgs={"method":"NM-TNC", "xBounds":(1e-2, 10), "ref":(0.8,0.8,1.5), "init":(0.7,0.7,2.5)}):
        # grab parameters as they are
        pars1 = self.pars.copy()
        # copy parameters for second calibration point (simultaneous)
        pars2 = pars1.copy()
        # ensure use of given salt values
        pars1.update( {"cs":cs_calibs[0]} )
        pars2.update( {"cs":cs_calibs[1]} )
        # given 'delta' and 'w2' used as seed / initial point
        dlt_init = pars1["delta"]
        w2_init = pars1["w2"]
        # make _vector_ function of 'delta' and 'w2', which will be numerically solved for zero by repeated optimization
        def dw2Func(vec):
            (d, w2) = vec       # unpack single vector argument
            # update parameter sets first
            pars1.update( {"delta":d, "w2":w2} )
            pars2.update( {"delta":d, "w2":w2} )
            # optimize at those points
            (alP1, alM1, x1) = self.optimize(parset=pars1, **optArgs)
            (alP2, alM2, x2) = self.optimize(parset=pars2, **optArgs)
            # check difference (from some metric)
            return [diff_metric(x1, x_calibs[0]), diff_metric(x2, x_calibs[1])]
        if notes:
            t1 = perf_counter()
            print("\nFinding 'delta' & 'w2' by iterating full optimization...")
        # now use SciPy to iterate -> find pair ('delta', 'w2') at which the given metric returns zero _vector_
        res = root(dw2Func, (dlt_init, w2_init), method="df-sane", tol=thr)
        (d_res, w2_res) = tuple(res.x)
        if notes:
            t2 = perf_counter()
            print("DONE :\tw2=%2.4f  (elapsed time: %2.3f)" % (w2_res, (t2-t1)))
        # return as _dictionary_
        return ({"delta":d_res, "w2":w2_res})

