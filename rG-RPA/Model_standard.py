##  Mike Phillips, 6/9/2022
##  * rG-RPA Class Definition *
##  Unified structure for all details in rG-RPA model
##  - takes 'Sequence' object: sequence dependence inherited
##  - handle / store all parameters necessary for model itself
##      > solute size 'Na'
##      > counter-ion valence 'zc'
##      > salt concentration 'salt', and valence 'zs'
##      > 2-body volume exclusion 'v2'
##      > Flory-Huggins interaction offset 'epsa'    [added: 8/18/2023]
##      > Flory-Huggins interaction parameter 'epsb'    [added: 6/23/2022]
##      > 3-body Flory-Huggins interaction parameter 'chi3'    [added: 8/23/2023]
##      > electrostatic potential choice 'potential'
##      > effective size of ions (counter-ions and salt), 'ionsize'
##      > fixed-Gaussian or renormalized-Gaussian, 'mode'
##      > threshold for zero solver (for 'x', if 'rG' mode), 'Xthr'
##  - define all necessary functions for model
##      > Free Energy (and derivatives): entropy + ions + polymers
##      > options: entropy (w/ or w/o counter-ions, salt), ions (point-like or smeared), polymer (combined or alone)
##      > main functions: i.e. 'xi', 'g', 'zeta' and their variants / derivatives
##      > many simple support functions from 'Xfuncs'
##      > others, like integrands (with more parameter dependence), kept with the object
##
##  Note (6/13/2022) -- This is the standard setup, with simplistic self-energy corrections.
##   > Free energy integrands use corrections with Taylor expansion just for large 'k'.
##      -> derivatives (except first) simply do not have self-energy corrections.
##
##  + UPDATE (4/14/2023) -- including net charge mean-field correction!  proportional to phi^2

import numpy as np
import scipy.optimize as opt
import scipy.integrate as integ
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from time import perf_counter

import Xfuncs_standard as Xfuncs           # support functions for 'x' solvers (related to integrands)

##  GLOBALS
DTYPE = np.float64  # unified data type
PI = DTYPE(np.pi)   # appropriate value for 'pi'
eps = 1e-15         # small number to avoid issues (i.e. log(eps)=number)

# print quick message before anything else...
print("\n > LOADED module 'Model_standard'.\n")

#####   Class Defnition : object for handling model paremeters and all complicated details
class RPA:
    def __init__(self, seq, pars="default"):
        self.small = 1e3 * eps  # small cutoff for logs
        self.seq = seq      # store sequence object
        self.setPars(pars)  # set parameters (at least default ones)
    #   #   #   #   #   #   #

    #   copy -- make identical copy for future reference (independent of original)
    def copy(self):
        if self.parChoice == "custom":
            pars = self.pars
        else:
            pars = self.parChoice
        return RPA(self.seq, pars)

    #   set basic parameters
    def setPars(self, pset="default"):
        """
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
        """
        ##   define parameters, etc.
        labels = ("Na", "cions", "salt", "zs", "v2", "epsa", "epsb", "chi3", "potential", "ionsize", "mode", "Xthr", "mean-field")
        vals = (1, None, 0, 0, 0, 0, 0, 0, "coulomb", "smear", "fixed-G", 1e-8, False)
        pdict = dict(zip(labels,vals))
        if type(pset) == str:
            self.parChoice = pset
            pset_l = pset.lower()
            if "cion" in pset_l:
                pdict.update( {"cions":1} )     # parameter here is valence charge of counter-ions, z_c
            if "salt" in pset_l:
                pdict.update( {"salt":0.005} )  # note: salt is in dim.less units, phi_s = rho_s[M]*(6.022e-4)*(l[A]^3) ; default is 150 mM equivalent
                pdict.update( {"zs":1} )    # default valency of salt ions is 1
            if ("fixed" in pset_l) or ("fg" in pset_l):
                pdict.update( {"mode":"fixed-G"} )
            elif ("renorm" in pset_l) or ("rg" in pset_l):
                pdict.update( {"mode":"renorm-G"} )
            if "coulomb" in pset_l:
                pdict.update( {"potential":"coulomb"} )
            elif "short" in pset_l:
                pdict.update( {"potential":"short"} )
            if ("volume" in pset_l) or ("v2" in pset_l):
                pdict.update( {"v2":(4*PI/3)} )
            if "point" in pset_l:
                pdict.update( {"ionsize":"point"} )
            elif "smear" in pset_l:
                pdict.update( {"ionsize":"smear"} )
            if ("eps" in pset_l) or ("fh" in pset_l) or ("flory" in pset_l):
#                pdict.update( {"epsb":1} )
                pdict.update( {"epsb":0.5} )
            if ("mf" in pset_l) or ("mean" in pset_l):
                pdict.update( {"mean-field":True} )
        elif type(pset) == dict:
            self.parChoice = "custom"
            pdict.update(pset)
        else:
            print("\nERROR: 'pset' must be entered as string [choice] or dict [pars] (given '%s').\n" % pset)
            return
        # check for valid settings: 'mode' and 'potential' and 'ionsize'
        mchoice = pdict["mode"]
        mchar = mchoice.lower()[0]    # first character of 'mode' setting
        mode_tags = ("f", "r")      # allowed options for mode
        if mchar not in mode_tags:
            print("\nWARNING: invalid mode provided ('%s'); reverting to default setting 'fixed-G'." % mchoice)
            pdict.update( {"mode":"fixed-G"} )
        scut = pdict["potential"]
        schar = scut.lower()[0]
        potential_tags = ("c", "s")     # allowed options for potential setting
        if schar not in potential_tags:
            print("\nWARNING: given potential tag '%s' is not valid -> using default potential (full-range)." % scut)
            schar = "c"
            pdict.update( {"potential":"coulomb"} )
        ichoice = pdict["ionsize"]
        ichar = ichoice.lower()[0]
        isize_tags = ("p", "s")
        if ichar not in isize_tags:
            print("\nWARNING: invalid ionsize provided ('%s'); reverting to default setting 'point'." % ichoice)
            pdict.update( {"ionsize":"point"} )
        elif ichar == "s" and schar != "c":
            print("\nWARNING: Coulomb potential is required in use of 'smeared' ions; setting potential to 'coulomb'.")
            schar = "c"
            pdict.update( {"potential":"coulomb"} )
        # set Fourier transform of potential, _!_ pre-multiplied by k^2 / l _!_
        # note: screening enters in a different way, with explicit counter-ions + salt
        if schar == "c":        # coulomb, full-range, 'regular' electrostatic potential
            # i.e. from U(r) = 1/r
            self.lam = lambda k,t: DTYPE( 4*PI/t )
        elif schar == "s":      # short-range coulomb, cut off at scale 'a', modified electrostatic potential
            # i.e. from U(r) = (1-exp(-r/a))/r
            a = 1       # length scale for potential, in units of bare Kuhn length 'l'
            self.lam = lambda k,t: DTYPE( 4*PI/(t*(1+((k*a*k*a)))) )    # better squaring (no '**2')
        # after all validation checks -> store dictionary of parameters
        self.pars = pdict.copy()
        # use some settings to evaluate / store others
        cions = pdict["cions"]
        if cions:           # handle counter-ions (charge balancing using given valence)
            self.zc = abs(cions)    # abs. value of charge only - always opposite sign from chains
            if np.isclose(self.seq.qtot,0) or np.isclose(self.zc,0):
                print("\n\tNOTE: sequence is neutral, setting ion charge zc=0...\n")
                self.zc = 0
                self.cionpar = 0
            else:
                # !! FOLLOWING NEEDS ADJUSTMENT TO CAPTURE NONZERO SALT!!  (_only_if_ salt is not neutral overall)
                self.cionpar = np.abs(self.seq.qtot) / (self.seq.N * self.zc)  # useful shortcut parameter
        else:
            self.zc = 0
            self.cionpar = 0
        if pdict["salt"]:
            if pdict["zs"] == 0:
                self.zs = 1     # valence charge on salt ions: assume equal to 1 (both + and - ions are present in salt)
            else:
                self.zs = pdict["zs"]
        else:
            self.zs = 0     # ensure zero salt charge if no salt is present
        # set remaining entries as attributes (faster evaluation)
        self.Na = pdict["Na"]
        self.cions = cions
        self.salt = pdict["salt"]
        self.v2 = pdict["v2"]
        self.epsa = pdict["epsa"] if 'epsa' in pdict else 0
        self.epsb = pdict["eps0"] if 'eps0' in pdict else pdict['epsb']
        self.chi3 = pdict["chi3"] if 'chi3' in pdict else 0
        self.Xthr = pdict["Xthr"]
        self.potential = pdict["potential"]
        self.mode = pdict["mode"]
        self.ionsize = pdict["ionsize"]
        self.cfac = (self.zc*self.zc) * self.cionpar
        self.sfac = (self.zs*self.zs) * 2 * self.salt        # factor of 2 used -> salt has + and - ions!
        self.isMF = pdict["mean-field"]     # ! NEW (apr 2023) !  enable mean-field corrections with net charge!
        self.isMF = ( self.isMF and self.potential=="coulomb" and ((self.sfac + self.cfac) > 0) )     # ensure coulomb, and ions are present
        if pdict["mean-field"] and not self.isMF:
            print("\n  * WARNING : attempted to include mean-field net charge contribution without appropriate potential, or without ions; reverting...\n")
        elif self.isMF and np.isclose(self.sfac, 0.):
            print("\n  * NOTE : mean-field correction will not have a meaningful effect at zero salt.\n")
        # store some ranges raised to powers (up to 5th)
        self.range1 = np.array(self.seq.refRange1)
        self.range2 = self.range1 * self.range1
        self.range3 = self.range2 * self.range1
        self.range4 = self.range3 * self.range1
        self.range5 = self.range4 * self.range1
        self.range0 = np.ones(len(self.range1))        # INCL. SELF ENERGY -> use 'ones' here, 'Xfuncs' boolean shift
        # entropy setting uses counter-ions and salt
        (self.fs0, self.fs1, self.fs2, self.fs3) = self.set_entropy()
        # integrand setting uses 'mode' (fG or rG)
        (self.int0, self.int1, self.int2, self.int3) = self.set_integrands()
        # correlation (structure) function setting uses 'v2' (simplified only if v2=0)
        (self.realG, self.realMinig, self.realZ) = self.set_corrs()
        return pdict

    #   print table of parameters
    def info(self, exclude=("Na", "zs", "Xthr")):
        print("\n" + "*****   "*5)
        print("  RPA parameters:\t%s" % self.parChoice)
        print("\n\t{:<10}\t{:<18}".format("PAR", "VALUE"))
        excludedKeys = [k for k in self.pars.keys() if k not in exclude]
        for par in excludedKeys:
            try:
                p = float(self.pars[par])
                print("\t{:<10}\t{:<18.5g}".format(par, p))
            except:
                print("\t{:<10}\t{:<18}".format(par, str(self.pars[par])))
        print("*****   "*5)
        print("")
        return

    #   set entropy function and 3 derivatives, handling presence/absence of counter-ions _and_ salt
    def set_entropy(self):
        A = self.cionpar
        B = 1 + self.cionpar
        N = self.seq.N
        Na = self.Na
        small = self.small
        salt = self.salt
        upper = (1-2*salt)/B
        if A:
            Apart = lambda p: DTYPE( (p*A)*np.log(p*A) )
            Apart1 = lambda p: DTYPE( A*(np.log(p*A)+1) )
            Apart2 = lambda p: DTYPE( A/p )
            Apart3 = lambda p: DTYPE( -A/(p*p) )
        else:
            Apart = lambda p: DTYPE(0)
            (Apart1, Apart2, Apart3) = (Apart, Apart, Apart)
        if salt:
            salt_shift = 2*salt*np.log(2*salt)      # constant shift for salt (technically unnecessary)
        else:
            salt_shift = 0
        def fs(phi):
            if phi < small:
                res = salt_shift + ((1-2*salt)/Na)*np.log(1-2*salt)
                return DTYPE(0)
            elif phi > upper-small:
                res = (upper/N)*np.log(upper-small) + Apart(upper-small) + salt_shift
                res += (small*B/Na)*np.log(small*B)
                return DTYPE(res)
            else:
                res = (phi/N)*np.log(phi) + Apart(phi) + salt_shift + ((1-phi*B-2*salt)/Na)*np.log(1-phi*B-2*salt)
                return DTYPE(res)
        def fs1(phi):
            if phi < small:
                res = (1/N)*(np.log(small)+1) - (B/Na)*(np.log(1-small*B-2*salt)+1) + Apart1(small)
                return DTYPE(res)
            elif phi > upper-small:
                res = (1/N)*(np.log(upper-small)+1) + Apart1(upper-small)
                res += - (B/Na)*(np.log(small*B)+1)
                return DTYPE(res)
            else:
                res = (1/N)*(np.log(phi)+1) - (B/Na)*(np.log(1-phi*B-2*salt)+1) + Apart1(phi)
                return DTYPE(res)
        def fs2(phi):
            if phi < small:
                res = (1/(small*N)) + (B*B/((1-small*B-2*salt)*Na)) + Apart2(small)
                return DTYPE(res)
            elif phi > upper-small:
                res = (1/((upper-small)*N)) + (B/(small*Na)) + Apart2(upper-small)
                return DTYPE(res)
            else:
                res = (1/(phi*N)) + (B*B/((1-phi*B-2*salt)*Na)) + Apart2(phi)
                return DTYPE(res)
        def fs3(phi):
            if phi < small:
                res = -(1/((small*small)*N)) + (B*B*B/(((1-small*B-2*salt)*(1-small*B-2*salt))*Na)) + Apart3(small)
                return DTYPE(res)
            elif phi > upper-small:
                res = -(1/(((upper-small)*(upper-small))*N)) + (B/((small*small)*Na)) + Apart3(upper-small)
                return DTYPE(res)
            else:
                res = -(1/((phi*phi)*N)) + (B*B*B/(((1-phi*B-2*salt)*(1-phi*B-2*salt))*Na)) + Apart3(phi)
                return DTYPE(res)
        return (fs, fs1, fs2, fs3)

    #   set functions for Free Energy integrands: up to third derivative
    def set_integrands(self):
        # handle smeared ions differently
        ichar = self.ionsize.lower()[0]
        if ichar == "s":
            i0 = self.integrand0_Ponly
            i1 = self.integrand1_Ponly
            i2 = self.integrand2_Ponly
            i3 = self.integrand3_Ponly
            return (i0, i1, i2, i3)
        i0 = self.integrand0        # zeroth derivative (actual FE) -> same for both modes
        mchar = self.mode.lower()[0]    # initial character of 'mode' setting: 'f' for fG _or_ 'r' for rG
        if mchar == "f":       # fixed Gaussian mode (~simple derivatives)
            i1 = self.integrand1_fg
            i2 = self.integrand2_fg
            i3 = self.integrand3_fg
        elif mchar == "r":     # renormalized Gaussian mode (complicated derivatives)
            i1 = self.integrand1_rg
            i2 = self.integrand2_rg
            i3 = self.integrand3_rg
        return (i0, i1, i2, i3)

    #   set correlation (structure) functions actually used ('g' and 'Z' are eliminated if no volume exclusion)
    def set_corrs(self):
        if np.isclose(self.v2, 0):
            return (lambda k2,x,rng_n: Xfuncs.totG(k2,x,self.seq,rng_n), lambda k2,x,rng_n: 0, lambda k2,x,rng_n: 0)
        else:
            return (lambda k2,x,rng_n: Xfuncs.totG(k2,x,self.seq,rng_n), \
                    lambda k2,x,rng_n: Xfuncs.totMinig(k2,x,self.seq,rng_n), \
                    lambda k2,x,rng_n: Xfuncs.totZ(k2,x,self.seq,rng_n) )

    #   free energy of polymer (combined with ions too, only if using point-like ions)
    def fp(self, phi, t, x=None, FintKW={}, XintKW={}):
        mchar = self.mode.lower()[0]    # initial character of 'mode' setting: 'f' for fG _or_ 'r' for rG
        if mchar == "f":
            x = 1       # fixed Gaussian: just use unity
        elif mchar == "r":
            if not x:
                x = self.find_x(phi, t, XintKW)     # renormalized Gaussian: solve for 'x' (if 'x' is not supplied)
        ifunc = lambda k: DTYPE( self.int0(k,phi,t,x) ) # integrand
        return ( self.FE_integrate(ifunc, **FintKW) )   # electronic contribution: polymer (maybe + ions)
    #   free energy of polymer : first derivative
    def dfp(self, phi, t, x=None, dx=None, FintKW={}, XintKW={}):
        mchar = self.mode.lower()[0]    # initial character of 'mode' setting: 'f' for fG _or_ 'r' for rG
        if mchar == "f":
            x = 1       # fixed Gaussian: just use unity
            dx = 0      # any derivatives are zero
        elif mchar == "r":
            if not x:
                x = self.find_x(phi, t, XintKW)             # renormalized Gaussian: solve for 'x'
            if not dx:
                XintD = self.XintD(phi, t, x, XintKW)       # common integral appearing in denominator of 'x' derivatives
                dx = self.find_dx(phi, t, x, XintD, XintKW) # also need derivative
        ifunc = lambda k: DTYPE( self.int1(k,phi,t,x,dx) )  # integrand
        return ( self.FE_integrate(ifunc, **FintKW) )       # electronic contribution: polymer (maybe + ions)
    #   free energy of polymer : second derivative
    def d2fp(self, phi, t, x=None, dx=None, d2x=None, FintKW={}, XintKW={}):
        mchar = self.mode.lower()[0]    # initial character of 'mode' setting: 'f' for fG _or_ 'r' for rG
        if mchar == "f":
            x = 1       # fixed Gaussian: just use unity
            dx = 0      # any derivatives are zero
            d2x = 0     #
        elif mchar == "r":
            if not x:
                x = self.find_x(phi, t, XintKW)             # renormalized Gaussian: solve for 'x'
            if (not dx) or (not d2x):
                XintD = self.XintD(phi, t, x, XintKW)       # common integral appearing in denominator of 'x' derivatives
                if not dx:
                    dx = self.find_dx(phi, t, x, XintD, XintKW)         # also need derivative
                if not d2x:
                    d2x = self.find_d2x(phi, t, x, dx, XintD, XintKW)   #
        ifunc = lambda k: DTYPE( self.int2(k,phi,t,x,dx,d2x) )          # integrand
        return ( self.FE_integrate(ifunc, **FintKW) )       # electronic contribution: polymer (maybe + ions)
    #   free energy of polymer : third derivative
    def d3fp(self, phi, t, x=None, dx=None, d2x=None, d3x=None, FintKW={}, XintKW={}):
        mchar = self.mode.lower()[0]    # initial character of 'mode' setting: 'f' for fG _or_ 'r' for rG
        if mchar == "f":
            x = 1       # fixed Gaussian: just use unity
            dx = 0      # any derivatives are zero
            d2x = 0     #
            d3x = 0     #
        elif mchar == "r":
            if not x:
                x = self.find_x(phi, t, XintKW)             # renormalized Gaussian: solve for 'x'
            if (not dx) or (not d2x) or (not d3x):
                XintD = self.XintD(phi, t, x, XintKW)       # common integral appearing in denominator of 'x' derivatives
                if not dx:
                    dx = self.find_dx(phi, t, x, XintD, XintKW)             # also need derivative
                if not d2x:
                    d2x = self.find_d2x(phi, t, x, dx, XintD, XintKW)       #
                if not d3x:
                    d3x = self.find_d3x(phi, t, x, dx, d2x, XintD, XintKW)  #
        ifunc = lambda k: DTYPE( self.int3(k,phi,t,x,dx,d2x,d3x) )          # integrand
        return ( self.FE_integrate(ifunc, **FintKW) )       # electronic contribution: polymer (maybe + ions)

    #   Shortcut function: Debye screening 'kappa' (dimensionless)
    def kappa(self, phi, t):
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        kap = np.sqrt( self.lam(0,t) * (salt_fac + cion_fac*phi) ) * (phi > 0)  # ensure physical result
        return DTYPE(kap)
    #   JUST IONIC part of free energy, to go with above polymer-only pieces    - zeroth derivative
    def F_ionsonly(self, phi, t):
        fac = -1/(4*PI)
        kap = self.kappa(phi, t)        # Debye screening (dimensionless)
        if np.isclose(kap,0):
            return DTYPE(0)
        terms = np.log(1+kap) - kap*(1 - 0.5*kap)
        return DTYPE(fac * terms)
    #   JUST IONIC part of free energy, to go with above polymer-only pieces    - first derivative
    def dF_ionsonly(self, phi, t):
        fac = -1/(8*PI)
        kap = self.kappa(phi, t)        # Debye screening (dimensionless)
        if np.isclose(kap,0):
            return DTYPE(0)
        lamf = self.lam(0,t) * self.cfac
        terms = lamf*kap/(1+kap)
        return DTYPE(fac * terms)
    #   JUST IONIC part of free energy, to go with above polymer-only pieces    - second derivative
    def d2F_ionsonly(self, phi, t):
        fac = -1/(16*PI)
        kap = self.kappa(phi, t)        # Debye screening (dimensionless)
        if np.isclose(kap,0):
            return DTYPE(0)
        lamf = self.lam(0,t) * self.cfac
        terms = ( (lamf/(1+kap))*(lamf/(1+kap)) ) / kap
        return DTYPE(fac * terms)
    #   JUST IONIC part of free energy, to go with above polymer-only pieces    - third derivative
    def d3F_ionsonly(self, phi, t):
        fac = 1/(32*PI)
        kap = self.kappa(phi, t)        # Debye screening (dimensionless)
        if np.isclose(kap,0):
            return DTYPE(0)
        lamf = self.lam(0,t) * self.cfac
        terms = ( (lamf/(kap*(1+kap)))*(lamf/(kap*(1+kap)))*(lamf/(kap*(1+kap))) ) * (1+3*kap)
        return DTYPE(fac * terms)

    #   free energy at given phi & temp. (per kT*M)
    def F(self, phi, t, x=None, FintKW={}, XintKW={}):
        # handle smeared ions -> separate contribution
        ichar = self.ionsize.lower()[0]
        if ichar == "s":
            fions = self.F_ionsonly(phi, t)
        else:
            fions = 0
        fs = self.fs0(phi)   # zeroth derivative of entropy contribution
        fel = self.fp(phi,t,x,FintKW,XintKW) + fions    # total electronic contribution: ions + polymer
        f0 = (self.v2) * (phi*phi) / 2      # mean-field contribution (k=0)
        ffh = (self.epsa + (self.epsb / t)) * phi * (1 - phi - (self.cionpar*phi) - self.salt)    # Flory-Huggins type interaction
        ffh += self.chi3 * phi*phi*phi      # 3-body Flory-Huggins type contribution
        # ! NEW (apr 2023) !    extra mean-field contribution related to net charge (if desired)
        if self.isMF:
            lam0 = (phi>0) / (self.sfac + self.cfac*phi)    # also ensuring positive phi
            qm = self.seq.qtot / self.seq.N     # net charge per monomer
            fmf = 0.5 * lam0 * phi*phi * qm*qm
        else:
            fmf = 0
        return DTYPE( fs + fel + f0 + ffh + fmf )

    #   chemical potential [slope] (dF/dphi) : for Maxwell construction
    def dF(self, phi, t, x=None, dx=None, FintKW={}, XintKW={}):
        # handle smeared ions -> separate contribution
        ichar = self.ionsize.lower()[0]
        if ichar == "s":
            fions = self.dF_ionsonly(phi, t)
        else:
            fions = 0
        fs = self.fs1(phi)   # first derivative of entropy contribution
        fel = self.dfp(phi,t,x,dx,FintKW,XintKW) + fions     # total electronic contribution: ions + polymer
        f0 = self.v2 * phi      # mean-field contribution (k=0)
        ffh = (self.epsa + (self.epsb / t)) * (1 - 2*phi - 2*(self.cionpar*phi) - self.salt)    # Flory-Huggins type interaction
        ffh += 3 * self.chi3 * phi*phi
        # ! NEW (apr 2023) !    extra mean-field contribution related to net charge (if desired)
        if self.isMF:
            lam0 = (phi>0) / (self.sfac + self.cfac*phi)    # also ensuring positive phi
            qm = self.seq.qtot / self.seq.N     # net charge per monomer
            fmf = (1 - 0.5 * lam0 * self.cfac * phi) * lam0 * phi * qm*qm
        else:
            fmf = 0
        return DTYPE( fs + fel + f0 + ffh + fmf )

    #   curvature (d2F/dphi2) : for building spinodal & critical point
    def d2F(self, phi, t, x=None, dx=None, d2x=None, FintKW={}, XintKW={}):
        # handle smeared ions -> separate contribution
        ichar = self.ionsize.lower()[0]
        if ichar == "s":
            fions = self.d2F_ionsonly(phi, t)
        else:
            fions = 0
        fs = self.fs2(phi)   # second derivative of entropy contribution
        fel = self.d2fp(phi,t,x,dx,d2x,FintKW,XintKW) + fions     # total electronic contribution: ions + polymer
        f0 = self.v2            # mean-field contribution (k=0)
        ffh = (self.epsa + (self.epsb / t)) * (-2) * (1 + (self.cionpar))    # Flory-Huggins type interaction
        ffh += 6 * self.chi3 * phi
        # ! NEW (apr 2023) !    extra mean-field contribution related to net charge (if desired)
        if self.isMF:
            lam0 = (phi>0) / (self.sfac + self.cfac*phi)    # also ensuring positive phi
            qm = self.seq.qtot / self.seq.N     # net charge per monomer
            fmf = (1 - 2 * lam0 * self.cfac * phi + lam0*lam0 * self.cfac*self.cfac * phi*phi) * lam0 * qm*qm
        else:
            fmf = 0
        return DTYPE( fs + fel + f0 + ffh + fmf )

    #   third derivative (d3F/dphi3) : for building / checking critical point
    def d3F(self, phi, t, x=None, dx=None, d2x=None, d3x=None, FintKW={}, XintKW={}):
        # handle smeared ions -> separate contribution
        ichar = self.ionsize.lower()[0]
        if ichar == "s":
            fions = self.d3F_ionsonly(phi, t)
        else:
            fions = 0
        fs = self.fs3(phi)   # third derivative of entropy contribution
        fel = self.d3fp(phi,t,x,dx,d2x,d3x,FintKW,XintKW) + fions     # total electronic contribution: ions + polymer
        # NO mean-field contribution (k=0) in third derivative
        # NO Flory-Huggins type interaction in third derivative -- except 3-body
        ffh = 6 * self.chi3
        # ! NEW (apr 2023) !    extra mean-field contribution related to net charge (if desired)
        if self.isMF:
            lam0 = (phi>0) / (self.sfac + self.cfac*phi)    # also ensuring positive phi
            qm = self.seq.qtot / self.seq.N     # net charge per monomer
            fmf = (- 3 * self.cfac + 6 * lam0 * self.cfac*self.cfac * phi - 3 * lam0*lam0 * self.cfac*self.cfac*self.cfac * phi*phi) * lam0*lam0 * qm*qm
        else:
            fmf = 0
        return DTYPE( fs + fel + ffh + fmf )

    #   function that should be zero at appropriate 'x'
    def Xzero(self, x, phi, t, intKW={}):
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        # function for 'k' integral
        ifunc = lambda k: DTYPE( self.J(k,phi,t,x,cion_fac,salt_fac) )
        intJ = self.X_integrate(ifunc, **intKW)
        return DTYPE(1 - 1/x - intJ)

    #   find 'x' under renormalized Gaussian setup   [also need 3 derivatives w/r/t phi]
    def find_x(self, phi, t, intKW={}, notes=False):
        N = self.seq.N
        if notes:
            print("\n\tFinding 'x' at (phi,t)=({:.4g},{:.4g}) ... ".format(phi,t))
            t1 = perf_counter()
        # use a root finder to get 'x'
        sol = opt.root_scalar(self.Xzero, args=(phi, t, intKW),
                rtol=self.Xthr, method="brenth", x0=0.9, x1=3.0, bracket=(1/(N*1000),10000*N))
        xres = sol.root
#        # function that should be minimized (i.e. zero) at the solution for 'x'
#        iscale = 1
#        def Xmin(x):
#            # function for 'k' integral
#            ifunc = lambda k: DTYPE( iscale * self.J(k,phi,t,x,cion_fac,salt_fac) )
#            intJ = self.X_integrate(ifunc, **intKW) / iscale
#            return np.abs(1 - 1/x - intJ)
#        # use minimizer to get 'x'
#        sol = opt.minimize_scalar(Xmin, bounds=(1e-2, 50), method="bounded")
#        xres = DTYPE(sol.x)
#        # function for finding a fixed point: f(x)=x
#        iscale = 1
#        def Xfixed(x):
#            # function for 'k' integral
#            ifunc = lambda k: DTYPE( iscale * self.J(k,phi,t,x,cion_fac,salt_fac) )
#            intJ = self.X_integrate(ifunc, **intKW) / iscale
#            return (1 - 1/x - intJ + x)
#        # use a fixed point finder to get 'x'
#        sol = opt.fixed_point(Xfixed, 1.0, xtol=self.thr, method="del2")
#        xres = DTYPE(sol)
        if notes:
            t2 = perf_counter()
            print("\t  FOUND x={:.4g}\t(elapsed time: {:.4g})\n".format(xres, t2-t1))
        return DTYPE(xres)

    #   denominator appearing in all 'x' derivatives is a common integral
    def XintD(self, phi, t, x, intKW={}):
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        # function for 'k' integral -> be sure to include factor of k^4 here!
        ifunc = lambda k: DTYPE( self.dJdx(k,phi,t,x,cion_fac,salt_fac) )
        intD = self.X_integrate(ifunc, **intKW)
        return DTYPE( intD )

    #   derivatives are analytic!   just need some integral values first, and lower derivatives
    def find_dx(self, phi, t, x, intD, intKW={}, notes=False):
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        if notes:
            print("\n\tFinding 'dx' at (phi,t)=({:.4g},{:.4g}), x={:.4g} ... ".format(phi,t,x))
            t1 = perf_counter()
        # offset
        dlt = 0
        # function for 'k' integral
        ifunc = lambda k: DTYPE( self.dJdp(k,phi,t,x,cion_fac,salt_fac) )
        intN = self.X_integrate(ifunc, **intKW)
        # no need for a solver; solution is simple ratio
        dxres = (intN + dlt) / ( (1/(x*x)) - intD )
        if notes:
            t2 = perf_counter()
            print("\t  FOUND dx={:.4g}\t(elapsed time: {:.4g})\n".format(dxres, t2-t1))
        return DTYPE( dxres )

    def find_d2x(self, phi, t, x, dx, intD, intKW={}, notes=False):
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        if notes:
            print("\n\tFinding 'd2x' at (phi,t)=({:.4g},{:.4g}), x={:.4g}, dx={:.4g} ... ".format(phi,t,x,dx))
            t1 = perf_counter()
        # offset
        dlt = 2 * ((dx/x)*(dx/x)) / x
        # function for 'k' integral
        ifunc = lambda k: DTYPE( self.XintN2(k,phi,t,x,dx,cion_fac,salt_fac) )
        intN = self.X_integrate(ifunc, **intKW)
        # no need for a solver; solution is simple ratio
        d2xres = (intN + dlt) / ( (1/(x*x)) - intD )
        if notes:
            t2 = perf_counter()
            print("\t  FOUND d2x={:.4g}\t(elapsed time: {:.4g})\n".format(d2xres, t2-t1))
        return DTYPE( d2xres )

    def find_d3x(self, phi, t, x, dx, d2x, intD, intKW={}, notes=False):
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        if notes:
            print("\n\tFinding 'dx' at (phi,t)=({:.4g},{:.4g}), x={:.4g}, dx={:.4g}, d2x={:.4g} ... ".format(phi,t,x,dx,d2x))
            t1 = perf_counter()
        # offset
        dlt = 6 * ( d2x*dx - (dx*dx*dx)/x ) / (x*x*x)
        # function for 'k' integral
        iscale = 1
        ifunc = lambda k: DTYPE( iscale * self.XintN3(k,phi,t,x,dx,d2x,cion_fac,salt_fac) )
        intN = self.X_integrate(ifunc, **intKW) / iscale
        # no need for a solver; solution is simple ratio
        d3xres = (intN + dlt) / ( (1/(x*x)) - intD )
        if notes:
            t2 = perf_counter()
            print("\t  FOUND d3x={:.4g}\t(elapsed time: {:.4g})\n".format(d3xres, t2-t1))
        return DTYPE( d3xres )


    #  manipulation plot of essential integrand in free energy
    def intplot(self, iphi=0.5, it=0.8, krng=(eps,50), kpts=250, sz=7, x=None):
        # integrand function
        (phi, t) = (iphi, it)
        (k1, k2) = krng
        kspace = np.linspace(k1, k2, kpts)
        mchar = self.mode.lower()[0]    # initial character of 'mode' setting: 'f' for fG _or_ 'r' for rG
        if mchar == "f":
            x = 1       # fixed Gaussian: just use unity
        elif mchar == "r":
            if not x:
                x = self.find_x(phi, t, XintKW)     # renormalized Gaussian: solve for 'x' (if 'x' is not supplied)
        y = self.int0(kspace, phi, t, x)
        # asymptotic form
#        asyfunc = lambda k,phi,t: - (0.5) * (( 4*PI*phi*self.seq.totalsig / (t*self.seq.N*k) )**2)
#        yasy = asyfunc(kspace, phi, t)
        # plot
        fig = plt.figure("Integrand Manipulation", (1.4*sz,sz))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.3)
        line, = ax.plot(kspace, y, "-<", markersize=1)
#        lasy, = ax.plot(kspace, yasy, "--o", markersize=0)
        ax.set_xlim(k1,k2+1)
        ax.set_ylim(min(y),max(y)+(0.1))
        ax.set_xlabel(r"dimensionless k-space, $k\ell$")
        ax.set_ylabel(r"integrand")
        ax.set_title(r"integrand from $f_{el}$ : $N_B=$%i, type '%s'" % (self.seq.N, self.seq.seqAlias))
        # manipulate
        axcolor = '#00035b'
        axphi = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        axx = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        sphi = Slider(axphi, r'$\phi$', 1e-6, 1-(1e-6), valinit=phi, valstep=0.01)
        st = Slider(axx, r'~$\ell/\ell_B$', 0.1, 3.0, valinit=t, valstep=0.01)
        def update(val):
            phi = sphi.val
            t = st.val
            y = self.int0(kspace,phi,t,x)
#            yasy = asyfunc(kspace, phi, t)
            line.set_ydata(y)
#            lasy.set_ydata(yasy)
            ax.set_ylim(min(y),max(y)+0.1)
            fig.canvas.draw_idle()
        sphi.on_changed(update)
        st.on_changed(update)
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'reset', color=axcolor, hovercolor='#929591')
        def reset(event):
            sphi.reset()
            st.reset()
        button.on_clicked(reset)
        plt.show()
        plt.close()

    #   generalized integrator function for Free Energies : takes scaled integrand function as input
    def FE_integrate(self, ifunc, points=(0.,3.,100.), big=np.inf, xbig=5e2, limit=250):
        f = ifunc   # grab integrand function (of 'k' only)
        prefac = 1/(4*PI*PI)    # integral prefactor
        if big == np.inf:
            (ires1, ierr1,) = integ.quad(f, 0., big, limit=limit)   # regular 'quadrature' (no 'points' with np.inf)
            ires2, ires3 = 0, 0
        else:
            (ires1, ierr1,) = integ.quad(f, 0., big, points=points, limit=limit)    # regular 'quadrature'
            (ires2, ierr2,) = integ.quad(f, big, xbig, points=(big, (big+xbig)/2, xbig), limit=limit)
            (ires3, ierr3,) = integ.quad(f, xbig, np.inf, limit=limit)
        ires = ires1 + ires2 + ires3  # total is sum of [0,med], [med,big], remainder [big,inf]
        return DTYPE( prefac*ires )

    #   integrator for 'x' and derivatives
    def X_integrate(self, ifunc, points=(0.,1.,100.), big=np.inf, xbig=5e2, limit=250):
        f = ifunc   # grab integrand function (of 'k' only)
        N = self.seq.N
        prefac = (N/(N-1)) * (1/(36*PI*PI))     # integral prefactor
        if big == np.inf:
            (ires1, ierr1,) = integ.quad(f, 0., big, limit=limit)    # regular 'quadrature' (no 'points' with np.inf)
            ires2, ires3 = 0, 0
        else:
            (ires1, ierr1,) = integ.quad(f, 0., big, points=points, limit=limit)    # regular 'quadrature'
            (ires2, ierr2,) = integ.quad(f, big, xbig, points=(big, (big+xbig)/2, xbig), limit=limit)
            (ires3, ierr3,) = integ.quad(f, xbig, np.inf, limit=limit)
        ires = ires1 + ires2 + ires3  # total is sum of [0,med], [med,big], remainder [big,inf]
        return DTYPE( prefac*ires )

    #   electronic part integrand (ions + polymer) : using dimensionless 'k'    - zeroth derivative (same for fG and rG)
    def integrand0(self, k, phi, t, x=1):
        k2lam = self.lam(k,t)       # lambda pre-multiplied by k^2
        k2 = (k*k)     # reduce squaring operations
        v2 = self.v2 * np.exp(-(k2)/6)        # volume interaction is regularized!
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        # correction due to self-energy     [EITHER: 'corr_self' subtracted, OR 'e_corr_self' divided in log]
        corr_self = k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )
#        e_corr_self = k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )
        # shortcuts / call functions only if necessary -> they enter if v2 is nonzero
        G = self.realG(k2, x, self.range0)
        minig = self.realMinig(k2, x, self.range0)
        Z = self.realZ(k2, x, self.range0)
        # big term appearing in log
        paren0 = ( k2 + k2lam*salt_fac + k2*phi*v2*minig +
                k2lam*phi*(G + cion_fac + salt_fac*v2*minig) +
                k2lam*v2*(phi*phi)*(G*minig - (Z*Z) + cion_fac*minig) )
        # main part of integrand
        main = (k2) * np.log( paren0 / k2 )
        return DTYPE(main - corr_self)
#        main = (k2) * np.log( paren0 / e_corr_self)
#        return DTYPE(main)

    #   electronic part integrand (ions + polymer) : using dimensionless 'k'    - first derivative (fixed-G)
    def integrand1_fg(self, k, phi, t, x=1, dx=0):
        # note: args 'x' and 'dx' are needed for most general use in Free Energy 'dF', but do not actually enter here (fG)
        k2lam = self.lam(k,t)       # lambda pre-multiplied by k^2
        k2 = (k*k)     # reduce squaring operations
        v2 = self.v2 * np.exp(-(k2)/6)        # volume interaction is regularized!
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        # correction due to self-energy
        corr_self = k2lam * ( self.seq.totalsig + cion_fac )   # derivative taken
#        e_corr_self = k2 * k2lam * ( (self.seq.totalsig + cion_fac) / \
#                        ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) )
        # shortcuts / call functions only if necessary -> they enter if v2 is nonzero
        G = self.realG(k2, x, self.range0)
        minig = self.realMinig(k2, x, self.range0)
        Z = self.realZ(k2, x, self.range0)
        # big term appearing in log
        paren0 = ( k2 + k2lam*salt_fac + k2*phi*v2*minig +
                k2lam*phi*(G + cion_fac + salt_fac*v2*minig) +
                k2lam*v2*(phi*phi)*(G*minig - (Z*Z) + cion_fac*minig) )
        # derivative of big term
        paren1 = k2*v2*minig + k2lam*(G + cion_fac + salt_fac*v2*minig) + \
                    2*k2lam*v2*phi*(G*minig - (Z*Z) + cion_fac*minig)
        # main part of integrand
        main = (k2) * paren1 / paren0
        return DTYPE(main - corr_self)

    #   electronic part integrand (ions + polymer) : using dimensionless 'k'    - second derivative (fixed-G)
    def integrand2_fg(self, k, phi, t, x=1, dx=0, d2x=0):
        # note: args 'x' and 'dx' are needed for most general use in Free Energy 'dF', but do not actually enter here (fG)
        k2lam = self.lam(k,t)       # lambda pre-multiplied by k^2
        k2 = (k*k)     # reduce squaring operations
        v2 = self.v2 * np.exp(-(k2)/6)        # volume interaction is regularized!
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        # NO correction due to self-energy in higher derivatives
#        e_corr_self = k2 * k2lam * k2lam * ( (self.seq.totalsig + cion_fac)*(self.seq.totalsig + cion_fac) / \
#                        ( ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) * \
#                         ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) ) )
        # shortcuts / call functions only if necessary -> they enter if v2 is nonzero
        G = self.realG(k2, x, self.range0)
        minig = self.realMinig(k2, x, self.range0)
        Z = self.realZ(k2, x, self.range0)
        # big term appearing in log
        paren0 = ( k2 + k2lam*salt_fac + k2*phi*v2*minig +
                k2lam*phi*(G + cion_fac + salt_fac*v2*minig) +
                k2lam*v2*(phi*phi)*(G*minig - (Z*Z) + cion_fac*minig) )
        # derivative of big term
        paren1 = k2*v2*minig + k2lam*(G + cion_fac + salt_fac*v2*minig) + \
                    2*k2lam*v2*phi*(G*minig - (Z*Z) + cion_fac*minig)
        # second derivative of big term
        paren2 = 2*k2lam*v2*(G*minig - (Z*Z) + cion_fac*minig)
        # main part of integrand
        main = (k2) * ( (-(paren1*paren1/paren0) + paren2)/paren0 )
        return DTYPE(main)

    #   electronic part integrand (ions + polymer) : using dimensionless 'k'    - third derivative (fixed-G)
    def integrand3_fg(self, k, phi, t, x=1, dx=0, d2x=0, d3x=0):
        # note: args 'x' and 'dx' are needed for most general use in Free Energy 'dF', but do not actually enter here (fG)
        k2lam = self.lam(k,t)       # lambda pre-multiplied by k^2
        k2 = (k*k)     # reduce squaring operations
        v2 = self.v2 * np.exp(-(k2)/6)        # volume interaction is regularized!
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        # NO correction due to self-energy in higher derivatives
#        e_corr_self = k2 * k2lam * k2lam * k2lam * ( ( (self.seq.totalsig + cion_fac) * \
#                        (self.seq.totalsig + cion_fac) * \
#                        (self.seq.totalsig + cion_fac) ) / \
#                        ( ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) * \
#                        ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) * \
#                        ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) ) )
        # shortcuts / call functions only if necessary -> they enter if v2 is nonzero
        G = self.realG(k2, x, self.range0)
        minig = self.realMinig(k2, x, self.range0)
        Z = self.realZ(k2, x, self.range0)
        # big term appearing in log
        paren0 = ( k2 + k2lam*salt_fac + k2*phi*v2*minig +
                k2lam*phi*(G + cion_fac + salt_fac*v2*minig) +
                k2lam*v2*(phi*phi)*(G*minig - (Z*Z) + cion_fac*minig) )
        # derivative of big term
        paren1 = k2*v2*minig + k2lam*(G + cion_fac + salt_fac*v2*minig) + \
                2*k2lam*v2*phi*(G*minig - (Z*Z) + cion_fac*minig)
        # second derivative of big term
        paren2 = 2*k2lam*v2*(G*minig - (Z*Z) + cion_fac*minig)
        # main part of integrand
        main = (k2) * ( (2*(paren1*paren1*paren1/paren0) - 3*paren2*paren1)/(paren0*paren0) )
        return DTYPE(main)

    #   electronic part integrand (ions + polymer) : using dimensionless 'k'    - first derivative (renorm-G)
    def integrand1_rg(self, k, phi, t, x=1, dx=0):
        # note: args 'x' and 'dx' are needed for most general use in Free Energy 'dF', but do not actually enter here (fG)
        k2lam = self.lam(k,t)       # lambda pre-multiplied by k^2
        k2 = (k*k)     # reduce squaring operations
        kfac1 = -(k2)/6       # overall factor related to 'k' (first power)
        v2 = self.v2 * np.exp(-(k2)/6)        # volume interaction is regularized!
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        args = (phi, v2, cion_fac, salt_fac, k2lam)
        # correction due to self-energy
        corr_self = k2lam * ( self.seq.totalsig + cion_fac )   # derivative taken
#        e_corr_self = k2 * k2lam * ( (self.seq.totalsig + cion_fac) / \
#                        ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) )
        # shortcuts / call functions only if necessary -> they enter if v2 is nonzero
        G0 = self.realG(k2,x,self.range0)
        G1 = self.realG(k2,x,self.range1)
        g0 = self.realMinig(k2,x,self.range0)
        g1 = self.realMinig(k2,x,self.range1)
        Z0 = self.realZ(k2,x,self.range0)
        Z1 = self.realMinig(k2,x,self.range1)
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        A1 = Xfuncs.alpha1(G0,G1, g0,g1, Z0,Z1)
        # big term appearing in log, and its derivatives
        Rshift = k2 + Xfuncs.R(*args, G0, g0, A0)
        dRdp = Xfuncs.dRdp(*args, G0, g0, A0)
        dRdx = Xfuncs.dRdx(kfac1, *args, G1, g1, A1)
        # integrand derivatives
        dIdp = Xfuncs.dIdp(Rshift, dRdp)
        dIdx = Xfuncs.dIdx(Rshift, dRdx)
        # main part of integrand
        main = k2 * (dIdp + (dIdx*dx))
        return DTYPE(main - corr_self)

    #   electronic part integrand (ions + polymer) : using dimensionless 'k'    - second derivative (renorm-G)
    def integrand2_rg(self, k, phi, t, x=1, dx=0, d2x=0):
        # note: args 'x' and 'dx' are needed for most general use in Free Energy 'dF', but do not actually enter here (fG)
        k2lam = self.lam(k,t)       # lambda pre-multiplied by k^2
        k2 = (k*k)     # reduce squaring operations
        kfac1 = -(k2)/6     # overall factor related to 'k' (first power)
        kfac2 = kfac1*kfac1    #  (second power)
        v2 = self.v2 * np.exp(-(k2)/6)        # volume interaction is regularized!
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        args = (phi, v2, cion_fac, salt_fac, k2lam)
        # NO correction due to self-energy in higher derivatives
#        e_corr_self = k2 * k2lam * k2lam * ( (self.seq.totalsig + cion_fac)*(self.seq.totalsig + cion_fac) / \
#                        ( ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) * \
#                         ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) ) )
        # shortcuts / call functions only if necessary -> they enter if v2 is nonzero
        G0 = self.realG(k2,x,self.range0)
        G1 = self.realG(k2,x,self.range1)
        G2 = self.realG(k2,x,self.range2)
        g0 = self.realMinig(k2,x,self.range0)
        g1 = self.realMinig(k2,x,self.range1)
        g2 = self.realMinig(k2,x,self.range2)
        Z0 = self.realZ(k2,x,self.range0)
        Z1 = self.realMinig(k2,x,self.range1)
        Z2 = self.realMinig(k2,x,self.range2)
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        A1 = Xfuncs.alpha1(G0,G1, g0,g1, Z0,Z1)
        A2 = Xfuncs.alpha2(G0,G1,G2, g0,g1,g2, Z0,Z1,Z2)
        # big term appearing in log, and its derivatives
        Rshift = k2 + Xfuncs.R(*args, G0, g0, A0)
        dRdp = Xfuncs.dRdp(*args, G0, g0, A0)
        dRdx = Xfuncs.dRdx(kfac1, *args, G1, g1, A1)
        d2Rdp2 = Xfuncs.d2Rdp2(*args, g0, A0)
        d2Rdxdp = Xfuncs.d2Rdxdp(kfac1, *args, G1, g1, A1)
        d2Rdx2 = Xfuncs.d2Rdx2(kfac2, *args, G2, g2, A2)
        # integrand derivatives
        dIdx = Xfuncs.dIdx(Rshift, dRdx)
        d2Idp2 = Xfuncs.d2Idp2(Rshift, dRdp, d2Rdp2)
        d2Idxdp = Xfuncs.d2Idxdp(Rshift, dRdp, dRdx, d2Rdxdp)
        d2Idx2 = Xfuncs.d2Idx2(Rshift, dRdx, d2Rdx2)
        # main part of integrand
        main = k2 * (d2Idp2 + (2*d2Idxdp + d2Idx2*dx)*dx + dIdx*d2x)
        return DTYPE(main)

    #   electronic part integrand (ions + polymer) : using dimensionless 'k'    - third derivative (renorm-G)
    def integrand3_rg(self, k, phi, t, x=1, dx=0, d2x=0, d3x=0):
        # note: args 'x' and 'dx' are needed for most general use in Free Energy 'dF', but do not actually enter here (fG)
        k2lam = self.lam(k,t)       # lambda pre-multiplied by k^2
        k2 = (k*k)     # reduce squaring operations
        kfac1 = -(k2)/6     # overall factor related to 'k' (first power)
        kfac2 = kfac1*kfac1    #  (second power)
        kfac3 = kfac2*kfac1    #  (third power)
        v2 = self.v2 * np.exp(-(k2)/6)        # volume interaction is regularized!
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        args = (phi, v2, cion_fac, salt_fac, k2lam)
        # NO correction due to self-energy in higher derivatives
#        e_corr_self = k2 * k2lam * k2lam * k2lam * ( ( (self.seq.totalsig + cion_fac) * \
#                        (self.seq.totalsig + cion_fac) * \
#                        (self.seq.totalsig + cion_fac) ) / \
#                        ( ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) * \
#                        ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) * \
#                        ( k2 + k2lam * ( salt_fac + phi*(self.seq.totalsig + cion_fac) )) ) )
        # shortcuts / call functions only if necessary -> they enter if v2 is nonzero
        G0 = self.realG(k2,x,self.range0)
        G1 = self.realG(k2,x,self.range1)
        G2 = self.realG(k2,x,self.range2)
        G3 = self.realG(k2,x,self.range3)
        g0 = self.realMinig(k2,x,self.range0)
        g1 = self.realMinig(k2,x,self.range1)
        g2 = self.realMinig(k2,x,self.range2)
        g3 = self.realMinig(k2,x,self.range3)
        Z0 = self.realZ(k2,x,self.range0)
        Z1 = self.realMinig(k2,x,self.range1)
        Z2 = self.realMinig(k2,x,self.range2)
        Z3 = self.realMinig(k2,x,self.range3)
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        A1 = Xfuncs.alpha1(G0,G1, g0,g1, Z0,Z1)
        A2 = Xfuncs.alpha2(G0,G1,G2, g0,g1,g2, Z0,Z1,Z2)
        A3 = Xfuncs.alpha3(G0,G1,G2,G3, g0,g1,g2,g3, Z0,Z1,Z2,Z3)
        # big term appearing in log, and its derivatives
        Rshift = k2 + Xfuncs.R(*args, G0, g0, A0)
        dRdp = Xfuncs.dRdp(*args, G0, g0, A0)
        dRdx = Xfuncs.dRdx(kfac1, *args, G1, g1, A1)
        d2Rdp2 = Xfuncs.d2Rdp2(*args, g0, A0)
        d2Rdxdp = Xfuncs.d2Rdxdp(kfac1, *args, G1, g1, A1)
        d2Rdx2 = Xfuncs.d2Rdx2(kfac2, *args, G2, g2, A2)
        d3Rdp3 = Xfuncs.d3Rdp3(*args)
        d3Rdxdp2 = Xfuncs.d3Rdxdp2(kfac1, *args, g1, A1)
        d3Rdx2dp = Xfuncs.d3Rdx2dp(kfac2, *args, G2, g2, A2)
        d3Rdx3 = Xfuncs.d3Rdx3(kfac3, *args, G3, g3, A3)
        # integrand derivatives
        dIdx = Xfuncs.dIdx(Rshift, dRdx)
        d2Idxdp = Xfuncs.d2Idxdp(Rshift, dRdp, dRdx, d2Rdxdp)
        d2Idx2 = Xfuncs.d2Idx2(Rshift, dRdx, d2Rdx2)
        d3Idp3 = Xfuncs.d3Idp3(Rshift, dRdp, d2Rdp2, d3Rdp3)
        d3Idxdp2 = Xfuncs.d3Idxdp2(Rshift, dRdp, dRdx, d2Rdp2, d2Rdxdp, d3Rdxdp2)
        d3Idx2dp = Xfuncs.d3Idx2dp(Rshift, dRdp, dRdx, d2Rdx2, d2Rdxdp, d3Rdx2dp)
        d3Idx3 = Xfuncs.d3Idx3(Rshift, dRdx, d2Rdx2, d3Rdx3)
        # main part of integrand
        main = k2 * (d3Idp3 + (3*d3Idxdp2 + (3*d3Idx2dp + d3Idx3*dx)*dx)*dx + 3*(d2Idxdp + d2Idx2*dx)*d2x + dIdx*d3x)
        return DTYPE(main)

    #   JUST POLYMER part of integrand : using dimensionless 'k'    - zeroth derivative (allowing for renorm-G)
    def integrand0_Ponly(self, k, phi, t, x=1):
        # note: args 'x' and 'dx' are needed for most general use in Free Energy 'dF', but do not actually enter here (fG)
        k2lam = self.lam(k,t)       # lambda pre-multiplied by k^2
        k2 = (k*k)     # reduce squaring operations
        v2 = self.v2 * np.exp(-(k2)/6)        # volume interaction is regularized!
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        nu = k2/k2lam + salt_fac + cion_fac*phi # effective inverse electronic interaction (i.e. screened coulomb)
        args = (phi, v2, cion_fac, nu)
        # correction due to self-energy     [EITHER: 'corr_self' subtracted, OR 'e_corr_self' divided in log]
        corr_self = k2lam * self.seq.totalsig * phi
#        e_corr_self = 1 + phi * self.seq.totalsig / nu
        # shortcuts / call functions only if necessary -> they enter if v2 is nonzero
        G0 = self.realG(k2,x,self.range0)
        g0 = self.realMinig(k2,x,self.range0)
        Z0 = self.realZ(k2,x,self.range0)
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        # big term appearing in log, and its derivatives
        Rshift = 1 + Xfuncs.R_Ponly(*args, G0, g0, A0)
        # main part of integrand
        main = k2 * np.log(Rshift)
        return DTYPE(main - corr_self)
#        main = k2 * np.log(Rshift / e_corr_self)
#        return DTYPE(main)

    #   JUST POLYMER part of integrand : using dimensionless 'k'    - first derivative (allowing for renorm-G)
    def integrand1_Ponly(self, k, phi, t, x=1, dx=0):
        # note: args 'x' and 'dx' are needed for most general use in Free Energy 'dF', but do not actually enter here (fG)
        k2lam = self.lam(k,t)       # lambda pre-multiplied by k^2
        k2 = (k*k)     # reduce squaring operations
        kfac1 = -(k2)/6       # overall factor related to 'k' (first power)
        v2 = self.v2 * np.exp(-(k2)/6)        # volume interaction is regularized!
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        nu = k2/k2lam + salt_fac + cion_fac*phi # effective inverse electronic interaction (i.e. screened coulomb)
        args = (phi, v2, cion_fac, nu)
        # correction due to self-energy
        corr_self = k2lam * self.seq.totalsig       # derivative taken
#        e_corr_self = k2lam * self.seq.totalsig / (1 + phi * self.seq.totalsig / nu)
        # shortcuts / call functions only if necessary -> they enter if v2 is nonzero
        G0 = self.realG(k2,x,self.range0)
        G1 = self.realG(k2,x,self.range1) if dx else 0
        g0 = self.realMinig(k2,x,self.range0)
        g1 = self.realMinig(k2,x,self.range1) if dx else 0
        Z0 = self.realZ(k2,x,self.range0)
        Z1 = self.realZ(k2,x,self.range1) if dx else 0
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        A1 = Xfuncs.alpha1(G0,G1, g0,g1, Z0,Z1)
        # big term appearing in log, and its derivatives
        Rshift = 1 + Xfuncs.R_Ponly(*args, G0, g0, A0)
        dRdp = Xfuncs.dRdp_Ponly(*args, G0, g0, A0)
        dRdx = Xfuncs.dRdx_Ponly(kfac1, *args, G1, g1, A1)
        # integrand derivatives
        dIdp = Xfuncs.dIdp(Rshift, dRdp)
        dIdx = Xfuncs.dIdx(Rshift, dRdx)
        # main part of integrand
        main = k2 * (dIdp + (dIdx*dx))
        return DTYPE(main - corr_self)

    #   JUST POLYMER part of integrand : using dimensionless 'k'    - second derivative (allowing for renorm-G)
    def integrand2_Ponly(self, k, phi, t, x=1, dx=0, d2x=0):
        # note: args 'x' and 'dx' are needed for most general use in Free Energy 'dF', but do not actually enter here (fG)
        k2lam = self.lam(k,t)       # lambda pre-multiplied by k^2
        k2 = (k*k)     # reduce squaring operations
        kfac1 = -(k2)/6     # overall factor related to 'k' (first power)
        kfac2 = kfac1*kfac1    #  (second power)
        v2 = self.v2 * np.exp(-(k2)/6)        # volume interaction is regularized!
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        nu = k2/k2lam + salt_fac + cion_fac*phi # effective inverse electronic interaction (i.e. screened coulomb)
        args = (phi, v2, cion_fac, nu)
        # NO correction due to self-energy in higher derivatives
#        e_corr_self = (k2lam / nu) * ( self.seq.totalsig * self.seq.totalsig / \
#                        ( (1 + phi * self.seq.totalsig / nu) * (1 + phi * self.seq.totalsig / nu) ) )
        # shortcuts / call functions only if necessary -> they enter if v2 is nonzero
        G0 = self.realG(k2,x,self.range0)
        G1 = self.realG(k2,x,self.range1) if dx else 0
        G2 = self.realG(k2,x,self.range2) if d2x else 0
        g0 = self.realMinig(k2,x,self.range0)
        g1 = self.realMinig(k2,x,self.range1) if dx else 0
        g2 = self.realMinig(k2,x,self.range2) if d2x else 0
        Z0 = self.realZ(k2,x,self.range0)
        Z1 = self.realZ(k2,x,self.range1) if dx else 0
        Z2 = self.realZ(k2,x,self.range2) if d2x else 0
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        A1 = Xfuncs.alpha1(G0,G1, g0,g1, Z0,Z1)
        A2 = Xfuncs.alpha2(G0,G1,G2, g0,g1,g2, Z0,Z1,Z2)
        # big term appearing in log, and its derivatives
        Rshift = 1 + Xfuncs.R_Ponly(*args, G0, g0, A0)
        dRdp = Xfuncs.dRdp_Ponly(*args, G0, g0, A0)
        dRdx = Xfuncs.dRdx_Ponly(kfac1, *args, G1, g1, A1)
        d2Rdp2 = Xfuncs.d2Rdp2_Ponly(*args, G0, g0, A0)
        d2Rdxdp = Xfuncs.d2Rdxdp_Ponly(kfac1, *args, G1, g1, A1)
        d2Rdx2 = Xfuncs.d2Rdx2_Ponly(kfac2, *args, G2, g2, A2)
        # integrand derivatives
        dIdx = Xfuncs.dIdx(Rshift, dRdx)
        d2Idp2 = Xfuncs.d2Idp2(Rshift, dRdp, d2Rdp2)
        d2Idxdp = Xfuncs.d2Idxdp(Rshift, dRdp, dRdx, d2Rdxdp)
        d2Idx2 = Xfuncs.d2Idx2(Rshift, dRdx, d2Rdx2)
        # main part of integrand
        main = k2 * (d2Idp2 + (2*d2Idxdp + d2Idx2*dx)*dx + dIdx*d2x)
        return DTYPE(main)

    #   JUST POLYMER part of integrand : using dimensionless 'k'    - third derivative (allowing for renorm-G)
    def integrand3_Ponly(self, k, phi, t, x=1, dx=0, d2x=0, d3x=0):
        # note: args 'x' and 'dx' are needed for most general use in Free Energy 'dF', but do not actually enter here (fG)
        k2lam = self.lam(k,t)       # lambda pre-multiplied by k^2
        k2 = (k*k)     # reduce squaring operations
        kfac1 = -(k2)/6     # overall factor related to 'k' (first power)
        kfac2 = kfac1*kfac1    #  (second power)
        kfac3 = kfac2*kfac1    #  (third power)
        v2 = self.v2 * np.exp(-(k2)/6)        # volume interaction is regularized!
        salt_fac = self.sfac     # overall factor for salt ions (+&-) : valence charge * _total_ concentration
        cion_fac = self.cfac     # overall factor for counter-ions : valence charge * scale parameter
        nu = k2/k2lam + salt_fac + cion_fac*phi # effective inverse electronic interaction (i.e. screened coulomb)
        args = (phi, v2, cion_fac, nu)
        # NO correction due to self-energy in higher derivatives
#        e_corr_self = (k2lam / nu / nu) * ( self.seq.totalsig * self.seq.totalsig * self.seq.totalsig / \
#                        ( (1 + phi * self.seq.totalsig / nu) * (1 + phi * self.seq.totalsig / nu) * \
#                        (1 + phi * self.seq.totalsig / nu) ) )
        # shortcuts / call functions only if necessary -> they enter if v2 is nonzero
        G0 = self.realG(k2,x,self.range0)
        G1 = self.realG(k2,x,self.range1) if dx else 0
        G2 = self.realG(k2,x,self.range2) if d2x else 0
        G3 = self.realG(k2,x,self.range3) if d3x else 0
        g0 = self.realMinig(k2,x,self.range0)
        g1 = self.realMinig(k2,x,self.range1) if dx else 0
        g2 = self.realMinig(k2,x,self.range2) if d2x else 0
        g3 = self.realMinig(k2,x,self.range3) if d3x else 0
        Z0 = self.realZ(k2,x,self.range0)
        Z1 = self.realZ(k2,x,self.range1) if dx else 0
        Z2 = self.realZ(k2,x,self.range2) if d2x else 0
        Z3 = self.realZ(k2,x,self.range3) if d3x else 0
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        A1 = Xfuncs.alpha1(G0,G1, g0,g1, Z0,Z1)
        A2 = Xfuncs.alpha2(G0,G1,G2, g0,g1,g2, Z0,Z1,Z2)
        A3 = Xfuncs.alpha3(G0,G1,G2,G3, g0,g1,g2,g3, Z0,Z1,Z2,Z3)
        # big term appearing in log, and its derivatives
        Rshift = 1 + Xfuncs.R_Ponly(*args, G0, g0, A0)
        dRdp = Xfuncs.dRdp_Ponly(*args, G0, g0, A0)
        dRdx = Xfuncs.dRdx_Ponly(kfac1, *args, G1, g1, A1)
        d2Rdp2 = Xfuncs.d2Rdp2_Ponly(*args, G0, g0, A0)
        d2Rdxdp = Xfuncs.d2Rdxdp_Ponly(kfac1, *args, G1, g1, A1)
        d2Rdx2 = Xfuncs.d2Rdx2_Ponly(kfac2, *args, G2, g2, A2)
        d3Rdp3 = Xfuncs.d3Rdp3_Ponly(*args, G0, g0, A0)
        d3Rdxdp2 = Xfuncs.d3Rdxdp2_Ponly(kfac1, *args, G1, g1, A1)
        d3Rdx2dp = Xfuncs.d3Rdx2dp_Ponly(kfac2, *args, G2, g2, A2)
        d3Rdx3 = Xfuncs.d3Rdx3_Ponly(kfac3, *args, G3, g3, A3)
        # integrand derivatives
        dIdx = Xfuncs.dIdx(Rshift, dRdx)
        d2Idxdp = Xfuncs.d2Idxdp(Rshift, dRdp, dRdx, d2Rdxdp)
        d2Idx2 = Xfuncs.d2Idx2(Rshift, dRdx, d2Rdx2)
        d3Idp3 = Xfuncs.d3Idp3(Rshift, dRdp, d2Rdp2, d3Rdp3)
        d3Idxdp2 = Xfuncs.d3Idxdp2(Rshift, dRdp, dRdx, d2Rdp2, d2Rdxdp, d3Rdxdp2)
        d3Idx2dp = Xfuncs.d3Idx2dp(Rshift, dRdp, dRdx, d2Rdx2, d2Rdxdp, d3Rdx2dp)
        d3Idx3 = Xfuncs.d3Idx3(Rshift, dRdx, d2Rdx2, d3Rdx3)
        # main part of integrand
        main = k2 * (d3Idp3 + (3*d3Idxdp2 + (3*d3Idx2dp + d3Idx3*dx)*dx)*dx + 3*(d2Idxdp + d2Idx2*dx)*d2x + dIdx*d3x)
        return DTYPE(main)


    #   integrand appearing in 'x' solution
    def J(self, k, phi, t, x, c_fac, s_fac):
        k2 = (k*k)       # reduce these operations
        ilam = (k2)/self.lam(k,t)         # inverse of real lambda (a.k.a. 'Lk')
#        v2 = self.v2 * np.exp(kfac1)        # volume interaction is regularized! <nope>
        v2 = self.v2        # NO regularization for 'x' integrals!
        # get correlations / moments only if needed, and just once for all terms!   <for derivatives: put this in integrands!>
        G0 = self.realG(k2,x,self.range0)
        G2 = self.realG(k2,x,self.range2)
        g0 = self.realMinig(k2,x,self.range0)
        g2 = self.realMinig(k2,x,self.range2)
        Z0 = self.realZ(k2,x,self.range0)
        Z2 = self.realZ(k2,x,self.range2)
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        B0 = Xfuncs.beta0(G0,G2, g0,g2, Z0,Z2)
        # big generic list of arguments for correlations functions & derivatives (w/r/t 'x')
        args = (phi,v2,c_fac,s_fac,ilam)
        # arrange 'E' and 'D' appropriately
        return ( (k2*k2) * Xfuncs.E(*args,G2,g2,B0) / Xfuncs.D(*args,G0,g0,A0) )

    #   first derivatives are integrands directly (appearing in numerator & denominator of 'x' derivative solutions)
    def dJdp(self, k, phi, t, x, c_fac, s_fac):
        k2 = (k*k)       # reduce these operations
        ilam = (k2)/self.lam(k,t)         # inverse of real lambda (a.k.a. 'Lk')
#        v2 = self.v2 * np.exp(kfac1)        # volume interaction is regularized! <nope>
        v2 = self.v2        # NO regularization for 'x' integrals!
        # get correlations / moments only if needed, and just once for all terms!   <for derivatives: put this in integrands!>
        G0 = self.realG(k2,x,self.range0)
        G2 = self.realG(k2,x,self.range2)
        g0 = self.realMinig(k2,x,self.range0)
        g2 = self.realMinig(k2,x,self.range2)
        Z0 = self.realZ(k2,x,self.range0)
        Z2 = self.realZ(k2,x,self.range2)
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        B0 = Xfuncs.beta0(G0,G2, g0,g2, Z0,Z2)
        # big generic list of arguments for correlations functions & derivatives (w/r/t 'x')
        args = (phi,v2,c_fac,s_fac,ilam)
        # arrange 'E' and 'D' appropriately
        D = Xfuncs.D(*args,G0,g0,A0)
        return ( (k2*k2) * (Xfuncs.dEdp(*args,g2,B0) - Xfuncs.E(*args,G2,g2,B0)*Xfuncs.dDdp(*args,G0,g0,A0)/D )/D )

    def dJdx(self, k, phi, t, x, c_fac, s_fac):
        k2 = (k*k)       # reduce these operations
        kfac1 = -(k2)/6       # overall factor related to 'k' (first power)
        ilam = k2/self.lam(k,t)       # inverse of real lambda (a.k.a. 'Lk')
#        v2 = self.v2 * np.exp(kfac1)        # volume interaction is regularized! <nope>
        v2 = self.v2        # NO regularization for 'x' integrals!
        # get correlations / moments only if needed, and just once for all terms!   <for derivatives: put this in integrands!>
        G0 = self.realG(k2,x,self.range0)
        G1 = self.realG(k2,x,self.range1)
        G2 = self.realG(k2,x,self.range2)
        G3 = self.realG(k2,x,self.range3)
        g0 = self.realMinig(k2,x,self.range0)
        g1 = self.realMinig(k2,x,self.range1)
        g2 = self.realMinig(k2,x,self.range2)
        g3 = self.realMinig(k2,x,self.range3)
        Z0 = self.realZ(k2,x,self.range0)
        Z1 = self.realZ(k2,x,self.range1)
        Z2 = self.realZ(k2,x,self.range2)
        Z3 = self.realZ(k2,x,self.range3)
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        A1 = Xfuncs.alpha1(G0,G1, g0,g1, Z0,Z1)
        B0 = Xfuncs.beta0(G0,G2, g0,g2, Z0,Z2)
        B1 = Xfuncs.beta1(G0,G1,G2,G3, g0,g1,g2,g3, Z0,Z1,Z2,Z3)
        # big generic list of arguments for correlations functions & derivatives (w/r/t 'x')
        args = (phi,v2,c_fac,s_fac,ilam)
        # arrange 'E' and 'D' appropriately
        D = Xfuncs.D(*args,G0,g0,A0)
        return ( (k2*k2) * (Xfuncs.dEdx(kfac1,*args,G3,g3,B1) - Xfuncs.E(*args,G2,g2,B0)*Xfuncs.dDdx(kfac1,*args,G1,g1,A1)/D )/D )

    #   integrands for numerators of 2nd & 3rd order 'x' derivative solutions
    def XintN2(self, k, phi, t, x, dx, c_fac, s_fac):
        k2 = (k*k)       # reduce these operations
        kfac1 = -(k2)/6       # overall factor related to 'k' (first power)
        kfac2 = (kfac1*kfac1)    #   (second power)
        ilam = k2/self.lam(k,t)       # inverse of real lambda (a.k.a. 'Lk')
#        v2 = self.v2 * np.exp(kfac1)        # volume interaction is regularized! <nope>
        v2 = self.v2        # NO regularization for 'x' integrals!
        # get correlations / moments only if needed, and just once for all terms!   <for derivatives: put this in integrands!>
        G0 = self.realG(k2,x,self.range0)
        G1 = self.realG(k2,x,self.range1)
        G2 = self.realG(k2,x,self.range2)
        G3 = self.realG(k2,x,self.range3)
        G4 = self.realG(k2,x,self.range4)
        g0 = self.realMinig(k2,x,self.range0)
        g1 = self.realMinig(k2,x,self.range1)
        g2 = self.realMinig(k2,x,self.range2)
        g3 = self.realMinig(k2,x,self.range3)
        g4 = self.realMinig(k2,x,self.range4)
        Z0 = self.realZ(k2,x,self.range0)
        Z1 = self.realZ(k2,x,self.range1)
        Z2 = self.realZ(k2,x,self.range2)
        Z3 = self.realZ(k2,x,self.range3)
        Z4 = self.realZ(k2,x,self.range4)
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        A1 = Xfuncs.alpha1(G0,G1, g0,g1, Z0,Z1)
        A2 = Xfuncs.alpha2(G0,G1,G2, g0,g1,g2, Z0,Z1,Z2)
        B0 = Xfuncs.beta0(G0,G2, g0,g2, Z0,Z2)
        B1 = Xfuncs.beta1(G0,G1,G2,G3, g0,g1,g2,g3, Z0,Z1,Z2,Z3)
        B2 = Xfuncs.beta2(G0,G1,G2,G3,G4, g0,g1,g2,g3,g4, Z0,Z1,Z2,Z3,Z4)
        # big generic list of arguments for correlations functions & derivatives (w/r/t 'x')
        args = (phi,v2,c_fac,s_fac,ilam)
        # get 'E' and 'D', arrange 'J' derivatives
        E = Xfuncs.E(*args, G2, g2, B0)
        D = Xfuncs.D(*args, G0, g0, A0)
        dEdp = Xfuncs.dEdp(*args, g2, B0)
        dEdx = Xfuncs.dEdx(kfac1, *args, G3, g3, B1)
        dDdp = Xfuncs.dDdp(*args, G0, g0, A0)
        dDdx = Xfuncs.dDdx(kfac1, *args, G1, g1, A1)
        d2Edx2 = Xfuncs.d2Edx2(kfac2, *args, G4, g4, B2)
        d2Edxdp = Xfuncs.d2Edxdp(kfac1, *args, g3, B1)
        d2Ddx2 = Xfuncs.d2Ddx2(kfac2, *args, G2, g2, A2)
        d2Ddxdp = Xfuncs.d2Ddxdp(kfac1, *args, G1, g1, A1)
        d2Ddp2 = Xfuncs.d2Ddp2(*args, g0, A0)
        termp2 = Xfuncs.d2Jdp2(E, D, dEdp, dDdp, d2Ddp2)
        termxp = 2 * Xfuncs.d2Jdxdp(E, D, dEdp, dEdx, dDdp, dDdx, d2Edxdp, d2Ddxdp)
        termx2 = Xfuncs.d2Jdx2(E, D, dEdx, dDdx, d2Edx2, d2Ddx2)
        return ( (k2*k2) * (termp2 + (termxp + termx2*dx)*dx) )

    def XintN3(self, k, phi, t, x, dx, d2x, c_fac, s_fac):
        k2 = (k*k)       # reduce these operations
        kfac1 = -(k2)/6       # overall factor related to 'k' (first power)
        kfac2 = kfac1*kfac1    #(k**4)/36       #   (second power)
        kfac3 = kfac2*kfac1    #-(k**6)/216     #   (third power)
        ilam = k2/self.lam(k,t)       # inverse of real lambda (a.k.a. 'Lk')
#        v2 = self.v2 * np.exp(kfac1)        # volume interaction is regularized! <nope>
        v2 = self.v2        # NO regularization for 'x' integrals!
        # get correlations / moments only if needed, and just once for all terms!   <for derivatives: put this in integrands!>
        G0 = self.realG(k2,x,self.range0)
        G1 = self.realG(k2,x,self.range1)
        G2 = self.realG(k2,x,self.range2)
        G3 = self.realG(k2,x,self.range3)
        G4 = self.realG(k2,x,self.range4)
        G5 = self.realG(k2,x,self.range5)
        g0 = self.realMinig(k2,x,self.range0)
        g1 = self.realMinig(k2,x,self.range1)
        g2 = self.realMinig(k2,x,self.range2)
        g3 = self.realMinig(k2,x,self.range3)
        g4 = self.realMinig(k2,x,self.range4)
        g5 = self.realMinig(k2,x,self.range5)
        Z0 = self.realZ(k2,x,self.range0)
        Z1 = self.realZ(k2,x,self.range1)
        Z2 = self.realZ(k2,x,self.range2)
        Z3 = self.realZ(k2,x,self.range3)
        Z4 = self.realZ(k2,x,self.range4)
        Z5 = self.realZ(k2,x,self.range5)
        A0 = Xfuncs.alpha0(G0, g0, Z0)
        A1 = Xfuncs.alpha1(G0,G1, g0,g1, Z0,Z1)
        A2 = Xfuncs.alpha2(G0,G1,G2, g0,g1,g2, Z0,Z1,Z2)
        A3 = Xfuncs.alpha3(G0,G1,G2,G3, g0,g1,g2,g3, Z0,Z1,Z2,Z3)
        B0 = Xfuncs.beta0(G0,G2, g0,g2, Z0,Z2)
        B1 = Xfuncs.beta1(G0,G1,G2,G3, g0,g1,g2,g3, Z0,Z1,Z2,Z3)
        B2 = Xfuncs.beta2(G0,G1,G2,G3,G4, g0,g1,g2,g3,g4, Z0,Z1,Z2,Z3,Z4)
        B3 = Xfuncs.beta3(G0,G1,G2,G3,G4,G5, g0,g1,g2,g3,g4,g5, Z0,Z1,Z2,Z3,Z4,Z5)
        # big generic list of arguments for correlations functions & derivatives (w/r/t 'x')
        args = (phi,v2,c_fac,s_fac,ilam)
        # get 'E' and 'D', arrange 'J' derivatives
        E = Xfuncs.E(*args, G2, g2, B0)
        D = Xfuncs.D(*args, G0, g0, A0)
        dEdp = Xfuncs.dEdp(*args, g2, B0)
        dEdx = Xfuncs.dEdx(kfac1, *args, G3, g3, B1)
        dDdp = Xfuncs.dDdp(*args, G0, g0, A0)
        dDdx = Xfuncs.dDdx(kfac1, *args, G1, g1, A1)
        d2Edx2 = Xfuncs.d2Edx2(kfac2, *args, G4, g4, B2)
        d2Edxdp = Xfuncs.d2Edxdp(kfac1, *args, g3, B1)
        d2Ddx2 = Xfuncs.d2Ddx2(kfac2, *args, G2, g2, A2)
        d2Ddxdp = Xfuncs.d2Ddxdp(kfac1, *args, G1, g1, A1)
        d2Ddp2 = Xfuncs.d2Ddp2(*args, g0, A0)
        d3Edx3 = Xfuncs.d3Edx3(kfac3, *args, G5, g5, B3)
        d3Edx2dp = Xfuncs.d3Edx2dp(kfac2, *args, g4, B2)
        d3Ddx3 = Xfuncs.d3Ddx3(kfac3, *args, G3, g3, A3)
        d3Ddx2dp = Xfuncs.d3Ddx2dp(kfac2, *args, G2, g2, A2)
        d3Ddxdp2 = Xfuncs.d3Ddxdp2(kfac1, *args, g1, A1)
        termxp = 3 * Xfuncs.d2Jdxdp(E, D, dEdp, dEdx, dDdp, dDdx, d2Edxdp, d2Ddxdp)
        termx2 = 3 * Xfuncs.d2Jdx2(E, D, dEdx, dDdx, d2Edx2, d2Ddx2)
        termp3 = Xfuncs.d3Jdp3(E, D, dEdp, dDdp, d2Ddp2)
        termxp2 = 3 * Xfuncs.d3Jdxdp2(E, D, dEdp, dEdx, dDdp, dDdx, d2Edxdp, d2Ddp2, d2Ddxdp, d3Ddxdp2)
        termx2p = 3 * Xfuncs.d3Jdx2dp(E, D, dEdp, dEdx, dDdp, dDdx, d2Edxdp, d2Edx2, d2Ddxdp, d2Ddx2, d3Edx2dp, d3Ddx2dp)
        termx3 = Xfuncs.d3Jdx3(E, D, dEdx, dDdx, d2Edx2, d2Ddx2, d3Edx3, d3Ddx3)
        return ( (k2*k2) * (termp3 + termxp*d2x + (termxp2 + termx2*d2x + (termx2p + termx3*dx)*dx)*dx) )
#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####
