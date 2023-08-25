##  Mike Phillips, 6/10/2022
##  * critical point & spinodal / binodal solver *
##  Simple structure for handling generic approach to LLPS
##  - takes some sort of Model object, i.e. 'RPA'
##  - Sequence and all details are inherited from the Model
##  - Solver holds at least one method (i.e. 'Maxwell' or 'partition')
##  Note: This solver is set up to handle LLPS calculations in a generic way,
##      but really expects some sort of rG-RPA framework (e.g. with 'x').
##      Particularly: see 'Ftot', 'extract' and 'extractMax'.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator  #MultipleLocator
import scipy.optimize as opt
from time import perf_counter

rand = np.random.rand

import myPlotOptions    # global options

##  Global Data Type
DTYPE = np.float64

eps = 1e-12         # small number to avoid issues (i.e. 'phi' near 0 or 1)

##  Options for single-/multi- variable methods
option1 = ("1d","1dim","single")
option2 = ("2d","2dim","multi")

#####   Class Defnition : object for finding critical point and/or binodal/spinodal coexistence curves.
class Coex:
    def __init__(self, model, methods=None, spars={}):
        self.model = model
        self.methods = {}
        if methods:
            self.add_methods(methods)
        else:
            self.add_methods({"maxwell":"1d"})
        self.pars = {"t_min":0.1, "t_points":20, "iterMax":50, "thr":1e-6}
        if spars:
            self.pars.update(spars)
        # store true maximum cutoff (less than 1, from any nonzero ionic densities)
        self.realMax = (1-self.model.salt)/(1+self.model.cionpar)
    #   #   #   #   #   #   #

    #   accessor -- critical point as ordered pair (phi_crit, Teff_crit)
    def get_crit(self):
        return (self.pcrit, self.tcrit)
    #   accessor -- effective temperature list (l/lB)
    def get_Teff(self):
        return self.ty
    #   accessor -- voume fraction [phi] list, for spinodal
    def get_spino(self):
        return self.spinox
    #   accessor -- voume fraction [phi] list, for binodal
    def get_bino(self):
        # first method taken as default
        method = list(self.methods.keys())[0]
        return self.results[method]

    #   copy -- make identical copy for future reference (independent of original)
    def copy(self):
        # create new object (not necessary to print info)
        ob = Coex(self.model, self.methods)
        # save items if available
        try:
            (ob.pcrit, ob.tcrit) = (self.pcrit, self.tcrit)
        except AttributeError:
            0
        try:
            ob.ty = self.ty
        except AttributeError:
            0
        try:
            ob.spinox = self.spinox
        except AttributeError:
            0
        try:
            ob.results = self.results
        except AttributeError:
            0
        return ob

    #   allow adding methods during or after creation of object
    def add_methods(self, methods={}):
        # possible methods:
        # "max#" with option "1d" or "2d"                    (first letter = "m")
        # "par#" with option ["1d", 0] or ["2d",0.2] etc.    (first letter = "p")
        for m in methods:
            if m not in self.methods:
                self.methods.update({m:methods[m]})
        return

    ## check for various forms of free energy method (offset and/or dimension) : return true values
    def pCheck(self,m,v):
        if type(v) == str:      # default to zero offset                       
            return (v,0)
        elif type(v) in (int, float):   # default to 1D iteration
            return ("1d",v)
        elif type(v) == list:
            if type(v[0]) == str:
                (dim, off) = v
                return (dim, off)
            elif type(v[0]) in (int, float):
                (off, dim) = v
                return (dim, off)
            else:
                print( ("\n\nWARNING: improper format given for options (method %s), '" % m) + str(v) + "'.")
                return
        else:
            print( ("\n\nWARNING: improper format given for options (method %s), '" % m) + str(v) + "'.")
            return
    ## check for valid option in maxwell construction
    def mCheck(self,m,v):
        if type(v) == str:
            return v
        else:
            print( ("\n\nWARNING: improper format given for options (method %s), '" % m) + str(v) + "'.")
            return

    ##  CRIT. PT. FINDER
    # --> using inverse temperature u=1/t as variable
    #   solve for 'u' first from d2F=0 (arbitrary 'phi')        <1D root finder: brenth>
    #   then find 'phi' by _minimizing 'u' across range of 'phi'   <1D minimizer: brent>
    def find_crit(self, min_frac=0.1, FintKW={}, XintKW={}, phi_bracket=(1e-8,5e-2,4e-1), u_bracket=(1e-4,1e3), \
                    u_bracket_notes=False):
        Na = self.model.Na
        N = self.model.seq.N
        thr = self.pars["thr"]
        # timer
        print("\nFinding critical point...")
        t1 = perf_counter()
        # Flory-Huggins critical point for some scale
        pc = Na**(0.5) / ( Na**(0.5) + N**(0.5) )
        tc = 2*Na*N / ( (Na**(0.5) + N**(0.5))**2 )
        uc = 1/tc
        # initial values
        pi = pc
        ui = uc
        pmin = self.model.small   # minimum 'phi'
        pmax = self.realMax - self.model.small    # maximum 'phi'
        (umin, umax) = u_bracket
        if u_bracket_notes:
            u_print = lambda p, d2f: print("\nD2F BRACKET TEST" + \
                        "(at phi={:5g}):\n\t(umin, umax)={}\n\t(d2Fmin, d2Fmax)={}\n".format( \
                        p, (umin, umax),  (d2f(umin), d2f(umax)) ) )
        else:
            u_print = lambda p, d2f: None
        def find_u(phi):
            d2f = lambda u: self.model.d2F(phi, 1/u, FintKW=FintKW, XintKW=XintKW)
            # root_scalar: method="brenth"
            u_print(phi, d2f)
            ures = opt.root_scalar(d2f, x0=ui, x1=ui/2, rtol=thr, bracket=(umin,umax))
            return DTYPE(ures.root)
        # obtain good bracket first (unless specified)
        if not phi_bracket:
            pres = opt.minimize_scalar(find_u, method="bounded", bounds=(pmin,pmax/5), options={"xatol":thr})
        else:
            (pa,pb,pc) = phi_bracket
            # minimize_scalar: method="brent"
            try:
                pres = opt.minimize_scalar(find_u, bracket=(pa,pb,pc), tol=thr)
            except ValueError:
                print("\nBRACKET FAILED:\n\t(pa,pb,pc)={}\n\t(ua, ub, uc)={}\n".format(  \
                            (pa, pb, pc),  (find_u(pa), find_u(pb), find_u(pc)) ) )
                return
        # end timer
        t2 = perf_counter()
        print("\nTIME to find crit : %2.4f" % (t2-t1))
        # final values
        (pf, uf) = (pres.x, pres.fun)
        tf = 1/uf   # effective temperature is inverse of 'u'
        (self.pcrit, self.tcrit) = (DTYPE(pf), DTYPE(tf))       # store critical point
        self.pars.update( {"t_min":(min_frac*self.tcrit)} )     # adapt minimum for better searching/plotting
        return (DTYPE(pf), DTYPE(tf))

    #   partition paramter ~~ average volume fraction of sorts : follow linear path with 't' [ichi] (arbitrary)
    def phipar(self, t, phioff=0):
        return ( self.pcrit + phioff * (1-t/self.tcrit) )

    # total free energy from partitioning into phi1 & phi2, using phipar
    def Ftot(self, phi1, phi2, t, phip, x=1, FintKW={}, XintKW={}):
        args = (t, x, FintKW, XintKW)
        res = ( ((phi2-phip)/(phi2-phi1)) * self.model.F(phi1,*args) + \
                ((phip-phi1)/(phi2-phi1)) * self.model.F(phi2,*args) )
        return DTYPE(res)

    #   use numerical second derivative w/r/t phi, zeros phi1,phi2 at given t (ichi)
    def build_spino(self, t, scale=1e3, FintKW={}, XintKW={}):
        thr = self.pars["thr"]
        func = lambda p: DTYPE( (scale) * (self.model.d2F(p, t, FintKW=FintKW, XintKW=XintKW)) )
        sol1 = opt.root_scalar(func, x0=0.7*self.p_min, x1=0.9*self.p_min, \
                    bracket=(eps, self.p_min), xtol=thr, rtol=thr).root
        sol2 = opt.root_scalar(func, x0=(self.p_max+0.1*(self.realMax-self.p_max)), \
                    x1=(self.p_max+0.2*(self.realMax-self.p_max)), \
                    bracket=(self.p_max, self.realMax-eps), xtol=thr, rtol=thr).root
        return (DTYPE(sol1), DTYPE(sol2))

    #   evaluate partition method for chosen pars, and path viz. 'phipar' (minimum of Ftot)
    def extract(self, t, offset, parm="1d", FintKW={}, XintKW={}):
        cmax = self.pars["iterMax"]
        thr = self.pars["thr"]
        fac = 1e3   # scale up to exaggrate extrema
        if parm.lower() in option1:
            phi1i = 0.80*self.p_min        # initial values
            phi2i = self.p_max + (0.20)*(1-self.p_max)
            (a1,b1) = (eps, self.p_min-eps)       # domains
            (a2,b2) = (self.p_max+eps, self.realMax-eps)
            for n in range(cmax):
                # define new functions in each iteration
                func1 = lambda p1: DTYPE(fac * self.Ftot(p1, phi2i, t, self.phipar(t, offset), FintKW=FintKW, XintKW=XintKW))
                func2 = lambda p2: DTYPE(fac * self.Ftot(phi1i, p2, t, self.phipar(t, offset), FintKW=FintKW, XintKW=XintKW))
                # optimize
                phi1f = opt.minimize_scalar(func1, bounds=(a1, b1), method="bounded").x
                phi2f = opt.minimize_scalar(func2, bounds=(a2, b2), method="bounded").x
                # percent change in values
                perc1 = abs(phi1f-phi1i)/phi1i
                perc2 = abs(phi2f-phi2i)/phi2i
                if (perc1 < thr) and (perc2 < thr):
                    break
                else:
                    phi1i = phi1f
                    phi2i = phi2f
                    continue
            return (DTYPE(phi1f), DTYPE(phi2f))
        elif parm.lower() in option2:
            bound1 = (eps, self.p_min-eps)       # domains
            bound2 = (self.p_max+eps, self.realMax-eps)
            func = lambda p: DTYPE(fac * self.Ftot(p[0], p[1], t, self.phipar(t,offset), FintKW=FintKW, XintKW=XintKW))
            suc = False
            counter = 0
            while not suc:
                counter += 1
                phi1i = 0.9*rand()*self.p_min        # initial values
                phi2i = self.p_max + 0.1*rand()*(1-self.p_max)
                sol = opt.minimize(func, (phi1i,phi2i), method="Nelder-Mead", \
                                   bounds=(bound1,bound2) , tol=thr)
                suc = sol.success
                if counter >= cmax:
                    perc1 = abs(sol.x[0]-phi1i)/phi1i
                    perc2 = abs(sol.x[1]-phi2i)/phi2i
                    half2 = "(perc1,perc2) = (%1.2e,%1.2e)\n" % (perc1,perc2)
                    print("\n\n\tNO CONVERGENCE (total free energy 2d) : " + half2)
                    return tuple(sol.x)
                if not suc:
                    continue
            return tuple(sol.x)
        else:
            print("\n\n\tWARNING: improper choice given for parametric construction ('%s').\n" % parm)
            return

    #   evaluate Maxwell construction, finding zeros from dF/dphi = (F2-F1)/(phi2-phi1)
    def extractMax(self, t, maxm="1d", FintKW={}, XintKW={}):     
        cmax = self.pars["iterMax"]
        thr = self.pars["thr"]
        mchar = self.model.mode.lower()[0]    # initial character of 'mode' setting: 'f' for fG _or_ 'r' for rG
        fac = 1e2
        if maxm.lower() in option1:      # impliment iterated 1D solver at given ichi
            phi1i = 0.80*self.p_min        # initial values
            phi2i = self.p_max + (0.20)*(1-self.p_max)
            (a1,b1) = (eps, self.p_min-eps)       # domains
            (a2,b2) = (self.p_max+eps, self.realMax-eps)
            for n in range(cmax):
                # must find 'x' at each initial phi2
                if mchar == "f":
                    x2 = 0
                elif mchar == "r":
                    x2 = self.model.find_x(phi2i, t, XintKW) 
                # define new function in each iteration
                def func1(p1):
                    if mchar == "f":
                        x = 1       # fixed Gaussian: just use unity
                        dx = 0      # any derivatives are zero
                    elif mchar == "r":
                        x = self.model.find_x(p1, t, XintKW)                     # renormalized Gaussian: solve for 'x'
                        XintD = self.model.XintD(p1, t, x, XintKW)                   # common integral (derivative denominators)
                        dx = self.model.find_dx(p1, t, x, XintD, XintKW)             # also need derivative
                    return DTYPE(abs(self.model.dF(p1,t,x,dx,FintKW) - \
                        ( self.model.F(phi2i,t,x2,FintKW) - self.model.F(p1,t,x,FintKW) ) / ( phi2i - p1 )))*fac
                # minimize absolute value to find approximate root
                phi1f = opt.minimize_scalar(func1, bounds=(a1, b1), method="bounded").x
                # must find 'x' at each final phi1
                if mchar == "f":
                    x1 = 0
                elif mchar == "r":
                    x1 = self.model.find_x(phi1f, t, XintKW)                   
                # repeat for phi2
                def func2(p2):
                    if mchar == "f":
                        x = 1       # fixed Gaussian: just use unity
                        dx = 0      # any derivatives are zero
                    elif mchar == "r":
                        x = self.model.find_x(p2, t, XintKW)                     # renormalized Gaussian: solve for 'x'
                        XintD = self.model.XintD(p2, t, x, XintKW)                   # common integral (derivative denominators)
                        dx = self.model.find_dx(p2, t, x, XintD, XintKW)             # also need derivative
                    return DTYPE(abs(self.model.dF(p2,t,x,dx,FintKW) - \
                        ( self.model.F(p2,t,x,FintKW) - self.model.F(phi1f,t,x1,FintKW) ) / ( p2 - phi1f )))*fac
                phi2f = opt.minimize_scalar(func2, bounds=(a2, b2), method="bounded").x
                # percent change upon update
                perc1 = abs(phi1f-phi1i)/phi1i
                perc2 = abs(phi2f-phi2i)/phi2i
                if (perc1 < thr) and (perc2 < thr):
                    break
                else:
                    phi1i = phi1f
                    phi2i = phi2f
                    if n == cmax-1:
                        half2 = "(perc1,perc2) = (%1.2e,%1.2e)\n" % (perc1,perc2)
                        print("\n\n\tNO CONVERGENCE (maxwell) : " + half2)
                    continue
            return (DTYPE(phi1f), DTYPE(phi2f))
        elif maxm.lower() in option2:     # impliment multidimensional solver at given ichi
            suc = False
            counter = 0
            init = [0.9*self.p_min,
                        self.p_max + 0.1*(self.realMax-self.p_max)]    # initial value of vector (phi1, phi2)
            def funcMax(pvec):
                if np.isclose(pvec[0],pvec[1],thr):
                    return [ DTYPE(2) , DTYPE(2)]
                slope = ( self.model.F(pvec[1],t) - self.model.F(pvec[0],t) ) / ( pvec[1] - pvec[0] )
                return [ DTYPE(self.model.dF(pvec[0],t) - slope) , DTYPE(slope - self.model.dF(pvec[1],t)) ]
            def jac(pvec):  # jacobian matrix -> better solving
                d1 = - ( self.model.F(pvec[1],t) - self.model.F(pvec[0],t) )/( (pvec[1] - pvec[0])*(pvec[1] - pvec[0]) )
                j12 = - d1 - self.model.dF(pvec[1],t)/(pvec[1] - pvec[0])
                j21 = - d1 - self.model.dF(pvec[0],t)/(pvec[1] - pvec[0])
                j11 = self.model.d2F(pvec[0],t) - j21
                j22 = - self.model.d2F(pvec[1],t) - j12
                return np.array([[j11,j12],[j21,j22]], dtype=DTYPE)
            while not suc:
                counter += 1               
                sol = opt.root(funcMax, init, jac=jac, method="df-sane")  #"df-sane"
                # 'minimize' methods: Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, trust-constr
                if counter >= cmax:
                    perc1 = abs(init[0]-sol.x[0])/init[0]
                    perc2 = abs(init[1]-sol.x[1])/init[1]
                    [perc1, perc2] = funcMax(sol.x)
                    half2 = "(perc1,perc2) = (%1.2e,%1.2e)\n" % (perc1,perc2)
                    print("\n\n\tNO CONVERGENCE (maxwell 2d) : " + half2)
                    return tuple(sol.x)
                if not suc:
                    init = sol.x
                    continue
            return tuple(sol.x)
        else:
            print("\n\n\tWARNING: improper choice given for Maxwell construction ('%s').\n" % maxm)
            return

    #   evaluate chosen method(s) & store resulting lists
    def evaluate(self, UNevenTspace=0.20):
        print("\nEvaluating...")
        t1 = perf_counter()
        (pc, tc) = (self.pcrit, self.tcrit)
        # grab temperature minimum & points
        t_min, t_points = self.pars["t_min"], self.pars["t_points"]
        ##  build binodal curve(s)
        spinox = [pc]
        ty = [tc]
        self.p_min, self.p_max = pc, pc     # both spino sides begin at critical
        res = {}
        start = spinox.copy()
        mds = self.methods
        for m in mds:
            m = m.lower()
            res.update({m:start.copy()})
        if not UNevenTspace:
            t_space = np.linspace(tc, t_min, t_points)[1:]    # evenly-spaced temperatures
        else:
            # alternative temperature space: make half the points in a small fraction of range, near T_crit ('tc')
            t_frac = UNevenTspace   # fraction of temp. range with finely-spaced points (half the points are in this range
            t_space = np.concatenate( ( np.linspace(tc, tc-t_frac*(tc-t_min), t_points//2)[1:],
                            np.linspace(tc-t_frac*(tc-t_min), t_min, t_points-t_points//2)[1:] ) )
        for ax in t_space:
            (self.p_min, self.p_max) = self.build_spino(ax)     # get spinodal boundaries
            spinox = [self.p_min] + spinox + [self.p_max]       # build spinodal list
            ty = [ax] + ty + [ax]   # build control parameter (ichi) space correspondingly
            for m in mds:
                v = mds[m]
                m = m.lower()
                if m[0] == "p":
                    (dim, off) = self.pCheck(m,v)
                    (ap1, ap2) = self.extract(ax,off,dim)    # extract binodal points from minimizing Ftot
                    nex = res[m]   # place new points on either side
                    nex = [ap1] + nex + [ap2]
                    res.update({m:nex})
                elif m[0] == "m":
                    v = self.mCheck(m,v)
                    (ap1, ap2) = self.extractMax(ax,v)   # extract binodal points from Maxwell construction
                    nex = res[m]   # place new points on either side
                    nex = [ap1] + nex + [ap2]
                    res.update({m:nex})
                else:
                    print( ("\n\nWARNING: invalid method found '%s'." % m) )
                    continue
            self.spinox = np.array(spinox, dtype=DTYPE)
            self.ty = np.array(ty, dtype=DTYPE)
            self.results = res
        t2 = perf_counter()
        print("\nTIME to evaluate : {:2.4f}\n\n".format(t2-t1))
        return

    #   nice plotter
    def plot(self,include="all", sz=7, leg_loc="best", crit_detail=True, minorticks=2,
                markers={}, colors={}, styles={}, line_widths={}, labels={}, grid_opts={}):
        # handle plot options - defaults & updates
#        clrs = {"spino":"#75bbfd","bino":"#ff474c", "crit":"#ffff84"}
        clrs = {"spino":"slateblue", "bino":"orangered", "crit":"green"}
        clrs.update(colors)
        stys = {"spino":"s-.", "bino":"o-", "crit":"^"}
        stys.update(styles)
        l_ws = {"spino":1.4, "bino":1.4, "crit":1.0}
        l_ws.update(line_widths)
        mks = {"spino":3, "bino":3, "crit":7}
        mks.update(markers)
        lbls = {"spino":"spinodal", "bino":"binodal", "crit":"crit. pt."}
        lbls.update(labels)
        grid_kw = {"visible":True, "which":"both", "dashes":(2,5), "linewidth":0.8}
        grid_kw.update(grid_opts)
        # can include all methods, or a single one
        if include == "all":
            mds = self.methods
        else:
            mds = {include:self.methods[include]}
        res = self.results
        fig = plt.figure("Coexistence Curves", (1.4*sz,sz))
        ax = fig.add_subplot(111)
        if crit_detail:     # show crit. pt. clearly
            tcr, pcr = self.tcrit, self.pcrit
            ax.plot([pcr,pcr], [0,tcr], ":", markersize=0, color=clrs["crit"], linewidth=l_ws["crit"], label=None)
            ax.plot([0,pcr], [tcr,tcr], ":", markersize=0, color=clrs["crit"], linewidth=l_ws["crit"], label=None)
        ax.plot(self.spinox, self.ty, stys["spino"], label=lbls["spino"], markersize=mks["spino"],
                linewidth=l_ws["spino"], color=clrs["spino"])
        for m in mds:
            v = mds[m]
            m.lower()
            if m[0] == "p":
                (dim, off) = self.pCheck(m,v)       # need to grab the correct offset
                pline = ax.plot(res[m], self.ty, stys["bino"], label=lbls["bino"], markersize=mks["bino"],
                                linewidth=l_ws["bino"], color=clrs["bino"])
                pcolor = pline[-1].get_color()
                ax.plot(self.phipar(self.ty,off), self.ty, ":", markersize=0, color=pcolor)
            elif m[0] == "m":
                ax.plot(res[m], self.ty, stys["bino"], label=lbls["bino"], markersize=mks["bino"],
                        linewidth=l_ws["bino"], color=clrs["bino"])
        ax.plot([self.pcrit],[self.tcrit], stys["crit"], markersize=mks["crit"], label=lbls["crit"], color=clrs["crit"])
        ax.set_xlim(0,1)
        ax.set_ylim(0,1.1*max(self.ty))
        ax.xaxis.set_minor_locator(AutoMinorLocator(minorticks))
        ax.yaxis.set_minor_locator(AutoMinorLocator(minorticks))
        if crit_detail:     # label values to emphasize crit. pt. (ticks are troublesome)
            xlbl = "%1.3e" % pcr
            plt.text(pcr, tcr/100, xlbl)
            ylbl = "%1.3f" % tcr
            plt.text(pcr/3, 1.01*tcr, ylbl)
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$\ell/\ell_B$")
        ax.set_title(r"LLPS : length $N=${:}, sequence '{}' ['{}']  (SCD={:.4f})".format( \
                    self.model.seq.N, self.model.seq.seqName, self.model.seq.seqAlias, self.model.seq.SCD) )
        ax.legend(loc=leg_loc, edgecolor="inherit")
        plt.grid(**grid_kw)
        fig.tight_layout()
        plt.show()
        plt.close()
        return
#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####
