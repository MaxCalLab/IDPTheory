##  Mike Phillips, 05/29/2022
##  Big Update: 06/10/2022
##  File with functions in support of 'Model' for rG-RPA calculations.
##  Attempt at simplifying 'Model' class definition.
##  -> There: _integrands_ must be called for each 'k',
##      thus requiring reference to sequence-specific quantities ('G', 'g', 'Z')
##      as well as parameter-specific quantities ('c_fac', 's_fac', 'v2', 'lam').
##  > Here: we may define functions of the above quantities, which do not reference
##     the sequence or parameters directly.
##  > Also: details of structure functions are here instead, loaded into 'Model' upon initialization.
##
##  <structure functions>
##  - 3 normal ones (power 0), 3 modified ones (designed for integer power n>0)
##    = 6 functions
##
##  Lots of functions!          *[Note: 'I', 'J', 'dJdp', 'dJdx' are actually integrands, therefore not here.]*
##  <'F' integrals>
##  - All mixed partials of 'I' up to third order (except zeroth: I)
##      > 9 functions: of function in log, 'R', and its derivatives
##  - All mixed partials of 'R' to third order (including zeroth)
##      > 10 functions: of above quantities (sequence- and parameter- specific), and 'phi', 'alpha'
##  - The 'alpha' quantity, and derivative forms to third order
##      > 4 functions: of just 'G', 'g', 'Z'
##    = 23 functions!   <just for FE derivatives>
##  <polymer-only integrals>
##  - Different 'R' function for just polymer, when ionic contribution is integrated separately
##      > 10 alternative functions for 'R' and its derivatives
##  <'x' solvers>
##  - All mixed partials of 'J' up to third order (except zeroth, and first order: J, dJ/dp & dJ/dx)
##      > 7 functions: of numerator 'E' and denominator 'D', and their derivatives
##  - All mixed partials of 'E' and 'D' to third order
##      > 20 functions: of above quantities, and 'phi', sometimes 'kfac' and/or 'beta'
##  - The 'beta' quantity, and its derivative forms to third order
##      > 4 functions: of 'G', 'g', 'Z'
##    = 31 functions!   <just for 'x' solvers>
##  = 70 functions total!


import numpy as np

##  scalar weighted 'total' of correlation matrix, renormalized-Gaussian option  [a.k.a. 'xi' cf. Chan & Ghosh]
def totG(k2, x, seq, rng_n=0):
    # total of nonzero terms is the sum of a product, now with 'n'th power range also
    total = np.sum( 2 * np.exp(-seq.refRange1*x*k2/6) * seq.sumsig1 * rng_n)
    # zero term vanishes with range powers n>0
    return (total/seq.N + seq.totalsig*(rng_n[-1]==1))
##   scalar non-weighted 'total' of correlation matrix, rG option    [a.k.a. 'g' cf. Chan & Ghosh]
def totMinig(k2, x, seq, rng_n=0):
    # total of nonzero terms is the sum of a product, now with 'n'th power range also
    total = np.sum( 2 * np.exp(-seq.refRange1*x*k2/6) * (seq.refRange1_dif) * rng_n)
    return (total/seq.N + 1*(rng_n[-1]==1))
##   scalar semi-weighted 'total' of correlation matrix, rG option   [a.k.a. 'zeta' cf. Chan & Ghosh]
def totZ(k2, x, seq, rng_n=0):
    # total of nonzero terms is the sum of a product, now with 'n'th power range also
    total = np.sum( np.exp(-seq.refRange1*x*k2/6) * seq.sumqoff1 * rng_n )
    return ((total + seq.qtot*(rng_n[-1]==1))/seq.N)


#   derivatives for FE integrands -> separate integrand functions will combine these [in 'RPA_coex_X']
##     [Note: factor k^2 is OMITTED in the following -> to be included in full integrand functinos.]
# using pre-calculated  Rshift = 1+R
def dIdp(Rshift, dRdp):
    return ( dRdp / (Rshift) )
def dIdx(Rshift, dRdx):
    return ( dRdx / (Rshift) )

def d2Idp2(Rshift, dRdp, d2Rdp2):
    return ( (d2Rdp2 - (dRdp*dRdp) / Rshift) / Rshift )
def d2Idxdp(Rshift, dRdp, dRdx, d2Rdxdp):
    return ( (d2Rdxdp - (dRdp*dRdx) / Rshift) / Rshift )
def d2Idx2(Rshift, dRdx, d2Rdx2):
    return ( (d2Rdx2 - (dRdx*dRdx) / Rshift) / Rshift )

def d3Idp3(Rshift, dRdp, d2Rdp2, d3Rdp3):
    return ( (d3Rdp3 - (3*d2Rdp2*dRdp - 2*(dRdp*dRdp*dRdp) / Rshift) / Rshift) / Rshift )
def d3Idxdp2(Rshift, dRdp, dRdx, d2Rdp2, d2Rdxdp, d3Rdxdp2):
    return ( (d3Rdxdp2 - (2*dRdp*d2Rdxdp + d2Rdp2*dRdx - 2*dRdx*(dRdp*dRdp) / Rshift) / Rshift) / Rshift )
def d3Idx2dp(Rshift, dRdp, dRdx, d2Rdx2, d2Rdxdp, d3Rdx2dp):
    return ( (d3Rdx2dp - (2*dRdx*d2Rdxdp + d2Rdx2*dRdp - 2*dRdp*(dRdx*dRdx) / Rshift) / Rshift) / Rshift )
def d3Idx3(Rshift, dRdx, d2Rdx2, d3Rdx3):
    return ( (d3Rdx3 - (3*d2Rdx2*dRdx - 2*(dRdx*dRdx*dRdx) / Rshift) / Rshift) / Rshift )

#   details of 'I': function in logarithm (R), _AND_ derivatives
def R(phi, v2, c_fac, s_fac, k2lam, G0, g0, A0):
    terms0 = k2lam * s_fac
    terms1 = k2lam * (G0 + c_fac + v2*s_fac*g0) + v2*g0
    terms2 = k2lam * v2 * (A0 + c_fac*g0)
    return ( terms0 + (terms1 + terms2*phi)*phi )

def dRdp(phi, v2, c_fac, s_fac, k2lam, G0, g0, A0):
    terms0 = k2lam * (G0 + c_fac + v2*s_fac*g0) + v2*g0
    terms1 = 2 * k2lam * v2 * (A0 + c_fac*g0)
    return ( terms0 + terms1*phi )
def dRdx(kfac1, phi, v2, c_fac, s_fac, k2lam, G1, g1, A1):
    terms1 = k2lam * (G1 + v2*s_fac*g1) + v2*g1
    terms2 = k2lam * v2 * (A1 + c_fac*g1)
    return ( kfac1 * (terms1 + terms2*phi)*phi )

def d2Rdp2(phi, v2, c_fac, s_fac, k2lam, g0, A0):
    return ( 2 * k2lam * v2 * (A0 + c_fac*g0) )
def d2Rdxdp(kfac1, phi, v2, c_fac, s_fac, k2lam, G1, g1, A1):
    terms0 = k2lam * (G1 + v2*s_fac*g1) + v2*g1
    terms1 = 2 * k2lam * v2 * (A1 + c_fac*g1)
    return ( kfac1 * (terms0 + terms1*phi) )
def d2Rdx2(kfac2, phi, v2, c_fac, s_fac, k2lam, G2, g2, A2):
    terms1 = k2lam * (G2 + v2*s_fac*g2) + v2*g2
    terms2 = k2lam * v2 * (A2 + c_fac*g2)
    return ( kfac2 * (terms1 + terms2*phi)*phi )

def d3Rdp3(phi, v2, c_fac, s_fac, k2lam):
    return 0
def d3Rdxdp2(kfac1, phi, v2, c_fac, s_fac, k2lam, g1, A1):
    terms0 = 2 * k2lam * v2 * (A1 + c_fac*g1)
    return ( kfac1 * terms0 )
def d3Rdx2dp(kfac2, phi, v2, c_fac, s_fac, k2lam, G2, g2, A2):
    terms0 = k2lam * (G2 + v2*s_fac*g2) + v2*g2
    terms1 = 2 * k2lam * v2 * (A2 + c_fac*g2)
    return ( kfac2 * (terms0 + terms1*phi) )
def d3Rdx3(kfac3, phi, v2, c_fac, s_fac, k2lam, G3, g3, A3):
    terms1 = k2lam * (G3 + v2*s_fac*g3) + v2*g3
    terms2 = k2lam * v2 * (A3 + c_fac*g3)
    return ( kfac3 * (terms1 + terms2*phi)*phi )

#   ALTERNATIVE details of 'I': log function (R) _just for polymer_ with ions captured in screening
def R_Ponly(phi, v2, c_fac, nu, G0, g0, A0):
    terms1 = G0/nu + v2*g0
    terms2 = (v2/nu)*A0
    return ( (terms1 + terms2*phi)*phi )

def dRdp_Ponly(phi, v2, c_fac, nu, G0, g0, A0):
    cnu = c_fac/(nu*nu)
    terms0 = G0/nu + v2*g0
    terms1 = 2*(v2/nu)*A0 - cnu*G0
    terms2 = -cnu*v2*A0
    return ( terms0 + (terms1 + terms2*phi)*phi )
def dRdx_Ponly(kfac1, phi, v2, c_fac, nu, G1, g1, A1):
    terms1 = G1/nu + v2*g1
    terms2 = (v2/nu)*A1
    return ( kfac1 * (terms1 + terms2*phi)*phi )

def d2Rdp2_Ponly(phi, v2, c_fac, nu, G0, g0, A0):
    cnu = c_fac/(nu*nu)
    cnu2 = cnu * c_fac/nu
    terms0 = 2*((v2/nu)*A0 - cnu*G0)
    terms1 = 2*(-2*cnu*v2*A0 + cnu2*G0)
    terms2 = 2*cnu2*v2*A0
    return ( terms0 + (terms1 + terms2*phi)*phi )
def d2Rdxdp_Ponly(kfac1, phi, v2, c_fac, nu, G1, g1, A1):
    cnu = c_fac/(nu*nu)
    terms0 = G1/nu + v2*g1
    terms1 = 2*(v2/nu)*A1 - cnu*G1
    terms2 = -cnu*v2*A1
    return ( kfac1 * (terms0 + (terms1 + terms2*phi)*phi) )
def d2Rdx2_Ponly(kfac2, phi, v2, c_fac, nu, G2, g2, A2):
    terms1 = G2/nu + v2*g2
    terms2 = (v2/nu)*A2
    return ( kfac2 * (terms1 + terms2*phi)*phi )

def d3Rdp3_Ponly(phi, v2, c_fac, nu, G0, g0, A0):
    cnu = c_fac/(nu*nu)
    cnu2 = cnu * c_fac/nu
    cnu3 = cnu2 * c_fac/nu
    terms0 = 6*(-cnu*v2*A0 + cnu2*G0)
    terms1 = 6*(2*cnu2*v2*A0 - cnu3*G0)
    terms2 = -6*cnu3*v2*A0
    return ( terms0 + (terms1 + terms2*phi)*phi )
def d3Rdxdp2_Ponly(kfac1, phi, v2, c_fac, nu, G1, g1, A1):
    cnu = c_fac/(nu*nu)
    cnu2 = cnu * c_fac/nu
    terms0 = 2*((v2/nu)*A1 - cnu*G1)
    terms1 = 2*(-2*cnu*v2*A1 + cnu2*G1)
    terms2 = 2*cnu2*v2*A1
    return ( kfac1 * (terms0 + (terms1 + terms2*phi)*phi) )
def d3Rdx2dp_Ponly(kfac2, phi, v2, c_fac, nu, G2, g2, A2):
    cnu = c_fac/(nu*nu)
    terms0 = G2/nu + v2*g2
    terms1 = 2*(v2/nu)*A2 - cnu*G2
    terms2 = - cnu*v2*A2
    return ( kfac2 * (terms0 + (terms1 + terms2*phi)*phi) )
def d3Rdx3_Ponly(kfac3, phi, v2, c_fac, nu, G3, g3, A3):
    terms1 = G3/nu + v2*g3
    terms2 = (v2/nu)*A3
    return ( kfac3 * (terms1 + terms2*phi)*phi )


#   handy quantity 'alpha', and its derivative forms [w/r/t 'x']
def alpha0(G0, g0, Z0):
    return ( G0*g0 - (Z0*Z0) )
def alpha1(G0,G1, g0,g1, Z0,Z1):
    return ( G1*g0 + G0*g1 - 2*Z0*Z1 )
def alpha2(G0,G1,G2, g0,g1,g2, Z0,Z1,Z2):
    return ( G2*g0 + 2*G1*g1 + G0*g2 - 2*((Z1*Z1)+Z0*Z2) )
def alpha3(G0,G1,G2,G3, g0,g1,g2,g3, Z0,Z1,Z2,Z3):
    return ( G3*g0 + 3*G2*g1 + 3*G1*g2 + G0*g3 - 2*(3*Z1*Z2 + Z0*Z3) )


#   (most) derivatives for 'x' integrands -> separate integrand functions will combine these [in 'RPA_coex_X']
##     [Note: factor k^4 is OMITTED in the following -> to be included with 'x' derivative solvers.]
def d2Jdp2(E, D, dEdp, dDdp, d2Ddp2):
    return ( (-2*dEdp*dDdp - E*d2Ddp2 + 2*E*(dDdp*dDdp) / D) / (D*D) )
def d2Jdxdp(E, D, dEdp, dEdx, dDdp, dDdx, d2Edxdp, d2Ddxdp):
    return ( (d2Edxdp - (dEdp*dDdx + dEdx*dDdp + E*d2Ddxdp - 2*E*dDdx*dDdp / D) / D) / D )
def d2Jdx2(E, D, dEdx, dDdx, d2Edx2, d2Ddx2):
    return ( (d2Edx2 - (2*dEdx*dDdx + E*d2Ddx2 - 2*E*(dDdx*dDdx) / D) / D) / D )

def d3Jdp3(E, D, dEdp, dDdp, d2Ddp2):
    return ( -3 * (dEdp*d2Ddp2 - 2*(dEdp*(dDdp*dDdp) + E*dDdp*d2Ddp2 - E*(dDdp*dDdp*dDdp) / D) / D) / (D*D) )
def d3Jdxdp2(E, D, dEdp, dEdx, dDdp, dDdx, d2Edxdp, d2Ddp2, d2Ddxdp, d3Ddxdp2):
    termsD2 = -2*d2Edxdp*dDdp - dEdx*d2Ddp2 - 2*dEdp*d2Ddxdp - E*d3Ddxdp2
    termsD3 = 2 * ( dEdx*(dDdp*dDdp) + 2*dEdp*dDdx*dDdp + 2*E*d2Ddxdp*dDdp + E*dDdx*d2Ddp2 )
    return ( (termsD2 + (termsD3 - 6*E*dDdx*(dDdp*dDdp) / D) / D) / (D*D) )
def d3Jdx2dp(E, D, dEdp, dEdx, dDdp, dDdx, d2Edxdp, d2Edx2, d2Ddxdp, d2Ddx2, d3Edx2dp, d3Ddx2dp):
    termsD2 = -2*d2Edxdp*dDdx - dEdp*d2Ddx2 - d2Edx2*dDdp - 2*dEdx*d2Ddxdp - E*d3Ddx2dp
    termsD3 = 2 * ( dEdp*(dDdx*dDdx) + 2*dEdx*dDdp*dDdx + 2*E*d2Ddxdp*dDdx + E*dDdp*d2Ddx2 )
    return ( ( d3Edx2dp + (termsD2 + (termsD3 - 6*E*dDdp*(dDdx*dDdx) / D) / D) / D ) / D )
def d3Jdx3(E, D, dEdx, dDdx, d2Edx2, d2Ddx2, d3Edx3, d3Ddx3):
    termsD2 = -3*d2Edx2*dDdx - 3*dEdx*d2Ddx2 - E*d3Ddx3
    termsD3 = 6 * ( dEdx*(dDdx*dDdx) + E*dDdx*(d2Ddx2) )
    return ( ( d3Edx3 + (termsD2 + (termsD3 - 6*E*(dDdx*dDdx*dDdx) / D) / D) / D ) / D )

#   details of 'J': NUMERATOR (E) and DENOMINATOR (D), _AND_ derivatives [all partials up to 3rd order!]
# NUMERATOR
def E(phi, v2, c_fac, s_fac, ilam, G2, g2, B0):
    return ( G2 + v2*g2*(ilam+s_fac) + v2*phi*(c_fac*g2+B0) )
# dE/dphi
def dEdp(phi, v2, c_fac, s_fac, ilam, g2, B0):
    return ( v2*(c_fac*g2+B0) )
# d^2E/dphi^2     : identically zero
def d2Edp2(phi, v2, c_fac, s_fac, ilam):
    return 0
# d^3E/dphi^3     : identically zero
def d3Edp3(phi, v2, c_fac, s_fac, ilam):
    return 0
# dE/dx
def dEdx(kfac1, phi, v2, c_fac, s_fac, ilam, G3, g3, B1):
    return ( kfac1 * (G3 + v2*g3*(ilam+s_fac) + v2*phi*(c_fac*g3+B1)) )
# d^2E/dxdphi
def d2Edxdp(kfac1, phi, v2, c_fac, s_fac, ilam, g3, B1):
    return ( kfac1 * v2*(c_fac*g3+B1) )
# d^2E/dx^2
def d2Edx2(kfac2, phi, v2, c_fac, s_fac, ilam, G4, g4, B2):
    return ( kfac2 * (G4 + v2*g4*(ilam+s_fac) + v2*phi*(c_fac*g4+B2)) )
# d^3E/dxdphi^2     : identically zero
def d3Edxdp2(kfac1, phi, v2, c_fac, s_fac, ilam):
    return 0
# d^3E/dx^2dphi
def d3Edx2dp(kfac2, phi, v2, c_fac, s_fac, ilam, g4, B2):
    return ( kfac2 * v2*(c_fac*g4+B2) )
# d^3E/dx^3
def d3Edx3(kfac3, phi, v2, c_fac, s_fac, ilam, G5, g5, B3):
    return ( kfac3 * (G5 + v2*g5*(ilam+s_fac) + v2*phi*(c_fac*g5+B3)) )

# DENOMINATOR
def D(phi, v2, c_fac, s_fac, ilam, G0, g0, A0):
    return ( (ilam+s_fac) + phi*(c_fac+G0) + v2*phi*(g0*(ilam+s_fac) + phi*(A0+c_fac*g0)) )
# dD/dphi
def dDdp(phi, v2, c_fac, s_fac, ilam, G0, g0, A0):
    return ( c_fac + G0 + v2*g0*(ilam+s_fac) + 2*v2*phi*(A0+c_fac*g0) )
# d^2D/dphi^2
def d2Ddp2(phi, v2, c_fac, s_fac, ilam, g0, A0):
    return ( 2*v2*(A0+c_fac*g0) )
# d^3D/dphi^3     : identically zero
def d3Ddp3(phi, v2, c_fac, s_fac, ilam):
    return 0
# dD/dx
def dDdx(kfac1, phi, v2, c_fac, s_fac, ilam, G1, g1, A1):
    return ( kfac1 * phi * (G1 + v2*g1*(ilam+s_fac) + v2*phi*(A1+c_fac*g1)) )
# d^2D/dxdphi
def d2Ddxdp(kfac1, phi, v2, c_fac, s_fac, ilam, G1, g1, A1):
    return ( kfac1 * (G1 + v2*g1*(ilam+s_fac) + 2*v2*phi*(A1+c_fac*g1)) )
# d^2D/dx^2
def d2Ddx2(kfac2, phi, v2, c_fac, s_fac, ilam, G2, g2, A2):
    return ( kfac2 * phi * (G2 + v2*g2*(ilam+s_fac) + v2*phi*(A2+c_fac*g2)) )
# d^3D/dxdphi^2
def d3Ddxdp2(kfac1, phi, v2, c_fac, s_fac, ilam, g1, A1):
    return ( kfac1 * 2*v2*(A1+c_fac*g1) )
# d^3D/dx^2dphi
def d3Ddx2dp(kfac2, phi, v2, c_fac, s_fac, ilam, G2, g2, A2):
    return ( kfac2 * (G2 + v2*g2*(ilam+s_fac) + 2*v2*phi*(A2+c_fac*g2) ) )
# d^3D/dx^3
def d3Ddx3(kfac3, phi, v2, c_fac, s_fac, ilam, G3, g3, A3):
    return ( kfac3 * phi * (G3 + v2*g3*(ilam+s_fac) + v2*phi*(A3+c_fac*g3)) )

#   handy quantity 'beta', and its derivative forms [w/r/t 'x']        <better: in terms of G0... directly>
#   also: now separated instead of using 'if/elif'
def beta0(G0,G2, g0,g2, Z0,Z2):
    return (G2*g0 + G0*g2 - 2*Z0*Z2)
# from d/dx
def beta1(G0,G1,G2,G3, g0,g1,g2,g3, Z0,Z1,Z2,Z3):
    return (G3*g0 + G2*g1 + G1*g2 + G0*g3 - 2*(Z1*Z2 + Z0*Z3))
# from d^2/dx^2
def beta2(G0,G1,G2,G3,G4, g0,g1,g2,g3,g4, Z0,Z1,Z2,Z3,Z4):
    return (G4*g0 + 2*G3*g1 + 2*G2*g2 + 2*G1*g3 + G0*g4 - 2*(Z2*Z2 + 2*Z1*Z3 + Z0*Z4))
# from d^3/dx^3
def beta3(G0,G1,G2,G3,G4,G5, g0,g1,g2,g3,g4,g5, Z0,Z1,Z2,Z3,Z4,Z5):
    return (G5*g0 + 3*G4*g1 + 4*G3*g2 + 4*G2*g3 + 3*G1*g4 + G0*g5 - 2*(4*Z2*Z3 + 3*Z1*Z4 + Z0*Z5))

