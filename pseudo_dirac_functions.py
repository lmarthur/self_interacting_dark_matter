#This file contains the necessary functions to calculate pseudo-dirac bound and scattering states

import numpy as np
import scipy.integrate as integrate
import math

from scipy.special import genlaguerre
    
#Coulomb radial component
def rad(r, ns, angl):
        return np.sqrt(((coeff/ns)**3)*math.factorial((ns-angl-1))/(2*ns*math.factorial(ns+angl)))*math.exp(-coeff*r/(2*ns))*((coeff*r/ns)**angl)*genlaguerre(ns-angl-1, 2*angl+1)(coeff*r/ns)
    
#L+S=even bound state calculator
def boundstates0(mchi, dim, l):
    snumber=2
    alpha=1
    coeff=1
    delta=0.01 
    ed=0
    mphi=1
    vmatrix=np.zeros((snumber, snumber), dtype=object)
    ephi=mphi/(mchi*alpha)
    
    #Radial wavefunction expansion
    def rad(r, ns, angl):
        return np.sqrt(((coeff/ns)**3)*math.factorial((ns-angl-1))/(2*ns*math.factorial(ns+angl)))*math.exp(-coeff*r/(2*ns))*((coeff*r/ns)**angl)*genlaguerre(ns-angl-1, 2*angl+1)(coeff*r/ns)

    #Custom Kronecker Delta Function
    def kronecker(a, b):
        if a==b:
            return 1
        else:
            return 0
    
    def vfunc(r):
        return np.exp(-ephi*r)/r
    
    def v00(r):
        return 0
    def v01(r):
        return vfunc(r)
    def v11(r):
        return -ed**2
    
    vmatrix[0, 0]=v00
    vmatrix[0, 1]=v01
    vmatrix[1, 0]=v01
    vmatrix[1, 1]=v11
    
    p=int(dim/snumber)
    
    mom=l
    h=np.zeros((dim, dim))
    
    integrand0 = lambda r : (r**2)*rad(r, i+mom, mom)*rad(r, j+mom, mom)*vmatrix[m-1, n-1](r)
    integrand1 = lambda r : (r**2)*rad(r, i+mom, mom)*rad(r, j+mom, mom)*coeff/r

    
    for i in range(1, p+1):
        for j in range(1, p+1):
            for n in range(1, snumber+1):
                for m in range(1, snumber+1):
                    new=-integrate.quad(integrand0, 0, np.inf)[0]+kronecker(m, n)*(-coeff**2/(4*(j+mom)**2)*kronecker(i, j)+integrate.quad(integrand1, 0, np.inf)[0])
                    h[i+(m-1)*p-1, j+(n-1)*p-1]=new
    evals, evecs = np.linalg.eig(h)
    evals = evals + delta/(mchi*alpha**2)
    sol=[evals, evecs]
    return sol

#L+S=odd bound state calculator
def boundstates1(mchi, dim, l):
    snumber=1
    alpha=1
    coeff=1
    ed=0
    mphi=1
    vmatrix=np.zeros((snumber, snumber), dtype=object)
    ephi=mphi/(mchi*alpha)
    
        #Radial wavefunction expansion
    def rad(r, ns, angl):
        return np.sqrt(((coeff/ns)**3)*math.factorial((ns-angl-1))/(2*ns*math.factorial(ns+angl)))*math.exp(-coeff*r/(2*ns))*((coeff*r/ns)**angl)*genlaguerre(ns-angl-1, 2*angl+1)(coeff*r/ns)

    #Custom Kronecker Delta Function
    def kronecker(a, b):
        if a==b:
            return 1
        else:
            return 0
    
    def vfunc(r):
        return np.exp(-ephi*r)/r
    
    def v00(r):
        return vfunc(r)
    
    vmatrix[0, 0]=v00
    
    p=int(dim/snumber)
    
    mom=l
    h=np.zeros((dim, dim))
    
    integrand0 = lambda r : (r**2)*rad(r, i+mom, mom)*rad(r, j+mom, mom)*vmatrix[m-1, n-1](r)
    integrand1 = lambda r : (r**2)*rad(r, i+mom, mom)*rad(r, j+mom, mom)*coeff/r

    
    for i in range(1, p+1):
        for j in range(1, p+1):
            for n in range(1, snumber+1):
                for m in range(1, snumber+1):
                    new=-integrate.quad(integrand0, 0, np.inf)[0]+kronecker(m, n)*(-coeff**2/(4*(j+mom)**2)*kronecker(i, j)+integrate.quad(integrand1, 0, np.inf)[0])
                    h[i+(m-1)*p-1, j+(n-1)*p-1]=new
    
    evals, evecs = np.linalg.eig(h)
    evals = evals + (ed**2)/2
    sol=[evals, evecs]
    return sol
