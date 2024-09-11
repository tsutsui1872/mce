"""
API for computing carbon cycle, developed based on
Hooss et al. (2001, https://doi.org/10.1007/s003820100170) and
Joos et al. (1996, https://doi.org/10.3402/tellusb.v48i3.15921).
"""

import numpy as np
from numpy.linalg import LinAlgError
from .. import MCECalibError
from . import ParmsBase
from . import ModelBase

class ParmsOcean(ParmsBase):
    """
    Ocean carbon cycle parameters
    """
    def __init__(self):
        self.add(
            'cco2_pi', 278.,
            'Preindustrial CO2 concentration', 'ppm', False,
        )
        # Prather et al. (2012, https://doi.org/10.1029/2012GL051440)
        # 0.1765 Teramoles per ppb of dry air
        self.add(
            'alphaa', 1./(0.1765 * 12.01),
            'Conversion factor of GtC to ppm', 'ppm GtC-1',
        )
        self.add(
            'mol2gt', 12.01e-15,
            'Conversion factor from molC to GtC', 'GtC molC-1',
        )
        self.add(
            'fbon', True,
            'When True, climate-carbon cycle feedback is activated', 'none', False,
        )
        self.add('m2lit', 1.e3, 'Conversion factor from m^3 to L', 'L m-3')
        self.add('aoc', 0.362e15, 'Area of ocean', 'm2')
        # Dickson et al. (2007, PICES Special Publication 3)
        # Chapter 5 - Physical and thermodynamic data
        # Table 2
        self.add('alk', 0.002400, 'Total alkalinity', 'mol/kg')
        self.add('bts', 0.000416, 'Boron concentration', 'mol/kg')
        # reference temperature: 13.5 degC, concentration units: mol/kg
        self.add('rhosw_ref', 1.02629, 'Reference sea water density', 'kg/L')

        # pkref and pkgrad
        # Reference values of the following equilibrium constants, and their
        # first derivative with respect to sea water temperature
        """
        k0 : -log10(K0) where K0 is the equilibrium constant for solubility
            of CO2 in sea water, [CO2*]/pCO2, at a reference temperature
            [CO2*] is the sum of [CO2(aq)] and [H2CO3(aq)],
            where brackets represent total concentrations in mol/kg
            pCO2 is the partial pressure of CO2 in atm,
            assumed to be the same as the fugacity of CO2
        k1 : -log10(K1) where K1 is the equilibrium constant for bicarbonate,
            [H^+][HCO3^-]/[CO2*], at a reference temperature
        k2 : -log10(K2) where K2 is the equilibrium constant for carbonate,
            [H^+][CO3^2-]/[HCO3^-], at a reference temperature
        kb : -log10(Kb) where Kb is the equilibrium constant for borate,
            [H^+][B(OH)4^-]/[B(OH)3], at a reference temperature
        kw : -log10(Kw) where Kw is the equilibrium constant for water,
            [H^+][OH^-], at a reference temperature
        """
        self.add(
            'pk0', [1.40645, 0.0133832],
            'K0 reference and temperature dependency', 'none',
        )
        self.add(
            'pk1', [5.95545, -0.0103683],
            'K1 reference and temperature dependency', 'none',
        )
        self.add(
            'pk2', [9.15411, -0.0167601],
            'K2 reference and temperature dependency', 'none',
        )
        self.add(
            'pkb', [8.73503, -0.0123654],
            'Kb reference and temperature dependency', 'none',
        )
        self.add(
            'pkw', [13.6875, -0.0428549],
            'Kw reference and temperature dependency', 'none',
        )
        # Hooss et al. (2001, https://doi.org/10.1007/s003820100170)
        self.add(
            'tau_ref', [236.5, 59.52, 12.17, 1.271],
            'Reference time constants', 'yr', False,
        )
        self.add(
            'amp_ref', [0.20, 0.24, 0.21, 0.25, 0.1],
            'Reference amplitudes', 'none', False,
        )
        # Box-model parameters for the reference IRM and preindustrial CO2
        # Impulse response model (IRM) for the airborne fraction is converted
        # into an equivalent box model with parameters hlk and etak
        self.add(
            'hlk', # hls, hl1, hl2, hl3
            [
                70.67364860121583,
                469.5764244975062,
                877.9485488217052,
                1126.0527277233425,
            ], 'Box-model layer depths', 'm', False,
        )
        self.add(
            'etak', # eta1, eta2, eta3
            [19.69858505590362, 7.771785495009128, 3.9335174388499183],
            'Exchange coefficients between adjacent layers', 'm yr-1', False,
        )
        self.add(
            'sigc_pi', 0., # value is set later
            'Preindustrial dissolved inorganic carbon', 'mol L-1', False,
        )
        self.add(
            'xibuf_pi', 0., # value is set later
            'Preindustrial Revelle buffer factor', 'none', False,
        )
        self.add(
            'hla', 0., # value is set later,
            'Atmospheric thickness in terms of excess carbon', 'm', False,
        )

class ModelOcean(ModelBase):
    def init_process(self, **kw):
        self.parms = ParmsOcean()
        self.parms.update(**kw)
        self.update_pi()
        self.msyst = self._msyst()

    def update_pi(self, cco2_pi=None):
        """
        Update the following cco2_pi-dependent parameters
        as well as cco2_pi itself if given:
        - pre-industrial DIC (sigc_pi)
        - pre-industrial buffer factor (xibuf_pi)
        - the atmospheric thickness in terms of excess carbon (hla)

        This function should be called when cco2_pi is initialized or changed

        hla is equal to hls * (ca - ca_pi) / (cs - cs_pi)
        or hls * (1 - amp[4]) / amp[4]
        The composit layer thickness (hl0 = hla + hls) is derived from
        hls / amp[4]
        """
        if cco2_pi is None:
            cco2_pi = self.parms.cco2_pi
        else:
            self.parms.cco2_pi = cco2_pi

        sigc = self.ppm2dic(cco2_pi)

        epsilon = 1e-7
        sigcm1 = self.ppm2dic((1.-epsilon) * cco2_pi)
        sigcp1 = self.ppm2dic((1.+epsilon) * cco2_pi)
        xibuf = sigc * 2. * epsilon / (sigcp1 - sigcm1)

        alphaa = self.parms.alphaa
        mol2gt = self.parms.mol2gt
        m2lit = self.parms.m2lit
        aoc = self.parms.aoc
        hla = xibuf * cco2_pi / alphaa / (mol2gt * m2lit * sigc * aoc)

        self.parms.sigc_pi = sigc
        self.parms.xibuf_pi = xibuf
        self.parms.hla = hla

    def get_chem_const(self, v, ta):
        """
        Return an equilibrium constant, where temperature dependency is
        parameterized by a linear regression.

        Parameters
        ----------
        v : str
            Element of pkref and pkgrad.

        ta : float
            Temperature anomaly in degC.

        Returns
        -------
        kx : float
            Equilibrium constant defined with mol kg-1 units.
        """
        def pkfi(x): return 10**(-x)

        pkref, pkgrad = getattr(self.parms, f'p{v}').tolist()
        tamin, tamax = -10., 20.
        if not self.parms.fbon:
            # Temperature feedback is not activated
            ta = 0.
        else:
            # Ensure temperatures to be within the assumed range
            ta = min(ta, tamax)
            ta = max(ta, tamin)

        kx = pkfi(pkref + pkgrad * ta)

        return kx

    def alk_newton(self, cin, coef, xinit, ta=0.):
        """
        Solve the equation of constant alkalinity by Newton method for a given
        amount of total carbon in the atmosphere and ocean mixed layer.

        Parameters
        ----------
        cin : float
            Total (preindustrial and excess) carbon in GtC in the combined
            atmosphere and ocean mixed layer. When coef is given by [1, 0],
            cin is treated as dissolved CO2 in mol/L.

        coef : float array
            [alphak, alphac]
            alphak = alphas / alphaa + alphac
            alphas: Henry's law constant; dissolved CO2 in mol/L to atmospheric
                CO2 concentration in ppm.
            alphaa: conversion factor of GtC to ppm.
            alphac: DIC in mol/L to carbon in GtC.

        xinit : float
            Initial value.

        ta : float, optional
            Temperature anomaly in degC, default 0.

        Returns
        -------
        sigc_ov_dco2 : float
            DIC in mol/L divided by dissolved CO2 in mol/L.

        h1 : float
            Hydrogen ion concentration in mol/L.
        """
        rhosw = self.parms.rhosw_ref
        ak1 = self.get_chem_const('k1', ta) * rhosw
        ak2 = self.get_chem_const('k2', ta) * rhosw
        akb = self.get_chem_const('kb', ta) * rhosw
        akw = self.get_chem_const('kw', ta) * (rhosw*rhosw)
        alk = self.parms.alk * rhosw
        bts = self.parms.bts * rhosw

        nmax = 10
        n = 0
        h1 = 2.*ak2/(np.sqrt(1.+4.*ak2/ak1*(xinit-1.))-1.)

        while 1:
            d1 = coef[0]*h1*h1 + coef[1]*ak1*(h1+ak2)
            d1p = 2.*coef[0]*h1 + coef[1]*ak1
            fp = cin*ak1*(d1-(h1+2.*ak2)*d1p)/(d1*d1) \
                - akb*bts/((akb+h1)*(akb+h1)) - akw/(h1*h1) - 1.
            f = cin*ak1*(h1+2.*ak2)/d1 + akb*bts/(akb+h1) + akw/h1 - h1 - alk
            h1 =  h1 - f/fp
            n += 1
            if np.abs(f/fp/h1) < 1.e-14 or n > nmax : break

        sigc_ov_dco2 = 1.+ak1/h1*(1.+ak2/h1)

        return sigc_ov_dco2, h1

    def ppm2dic(self, cco2, ta=0.):
        """
        Returns DIC for a given CO2 concentraion.

        Parameters
        ----------
        cco2 : float
            Atmospheric CO2 concentration in ppm.

        ta : float, optional
            Temperature anomaly in degC, default 0.

        Returns
        -------
        sigc : float
            DIC in mol/L
        """
        k0 = self.get_chem_const('k0', ta)
        alphas = 1.e6 / ( k0 * self.parms.rhosw_ref )
        dco2 = cco2/alphas
        sigc_ov_dco2, hplus= self.alk_newton(dco2, [1., 0.], 0.0021/dco2)
        sigc = dco2 * sigc_ov_dco2
        return sigc

    def partit(self, c0, hls, ta=0.):
        """
        Partition a given excess carbon in the combined atmosphere-ocean
        mixed layer.

        Parameters
        ----------
        c0 : float
            Excess carbon in GtC in the combined layer.

        hls : float
            Equivalent depth in meter of the combined layer.

        ta : float, optional
            Temperature anomaly in degC, default 0.

        Returns
        -------
        catm : float
            Excess carbon in GtC in the atmosphere.
        """
        if c0 == 0. : return 0.

        cco2_pi = self.parms.cco2_pi
        sigc_pi = self.parms.sigc_pi
        k0 = self.get_chem_const('k0', ta)
        alphaa = self.parms.alphaa
        alphac = self.parms.aoc*hls*self.parms.mol2gt*self.parms.m2lit
        alphas = 1.e6 / ( k0 * self.parms.rhosw_ref )

        alphak = alphas / alphaa + alphac
        cin = cco2_pi/alphaa + sigc_pi*alphac + c0
        sigc = sigc_pi + c0*(1.-0.925)/alphac
        dco2 = (cco2_pi + c0*0.925*alphaa)/alphas

        sigc_ov_dco2, hplus = \
            self.alk_newton(cin, [alphak, alphac], sigc/dco2, ta)
        catm = cin/(1.+alphaa*alphac/alphas*sigc_ov_dco2) - cco2_pi/alphaa

        return catm
    
    def partit_inv(self, catm, hls, ta=0.):
        """
        Inverse function of partit, i.e., return an excess carbon in the
        combined atmosphere-ocean mixed layer for a given excess carbon in the
        atmosphere.

        Parameters
        ----------
        catm : float
            Excess carbon in GtC in the atmosphere.

        hls : float
            Equivalent depth in meter of the combined layer.

        ta : float, optional
            Temperature anomaly in degC, default 0.

        Returns
        -------
        c0 : float
            Excess carbon in GtC in the combined layer.
        """
        if catm == 0. : return 0.

        cco2_pi = self.parms.cco2_pi
        k0 = self.get_chem_const('k0', ta)
        sigc_pi = self.parms.sigc_pi
        alphaa = self.parms.alphaa
        alphac = self.parms.aoc*hls*self.parms.mol2gt*self.parms.m2lit
        try:
            alphas = 1.e6 / ( k0 * self.parms.rhosw_ref )
        except:
            print(k0, ta)

        dco2 = (cco2_pi + catm*alphaa)/alphas
        sigc = sigc_pi + catm*(1.-0.925)/0.925/alphac
        sigc_ov_dco2, hplus = self.alk_newton(dco2, [1., 0.], sigc/dco2, ta)
        sigc = dco2 * sigc_ov_dco2
        c0 = catm + alphac*(sigc-sigc_pi)

        return c0

    def _msyst(self, **kw):
        """
        Construct a system matrix from hlk and etak
        where hlk=[hls or hl0, hl1, hl2, hl3],
        and etak=[eta1, eta2, eta3]
        hlk and etak are given by keyword arguments
        or refer to parms 
        """
        hlk = kw.get('hlk', self.parms.hlk)
        etak = kw.get('etak', self.parms.etak)

        a10 = etak[0] / hlk[0]
        a11 = etak[0] / hlk[1]
        a21 = etak[1] / hlk[1]
        a22 = etak[1] / hlk[2]
        a32 = etak[2] / hlk[2]
        a33 = etak[2] / hlk[3]

        return np.array([
            [a10, -a11, 0., 0.],
            [-a10, a11 + a21, -a22, 0.],
            [0., -a21, a22 + a32, -a33],
            [0., 0., -a32, a33],
        ])

    def _calib(self, **kw):
        """
        Calibrate hlk and etak for specific impulse response parameters
        by solving equations in terms of impulse responses
        analyzed through diagonalizing the system matrix

        Parameters
        ----------
        kw : dict, optional
            tau : 1-D array-like
                Time constants, default parms.tau_ref
            amp : 1-D array-like
                Amplitudes, default parms.amp_ref
            x0 : 1-D array-like, default [1000., 1000., 10., 10., 10.]
                Initial values for the solution of [hl2, hl3, eta1, eta2, eta3]

        Returns
        -------
        ret : dict
        """
        import scipy.optimize

        tau = kw.get('tau', self.parms.tau_ref)
        amp = kw.get('amp', self.parms.amp_ref)
        x0 = kw.get('x0', [1000., 1000., 10., 10., 10.])
        xtol = kw.get('xtol', 1e-10)
        outtol = 1e-10

        tau = np.array(tau)
        amp = np.array(amp)
        x0 = np.array(x0)

        hl0 = self.parms.hla / (1 - amp[4])

        sol = scipy.optimize.root(
            self._func_calib, x0, args=(amp, tau, hl0),
            method='hybr', options={'xtol': xtol},
        )
        if not sol.success:
            raise MCECalibError(sol.message)
        elif (sol.fun * sol.fun).sum() > outtol:
            raise MCECalibError('solution may not converged')
        elif np.any(sol.x < 0.):
            raise MCECalibError('negative solution found')

        hlk = np.hstack([hl0 * amp[4], 0., sol.x[:2]])
        # hlk[0] contains hls

        # airborne fraction after the shortest time-scale term dropped
        # i.e., the airborne fraction of the combined layer
        af = amp[0] / (1. - amp[4])
        # which relates hl0 / af = (hl1 + hl2 + hl3) / (1 - af)
        hlk[1] = hl0 * (1./af - 1.) - (hlk[2] + hlk[3])

        etak = sol.x[2:]

        return hlk, etak

    def _func_calib(self, x0, *args):
        amp, tau, hl0 = args
        ak = amp[:4][::-1] / (1. - amp[4])
        taui = 1. / tau[:3][::-1]

        hl = np.hstack([hl0, 0., x0[:2]])
        hl[1] = hl[0] * (1 / ak[3] - 1) - (hl[2] + hl[3])
        eta = x0[2:]

        msyst = self._msyst(hlk=hl, etak=eta)

        eigval, eigvec = np.linalg.eig(msyst)

        eigval_i = np.imag(eigval)
        epsm = 1e-7
        if np.any(eigval_i * eigval_i > epsm):
            raise MCECalibError('complex eigenvalues')
        
        eigval = np.real(eigval)
        eigvec = np.real(eigvec)

        idx = np.argsort(eigval * eigval)[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]

        epulse = np.hstack([1., np.zeros(3)])
        try:
            rk = np.linalg.solve(eigvec, epulse)
        except LinAlgError as e:
            raise MCECalibError('linalg error {}'.format(e))
        except:
            raise MCECalibError('linalg other error')

        ak_r = rk * eigvec[0, :] / epulse[0]

        return np.hstack([
            ak_r[:2] / ak[:2] - 1.,
            eigval[:3] / taui[:3] - 1.,
        ])


class ParmsLand(ParmsBase):
    """
    Land carbon cycle parameters
    """
    def __init__(self):
        self.add(
            'alphaa', 1./(0.1765 * 12.01),
            'Conversion factor of GtC to ppm', 'ppm GtC-1',
        )
        self.add(
            'cco2_pi', 280.,
            'Preindustrial CO2 concentration (fixed)', 'ppm',
            # slightly different from that in the ocean module
            # value is fixed
        )
        self.add('cco2_b', 31., 'CO2 concentration at which NPP is zero', 'ppm')
        self.add('fnpp0', 60., 'Base NPP', 'GtC yr-1')
        self.add(
            'beta', 0.43,
            'Control parameter of CO2 fertilization effect', 'none', False,
        )
        """
        The fertilization factor is formulated with a sigmoid curve with regard
        to CO2 concentration, as described in Meinshausen et al.
        (2011, https://doi.org/10.5194/acp-11-1417-2011).
        This implementation is connected to a conventional logarithmic formula
        beta = 1 + beta_l ln(C/C0)
        such that the sigmoid and logarithmic curves are equal in terms of an
        increase ratio at 680 ppm relative to 340 ppm, and beta_l is used
        as the control parameter.
        """

        self.add('aparm', 0., 'Fertilization parameter A', 'ppm-1', False)
        self.add('bparm', 0., 'Fertilization parameter B', 'ppm-1', False)
        # aparm and bparm are set later

        self.add(
            'fbon', True,
            'When True, climate-carbon cycle feedback is activated', 'none', False,
        )
        self.add(
            'fb_alpha', 0.87,
            'Control parameter of reducing overturning times in terms of magnitude',
            'none', False,
        )
        self.add(
            'tsa_ref2', 3.5,
            'Control parameter of reducing overturning times in terms of sensitivity to temperature',
            'degC', False,
        )
        """
        fb_alpha is an asymptotic minimum value of a logistic curve
        to define an adjustment coefficient to reduce overturning times
        as a function of surface warming,
        and tsa_ref2 is the temperature at which the curve has the maximum gradient
        """

        tau = np.array([120./55., 1./0.35, 20., 100.])
        amp = np.array([-0.0653145*11, 0.100302*7, 0., 0.])
        amp[3] = (1.-(amp[0]*tau[0]+amp[1]*tau[1])
                    +tau[2]*(amp[0]+amp[1]) ) / (tau[3]-tau[2])
        amp[2] = -(amp[0]+amp[1]+amp[3])
        self.add('tau', tau, 'Overturning times', 'yr', False)
        self.add('amp', amp, 'Response amplitudes', 'yr-1', False)
        """
        Base values of impulse response parameters based on Joos et al.
        (1996, https://doi.org/10.3402/tellusb.v48i3.15921)
        2.2, 2.9, 20, and 100 years for overturning times, and
        -0.71846, 0.70211, 0.013414, and 0.0029323 for response amplitudes
        """

    def update_deriv(self):
        beta = self.beta
        cco2_b = self.cco2_b
        cco2_pi = self.cco2_pi
        c1 = 340.
        c2 = c1 + 340.
        r = (1.+beta*np.log(c2/cco2_pi)) / (1.+beta*np.log(c1/cco2_pi))
        bparm = (1./(c1-cco2_b) - r/(c2-cco2_b)) / (r-1.)
        aparm = (1.+bparm*(cco2_pi - cco2_b))/(cco2_pi - cco2_b)
        self.bparm = bparm
        self.aparm = aparm

        # att0 = amp[0]*tau[0]*tau[0] + amp[1]*tau[1]*tau[1]


class ModelLand(ModelBase):
    """
    Impulse response model to calculate the decay of land carbon
    accumulated by the CO2 fertilization effect in four carbon pools,
    representing detritus, ground vegetation, wood, and soil organic carbon
    """
    def init_process(self, **kw):
        self.parms = ParmsLand()
        self.parms.update(**kw)
        self.parms.update_deriv()

    def fertmm(self, catm):
        """
        Returns NPP in GtC/y adjusted with a sigmoidal growth function
        based on Equation A19 in Meinshausen et al. (2011).

        Parameters
        ----------
        catm : float
            Excess carbon in the atmosphere in GtC.

        Returns
        -------
        npp : float
            NPP in GtC/y.
        """
        parms = self.parms
        a = parms.aparm
        b = parms.bparm
        c = parms.cco2_pi + catm * parms.alphaa - parms.cco2_b
        npp = parms.fnpp0*a*c/(1.+b*c)
        return npp

    def fertmm_deriv(self, catm):
        """
        Returns first derivative of NPP.

        Parameters
        ----------
        catm : float
            Excess carbon in the atmosphere in GtC.

        Returns
        -------
        npp_deriv : float
            First derivative of NPP.
        """
        parms = self.parms
        a = parms.aparm
        b = parms.bparm
        c = parms.cco2_pi + catm * parms.alphaa - parms.cco2_b
        npp_deriv = parms.fnpp0 * a * parms.alphaa/(1.+b*c)**2
        return npp_deriv

    def tempfb(self, tsa):
        """
        Returns an adjustment coefficient to reduce overturning times,
        implemented with a logistic curve with respect to surface warming.

        Parameters
        ----------
        tsa : float
            Surface temperature anomaly.

        Returns
        -------
        coef : float
            Adjustment coefficient to reduce overturning times.
        """
        alpha = self.parms.fb_alpha
        tsa_ref2 = self.parms.tsa_ref2

        x0 = 0.1
        tsa_ref = 0.
        a = (1.-alpha)/(1.-x0)
        r = np.log((1.-x0)/x0)/(tsa_ref2-tsa_ref)
        x = 1./(1.+(1./x0-1.)*np.exp(-r*(tsa-tsa_ref)))
        coef = 1.-a*(x-x0)
        return coef

    def get_tau(self, tsa):
        """
        Returns overturning times adjusted by surface warming.

        Parameters
        ----------
        tsa : float
            Surface temperature anomaly.

        Returns
        -------
        tau : array
            Overturning times.
        """
        parms = self.parms
        tamin, tamax = -10., 20.
        tau = np.array(parms.tau)
        if parms.fbon:
            tsa = min(tsa, tamax)
            tsa = max(tsa, tamin)
            tau[2:] *= self.tempfb(tsa)

        return tau

