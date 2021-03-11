"""
API for computing carbon cycle, developed based on
Hooss et al. (2001, https://doi.org/10.1007/s003820100170) and
Joos et al. (1996, https://doi.org/10.3402/tellusb.v48i3.15921).
"""

import numpy as np
from mce.core import ModelBase

class ModelOcean(ModelBase):
    def init_process(self, **kwargs):
        """
        Define and update the following parameters.

        cco2_pi : preindustrial CO2 concentration in ppm
        alphaa : conversion factor of GtC to ppm
        mol2gt : conversion factor from molC to GtC
        m2lit : conversion factor from m^3 to L
        aoc : area of ocean in m^2
        alk : total alkalinity in mol/kg 
        bts : Boron concentration in mol/kg
        rhosw_ref : reference sea water density in kg/L

        Reference values of the following equilibrium constants, and their
        first derivative with respect to sea water temperature, stored in
        pkref and pkgrad
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

        fbon : when True (default), climate-carbon cycle feedback is activated

        Impulse response model (IRM) for the airborne fraction is converted
        into an equivalent box model with the following parameters
        hls, hl1, hl2, hl3 : box-model layer depth in meters
        eta1, eta2, eta3 : exchange coefficients between layers in m/yr

        sigc_pi : preindustrial dissolved inorganic carbon (DIC) in mol/L

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments to set module parameters
        """
        self.parms.update(
            cco2_pi = 278.,

            # Prather et al. (2012, https://doi.org/10.1029/2012GL051440)
            # 0.1765 Teramoles per ppb of dry air
            alphaa = 1./(0.1765 * 12.01),
            mol2gt = 12.01e-15,

            fbon = True,
            m2lit = 1.e3,
            aoc = 0.362e15,

            # Dickson et al. (2007, PICES Special Publication 3)
            # Chapter 5 - Physical and thermodynamic data
            # Table 2
            alk = 0.002400,
            bts = 0.000416,
            # reference temperature: 13.5 degC, concentration units: mol/kg
            rhosw_ref = 1.02629,
            pkref = {
                'k0': 1.40645,
                'k1': 5.95545,
                'k2': 9.15411,
                'kb': 8.73503,
                'kw': 13.6875,
            },
            pkgrad = {
                'k0': 0.0133832,
                'k1': -0.0103683,
                'k2': -0.0167601,
                'kb': -0.0123654,
                'kw': -0.0428549,
            },
            # Reference time constants are adopted from Hooss et al.
            # (2001, https://doi.org/10.1007/s003820100170),
            # and reference amplitudes are set empirically
            tau_ref = np.array([236.5, 59.52, 12.17, 1.271]),
            amp_ref = np.array([0.20, 0.24, 0.21, 0.25, 0.1]),
            # Box-model parameters for the reference IRM and preindustrial CO2
            eta1 = 19.69858505590362,
            eta2 = 7.771785495009128,
            eta3 = 3.9335174388499183,
            hl1 = 469.5764244975062,
            hl2 = 877.9485488217052,
            hl3 = 1126.0527277233425,
            hls = 70.67364860121583,
        )

        self.parms_update(**kwargs)

        cco2_pi = self.parms['cco2_pi']
        self.parms['sigc_pi'] = self.ppm2dic(cco2_pi)

        self.msyst = self._get_msyst()

    def _pkfi(self, x): return 10**(-x)

    def get_chem_const(self, v, ta):
        """
        Return an equilibrium constant, where temperature dependency is
        parameterized by a linear regression.

        Parameters
        ----------
        v : str
            Identifier defined as the keys of pkref and pkgrad.

        ta : float
            Temperature anomaly in degC.

        Returns
        -------
        kx : float
            Equilibrium constant defined with mol/kg units.
        """
        pkref = self.parms['pkref']
        pkgrad = self.parms['pkgrad']
        tamin, tamax = -10., 20.
        if not self.parms['fbon']:
            # Temperature feedback is not activated
            ta = 0.
        else:
            # Ensure temperatures to be within the assumed range
            ta = min(ta, tamax)
            ta = max(ta, tamin)

        kx = self._pkfi(pkref[v] + pkgrad[v]*ta)

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
        rhosw = self.parms['rhosw_ref']
        ak1 = self.get_chem_const('k1', ta) * rhosw
        ak2 = self.get_chem_const('k2', ta) * rhosw
        akb = self.get_chem_const('kb', ta) * rhosw
        akw = self.get_chem_const('kw', ta) * (rhosw*rhosw)
        alk = self.parms['alk'] * rhosw
        bts = self.parms['bts'] * rhosw

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
        alphas = 1.e6 / ( k0 * self.parms['rhosw_ref'] )
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

        cco2_pi = self.parms['cco2_pi']
        sigc_pi = self.parms['sigc_pi']
        k0 = self.get_chem_const('k0', ta)
        alphaa = self.parms['alphaa']
        alphac = self.parms['aoc']*hls*self.parms['mol2gt']*self.parms['m2lit']
        alphas = 1.e6 / ( k0 * self.parms['rhosw_ref'] )

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

        cco2_pi = self.parms['cco2_pi']
        k0 = self.get_chem_const('k0', ta)
        sigc_pi = self.parms['sigc_pi']
        alphaa = self.parms['alphaa']
        alphac = self.parms['aoc']*hls*self.parms['mol2gt']*self.parms['m2lit']
        try:
            alphas = 1.e6 / ( k0 * self.parms['rhosw_ref'] )
        except:
            print(k0, ta)

        dco2 = (cco2_pi + catm*alphaa)/alphas
        sigc = sigc_pi + catm*(1.-0.925)/0.925/alphac
        sigc_ov_dco2, hplus = self.alk_newton(dco2, [1., 0.], sigc/dco2, ta)
        sigc = dco2 * sigc_ov_dco2
        c0 = catm + alphac*(sigc-sigc_pi)

        return c0

    def _get_msyst(self, hl0=None):
        """
        Construct a system matrix
        """
        if hl0 is None:
            a10 = self.parms['eta1'] / self.parms['hls']
        else:
            a10 = self.parms['eta1'] / hl0

        a11 = self.parms['eta1'] / self.parms['hl1']
        a21 = self.parms['eta2'] / self.parms['hl1']
        a22 = self.parms['eta2'] / self.parms['hl2']
        a32 = self.parms['eta3'] / self.parms['hl2']
        a33 = self.parms['eta3'] / self.parms['hl3']

        msyst = np.zeros((4, 4))
        msyst[0, 0:2] = [a10, -a11]
        msyst[1, 0:3] = [-a10, a11+a21, -a22]
        msyst[2, 1:4] = [-a21, a22+a32, -a33]
        msyst[3, 2:4] = [-a32, a33]

        return msyst


class ModelLand(ModelBase):
    def init_process(self, **kwargs):
        """
        Define and update the following parameters.

        alphaa : conversion factor of GtC to ppm
        cco2_pi : preindustrial CO2 concentration in ppm
        cco2_b : CO2 concentration at which NPP is zero
        fnpp0 : base NPP in GtC/y
        beta : control parameter of CO2 fertilization effect
        fbon : when True (default), climate-carbon cycle feedback is activated
        fb_alpha : asymptotic minimum value of a logistic curve to define
            an adjustment coefficient to reduce overturning times as a function
            of surface warming
        tsa_ref2 : temperature at which the curve has the maximum gradient

        Impulse response model (IRM) calculates the decay of land carbon
        accumulated by the CO2 fertilization effect in four carbon pools,
        representing detritus, ground vegetation, wood, and soil organic carbon
        with parameters
        tau : overturning times in year
        amp : response amplitudes in year^(-1)

        The fertilization factor is formulated with a sigmoid curve with regard
        to CO2 concentration, as described in Meinshausen et al.
        (2011, https://doi.org/10.5194/acp-11-1417-2011).
        This implementation is connected to a conventional logarithmic formula
        beta = 1 + beta_l ln(C/C0)
        such that the sigmoid and logarithmic curves are equal in terms of an
        increase ratio at 680 ppm relative to 340 ppm, and beta_l is used
        as a control parameter.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments to set module parameters
        """
        self.parms.update(
            alphaa = 1./(0.1765 * 12.01),
            cco2_pi = 280.,
            # slightly different from that used in the ocean module
            fnpp0 = 60.,
            beta = 0.43,
            cco2_b = 31.,
            fbon = True,
            fb_alpha = 0.87,
            tsa_ref2 = 3.5,
        )

        self.parms_update(**kwargs)

        beta = self.parms['beta']
        cco2_b = self.parms['cco2_b']
        cco2_pi = self.parms['cco2_pi']
        c1 = 340.
        c2 = c1 + 340.
        r = (1.+beta*np.log(c2/cco2_pi)) / (1.+beta*np.log(c1/cco2_pi))
        bparm = (1./(c1-cco2_b) - r/(c2-cco2_b)) / (r-1.)
        aparm = (1.+bparm*(cco2_pi - cco2_b))/(cco2_pi - cco2_b)
        self.parms['bparm'] = bparm
        self.parms['aparm'] = aparm

        # Base values of IRM parameters are fixed to
        # 2.2, 2.9, 20, and 100 years for overturning times, and
        # -0.71846, 0.70211, 0.013414, and 0.0029323 for response amplitudes
        # based on Joos et al.
        # (1996, https://doi.org/10.3402/tellusb.v48i3.15921)
        tau = np.array([120./55., 1./0.35, 20., 100.])
        amp = np.array([-0.0653145*11, 0.100302*7, 0., 0.])
        amp[3] = (1.-(amp[0]*tau[0]+amp[1]*tau[1])
                    +tau[2]*(amp[0]+amp[1]) ) / (tau[3]-tau[2])
        amp[2] = -(amp[0]+amp[1]+amp[3])
        self.parms['tau'] = tau
        self.parms['amp'] = amp
        self.parms['att0'] = amp[0]*tau[0]*tau[0] + amp[1]*tau[1]*tau[1]

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
        a = parms['aparm']
        b = parms['bparm']
        c = parms['cco2_pi'] + catm * parms['alphaa'] - parms['cco2_b']
        npp = parms['fnpp0']*a*c/(1.+b*c)
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
        a = parms['aparm']
        b = parms['bparm']
        c = parms['cco2_pi'] + catm * parms['alphaa'] - parms['cco2_b']
        npp_deriv = parms['fnpp0'] * a * parms['alphaa']/(1.+b*c)**2
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
        alpha = self.parms['fb_alpha']
        tsa_ref2 = self.parms['tsa_ref2']

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
        tau = np.array(parms['tau'])
        if parms['fbon']:
            tsa = min(tsa, tamax)
            tsa = max(tsa, tamin)
            tau[2:] *= self.tempfb(tsa)

        return tau

