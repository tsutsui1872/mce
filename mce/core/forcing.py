"""
API for computing effective radiative forcing (ERF) of CO2 and others.
"""

from collections import namedtuple
import numpy as np
from . import ParmsBase
from . import ModelBase

Molecule = namedtuple(
    # Properties of each greenhouse gas
    'Mulecule', [
        'name',
        'category',
        'formula',
        'weight', # g mol-1
        'efficiency', # W m-2 ppb-1
        'lifetime', # year
    ],
)

ODS = namedtuple(
    # Specific properties of each ozone depleting substance
    'ODS', [
        'name',
        'fracrel', # fractional stratospheric release for ozone depletion
        'cl_atoms', # number of chlorine atoms
        'br_atoms', # number of bromine atoms
    ],
)

O3_coeff = namedtuple('O3_coeff', ['mean', 'u90'])


class ParmsCO2(ParmsBase):
    """
    Define alpha, beta, and ccref, used in MCE CO2 scheme.
    Default values can be updated by keyword arguments.
    """
    def __init__(self, **kw):
        self.add('alpha', 4.617, 'CO2 scaling factor', 'W m-2', False)
        # adjusted CMIP5 mean for IRM-3
        # 4.701, # CMIP5 mean
        # 4.561, # adjusted CMIP5 mean for IRM-2
        self.add('beta', 1.068, 'CO2 amplification factor', 'none', False)
        # CMIP5 mean for IRM-3
        # 1.062, # CMIP5 mean for IRM-2
        self.add('ccref', 278., 'Base concentration of CO2', 'ppm', False)

        self.update(**kw)

class RfCO2(ModelBase):
    """
    MCE CO2 scheme, assuming that forcing is defined as effective radiative forcing.

    Parameters are accessed through attribute 'parms',
    which is an instance that contains parameter values as ParmsCO2 class variables,
    shared among all instances of the class.
    Referencing and updating parameter values are implemented using property(),
    and the description of defined parameters is presented by print().
    """
    def init_process(self, *args, **kw):
        self.parms = ParmsCO2()
        self.parms.update(**kw)

    def c2erf(self, *args):
        """
        Apply MCE forcing scheme to CO2 concentrations in ppm.
        Input concentrations, given by multiple arguments, to be passed to numpy.hstack.

        Returns
        -------
            Effective radiative forcing in W/m2
        """
        ccref = self.parms.ccref
        erf = self.xl2erf(np.log(np.hstack(args)/ccref))
        return erf

    def x2erf(self, *args):
        """
        Apply MCE forcing scheme to C/Cref, where C is a CO2 concentration,
        and Cref is its reference value in a pre-industrial period.
        Input values, given by multiple arguments, to be passed to numpy.hstack.

        Returns
        -------
            Effective radiative forcing in W/m2
        """
        erf = self.xl2erf(np.log(np.hstack(args)))
        return erf

    def xl2erf(self, *args):
        """
        Apply MCE forcing scheme to log(x), where x is a ratio of CO2 concentration
        to its reference value in a pre-industrial period.
        Input values, given by multiple arguments, to be passed to numpy.hstack.

        Returns
        -------
            Effective radiative forcing in W/m2
        """
        alpha = self.parms.alpha
        beta = self.parms.beta

        xl = np.hstack(args)
        erf = alpha * xl

        if beta != 1:
            f2x = alpha * np.log(2.)
            ix4 = erf > 2*f2x
            ix2 = (erf > f2x) & (erf <= 2*f2x)
            # erf[ix4] = beta*erf[ix4]
            # modified 2020-12-11
            erf[ix4] = (4.*beta-3.)*erf[ix4] - 6.*f2x*(beta-1.)
            erf[ix2] = (
                (beta-1) * (erf[ix2]-2*f2x) * (2/f2x*erf[ix2]-1)
                + beta*erf[ix2]
            )

        return erf.item() if len(erf)==1 else erf

class ParmsEtminan(ParmsBase):
    """
    Define base concentrations and coefficients for Etminan scheme.
    Default values can be updated by keyword arguments.
    """
    def __init__(self, **kw):
        # WGI AR5 Table AII.1
        self.add('C0', 278., 'Base concentration of CO2', 'ppm')
        self.add('N0', 270., 'Base concentration of N2O', 'ppb')
        self.add('M0', 722., 'Base concentration of CH4', 'ppb')
        # Etminan et al. (2016) Table 1
        self.add('a1', -2.4e-7, 'CO2 coefficient a1', 'W m-2 ppm-2')
        self.add('b1', 7.2e-4, 'CO2 coefficient b1', 'W m-2 ppm-1')
        self.add('c1', -2.1e-4, 'CO2 coefficient c1', 'W m-2 ppb-1')
        self.add('a2', -8.e-6, 'N2O coefficient a2', 'W m-2 ppm-1')
        self.add('b2', 4.2e-6, 'N2O coefficient b2', 'W m-2 ppb-1')
        self.add('c2', -4.9e-6, 'N2O coefficient c2', 'W m-2 ppb-1')
        self.add('a3', -1.3e-6, 'CH4 coefficient a3', 'W m-2 ppb-1')
        self.add('b3', -8.2e-6, 'CH4 coefficient b3', 'W m-2 ppb-1')

        self.update(**kw)

class ParmsAR6GHG(ParmsBase):
    """
    Define base concentrations and coefficients given by WGI AR6 Table 7.SM.1.
    Default values can be updated by keyword arguments.
    """
    def __init__(self, **kw):
        self.add('a1', -2.4785e-7, 'CO2 coefficient a1', 'W m-2 ppm-2')
        self.add('b1', 7.5906e-4, 'CO2 coefficient b1', 'W m-2 ppm-1')
        self.add('c1', -2.1492e-3, 'CO2 coefficient c1', 'W m-2 ppb-1/2')
        self.add('d1', 5.2488, 'CO2 coefficient d1', 'W m-2')
        self.add('C0', 277.15, 'Base concentration of CO2', 'ppm')
        self.add('a2', -3.4197e-4, 'N2O coefficient a2', 'W m-2 ppm-1')
        self.add('b2', 2.5455e-4, 'N2O coefficient b2', 'W m-2 ppb-1')
        self.add('c2', -2.4357e-4, 'N2O coefficient c2', 'W m-2 ppb-1')
        self.add('d2', 0.12173, 'N2O coefficient d2', 'W m-2 ppb-1/2')
        self.add('N0', 273.87, 'Base concentration of N2O', 'ppb')
        self.add('a3', -8.9603e-5, 'CH4 coefficient a3', 'W m-2 ppb-1')
        self.add('b3', -1.2462e-4, 'CH4 coefficient b3', 'W m-2 ppb-1')
        self.add('d3', 0.045194, 'CH4 coefficient c3', 'W m-2 ppb-1/2')
        self.add('M0', 731.41, 'Base concentration of CH4', 'ppb')

        # AR6 WG1 7.SM.1.3.1
        self.add('C0_1750', 278.3, 'Base concentration of CO2 in 1750', 'ppm')
        self.add('N0_1750', 270.1, 'Base concentration of N2O in 1750', 'ppb')
        self.add('M0_1750', 729.2, 'Base concentration of CH4 in 1750', 'ppb')

        self.add('CO2_adj', 0.05, 'Tropospheric adjustment of CO2', 'none')
        self.add('N2O_adj', 0.07, 'Tropospheric adjustment of N2O', 'none')
        self.add('CH4_adj', -0.14, 'Tropospheric adjustment of CH4', 'none')
        self.add('CFC-12_adj', 0.12, 'Tropospheric adjustment of CFC-12', 'none')
        self.add('CFC-11_adj', 0.13, 'Tropospheric adjustment of CFC-11', 'none')

        self.add(
            'fb_o3_forcing', -0.037,
            'Factor of temperature feedback to ozone forcing', 'W m-2 degC-1', False,
        )

        self.update(**kw)


class RfAll(RfCO2):
    """
    Multi-gas scheme.

    Parameters are accessed through attribute 'parms', 'parms_etminan', and 'parms_ar6_ghg',
    each of which is an instance that contains parameter values as class variables
    of ParmsCO2, ParmsEtminan, and ParmsAR6GHG, respectively.
    The parameter values contained in each class are shared among all instances of the class.
    Referencing and updating the values are implemented using property(),
    and the description of defined parameters is presented by print().
    """
    def init_process(self, *args, **kw):
        super().init_process()
        self.parms.update(**{
            k: kw[k]
            for k in self.parms.names() if k in kw
        })
        kw_etminan = kw.get('etminan', {})
        kw_ar6 = kw.get('ar6', {})

        self.parms_etminan = ParmsEtminan(**kw_etminan)
        self.parms_ar6_ghg = ParmsAR6GHG(**kw_ar6)

        # molecule weights:
        # https://www.lenntech.com/calculators/molecular/molecular-weight-calculator.htm
        # radiative efficiency:
        #   Chapter-7-main src/ar6/constants/gases.py
        #   commented-out efficiency from AR5 WG1 Appendiex 8.A
        # lifetime:
        #   Table 7.SM.7 of AR6 WG1
        #   slightly changed for CH2Cl2 and CHCl3 as in the Chapter 7 source (ar6/constants/gases.py)
        self.ghgs = {
            'CO2': Molecule('carbon dioxide', 'CO2', 'CO2', 44.01, None, None),
            'CH4': Molecule('methane', 'CH4', 'CH4', 16.04, None, 11.8),
            'N2O': Molecule('nitrous oxide', 'N2O', 'N2O', 44.01, None, 109.),
            # Chlorofluorocarbons
            'CFC-12': Molecule('CFC-12', 'Montreal Gases', 'CCl2F2', 120.91, 0.31998, 102.), # 0.32
            'CFC-11': Molecule('CFC-11', 'Montreal Gases', 'CCl3F', 137.37, 0.25941, 52.), # 0.26
            'CFC-113': Molecule('CFC-113', 'Montreal Gases', 'CCl2FCClF2', 187.38, 0.30142, 93.), # 0.30
            'CFC-114': Molecule('CFC-114', 'Montreal Gases', 'CClF2CClF2', 170.92, 0.31433, 189.), # 0.31
            'CFC-115': Molecule('CFC-115', 'Montreal Gases', 'CClF2CF3', 154.47, 0.24625, 540.), # 0.20
            'CFC-13': Molecule('CFC-13', 'Montreal Gases', 'CClF3', 104.46, 0.27752, 640.),
            'CFC-112': Molecule('CFC-112', 'Montreal Gases', 'CCl2FCCl2F', 203.83, 0.28192, 63.6),
            'CFC-112a': Molecule('CFC-112a', 'Montreal Gases', 'CCl3CClF2', 203.83, 0.24564, 52.),
            'CFC-113a': Molecule('CFC-113a', 'Montreal Gases', 'CCl3CF3', 187.38, 0.24094, 55.),
            'CFC-114a': Molecule('CFC-114a', 'Montreal Gases', 'CCl2FCF3', 170.92, 0.29747, 105.),
            # Hydrofluorochlorocarbons
            'HCFC-22': Molecule('HCFC-22', 'Montreal Gases', 'CHClF2', 86.47, 0.21385, 11.9), # 0.21
            'HCFC-141b': Molecule('HCFC-141b', 'Montreal Gases', 'CH3CCl2F', 116.95, 0.16065, 9.4), # 0.1
            'HCFC-142b': Molecule('HCFC-142b', 'Montreal Gases', 'CH3CClF2', 100.49, 0.19329, 18.), # 0.19
            'HCFC-133a': Molecule('HCFC-133a', 'Montreal Gases', 'CH2ClCF3', 118.49, 0.14995, 4.6),
            'HCFC-31': Molecule('HCFC-31', 'Montreal Gases', 'CH2ClF', 68.48, 0.068, 1.2),
            'HCFC-124': Molecule('HCFC-124', 'Montreal Gases', 'CHClFCF3', 136.48, 0.20721, 5.9),
            # Chlorocarbons and Hydrochlorocarbons
            'CH3CCl3': Molecule('CH3CCl3', 'Montreal Gases', 'CH3CCl3', 133.40, 0.06454, 5.), # 0.07
            'CCl4': Molecule('CCl4', 'Montreal Gases', 'CCl4', 153.82, 0.16616, 32.), # 0.17
            'CH3Cl': Molecule('CH3Cl', 'Montreal Gases', 'CH3Cl', 50.49, 0.00466, 0.9), # 0.01
            'CH2Cl2': Molecule('CH2Cl2', 'Montreal Gases', 'CH2Cl2', 84.93, 0.02882, 0.4932), # 0.03
            'CHCl3': Molecule('CHCl3', 'Montreal Gases', 'CHCl3', 119.38, 0.07357, 0.5014), # 0.08
            # Bromocarbons, hydrobromocarbons and halons
            'CH3Br': Molecule('CH3Br', 'Montreal Gases', 'CH3Br', 94.94, 0.00432, 0.8), # 0.004
            'Halon-1202': Molecule('Halon-1202', 'Montreal Gases', 'CBr2F2', 209.82, 0.272, 2.5), # 0.27
            'Halon-1211': Molecule('Halon-1211', 'Montreal Gases', 'CBrClF2', 165.36, 0.30014, 16.), # 0.29
            'Halon-1301': Molecule('Halon-1301', 'Montreal Gases', 'CBrF3', 148.91, 0.29943, 72.), # 0.30
            'Halon-2402': Molecule('Halon-2402', 'Montreal Gases', 'CBrF2CBrF2', 259.82, 0.31169, 28.), # 0.31
            # Hydrofluorocarbons
            'HFC-134a': Molecule('HFC-134a', 'F-Gases', 'CH2FCF3', 102.03, 0.16714, 14.), # 0.16
            'HFC-23': Molecule('HFC-23', 'F-Gases', 'CHF3', 70.01, 0.19111, 228.), # 0.18
            'HFC-32': Molecule('HFC-32', 'F-Gases', 'CH2F2', 52.02, 0.11144, 5.4), # 0.11
            'HFC-125': Molecule('HFC-125', 'F-Gases', 'CHF2CF3', 120.02, 0.23378, 30.), # 0.23
            'HFC-143a': Molecule('HFC-143a', 'F-Gases', 'CH3CF3', 84.04, 0.168, 51.), # 0.16
            'HFC-152a': Molecule('HFC-152a', 'F-Gases', 'CH3CHF2', 66.05, 0.10174, 1.6), # 0.10
            'HFC-227ea': Molecule('HFC-227ea', 'F-Gases', 'CF3CHFCF3', 170.03, 0.27325, 36.), # 0.26
            'HFC-236fa': Molecule('HFC-236fa', 'F-Gases', 'CF3CH2CF3', 152.04, 0.25069, 213.), # 0.24
            'HFC-245fa': Molecule('HFC-245fa', 'F-Gases', 'CHF2CH2CF3', 134.05, 0.24498, 7.9), # 0.24
            'HFC-365mfc': Molecule('HFC-365mfc', 'F-Gases', 'CH3CF2CH2CF3', 148.07, 0.22813, 8.9), # 0.22
            'HFC-43-10mee': Molecule('HFC-43-10mee', 'F-Gases', 'CF3CHFCHFCF2CF3', 252.05, 0.35731, 17.), # 0.42
            # Fully fluorinated species
            'NF3': Molecule('NF3', 'F-Gases', 'NF3', 71.00, 0.20448, 569.), # 0.20
            'SF6': Molecule('SF6', 'F-Gases', 'SF6', 146.06, 0.56657, 3200.), # 0.57
            'SO2F2': Molecule('SO2F2', 'F-Gases', 'SO2F2', 102.06, 0.21074, 36.), # 0.20
            'CF4': Molecule('CF4', 'F-Gases', 'CF4', 88.00, 0.09859, 50000.), # 0.09
            'C2F6': Molecule('C2F6', 'F-Gases', 'C2F6', 138.01, 0.26105, 10000.), # 0.25
            'C3F8': Molecule('C3F8', 'F-Gases', 'C3F8', 188.02, 0.26999, 2600.), # 0.28
            'n-C4F10': Molecule('n-C4F10', 'F-Gases', 'n-C4F10', 238.03, 0.36874, 2600.), # 0.36
            'n-C5F12': Molecule('n-C5F12', 'F-Gases', 'n-C5F12', 288.03, 0.4076, 4100.), # 0.41
            'n-C6F14': Molecule('n-C6F14', 'F-Gases', 'n-C6F14', 338.04, 0.44888, 3100.), # 0.44
            'C7F16': Molecule('C7F16', 'F-Gases', 'C7F16', 388.05, 0.50312, 3000.), # maybe n-C7F16, 0.50
            'C8F18': Molecule('C8F18', 'F-Gases', 'C8F18', 438.06, 0.55787, 3000.), # maybe n-C8F18, 0.55
            # Others
            'i-C6F14': Molecule('i-C6F14', 'F-Gases', 'i-C6F14', 338.04, 0.44888, 3100.), # assume the same as n-C6F14
            'c-C4F8': Molecule('c-C4F8', 'F-Gases', 'c-C4F8', 200.03, 0.31392, 3200.), # use AR5
        }
        # Ozone depleting substances parameters:
        # fractional stratospheric release for ozone depletion,
        # number of chlorine atoms,
        # and number of bromine atoms
        # Source: Chapter-7-main src/ar6/constants/gases.py
        self.ods = {
            'CCl4': ODS('CCl4', 0.56, 4, 0),
            'CFC-11': ODS('CFC-11', 0.47, 3, 0),
            'CFC-113': ODS('CFC-113', 0.29, 3, 0),
            'CFC-114': ODS('CFC-114', 0.12, 2, 0),
            'CFC-115': ODS('CFC-115', 0.04, 1, 0),
            'CFC-12': ODS('CFC-12', 0.23, 2, 0),
            'CH2Cl2': ODS('CH2Cl2', 0.0, 2, 0),
            'CH3Br': ODS('CH3Br', 0.6, 0, 1),
            'CH3CCl3': ODS('CH3CCl3', 0.67, 3, 0),
            'CH3Cl': ODS('CH3Cl', 0.44, 1, 0),
            'CHCl3': ODS('CHCl3', 0.0, 3, 0),
            'HCFC-141b': ODS('HCFC-141b', 0.34, 2, 0),
            'HCFC-142b': ODS('HCFC-142b', 0.17, 1, 0),
            'HCFC-22': ODS('HCFC-22', 0.13, 1, 0),
            # 'Halon-1202': Halogen('Halon-1202', 0.62, 0, 2), # not used, but defined in FaIR-1.6.2
            'Halon-1211': ODS('Halon-1211', 0.62, 0, 1), # maybe wrong
            # 'Halon-1211': ODS('Halon-1211', 0.62, 1, 1), # correct
            'Halon-1301': ODS('Halon-1301', 0.28, 0, 1),
            'Halon-2402': ODS('Halon-2402', 0.65, 0, 2),
        }
        # Chapter-7 data_input/tunings/cmip6_ozone_skeie_fits.csv
        self.o3_coeff = {
            'CH4': O3_coeff(0.00017475672916757873, 6.207207761387641e-5),
            'N2O': O3_coeff(0.0007100176652551478, 0.0004707586412175701),
            'ODS': O3_coeff(-0.00012500512605803356, -0.00011302006242177078),
            'CO': O3_coeff(0.00015486441650972913, 0.00013103040974940267),
            'VOC': O3_coeff(0.0003293805536946262, 0.00032758074092083077),
            'NOx': O3_coeff(0.001796619335549982, 0.0009827412018403123),
        }

    def _etminan_CO2(self, *args, **kw):
        """
        Apply Etminan scheme to CO2 concentrations in ppm.

        Returns
        -------
            Radiative forcing in W/m2
        """
        parms = self.parms_etminan

        cc0 = parms.C0
        cn0 = parms.N0
        cc = np.hstack(args)
        cn = np.array(kw.get('cn2o', cn0))

        return (
            parms.a1 * (cc-cc0)**2
            + parms.b1 * np.fabs(cc-cc0)
            + parms.c1 * 0.5*(cn + cn0)
            + 5.36
        ) * np.log(cc/cc0)

    def _etminan_N2O(self, *args, **kw):
        """
        Apply Etminan scheme to N2O concentrations in ppb.

        Returns
        -------
            Radiative forcing in W/m2
        """
        parms = self.parms_etminan

        cn0 = parms.N0
        cc0 = parms.C0
        cm0 = parms.M0
        cn = np.hstack(args)
        cc = np.array(kw.get('cco2', cc0))
        cm = np.array(kw.get('cch4', cm0))

        return (
            parms.a2 * 0.5*(cc + cc0)
            + parms.b2 * 0.5*(cn + cn0)
            + parms.c2 * 0.5*(cm + cm0)
            + 0.117
        ) * (np.sqrt(cn) - np.sqrt(cn0))

    def _etminan_CH4(self, *args, **kw):
        """
        Apply Etminan scheme to CH4 concentrations in ppb.

        Returns
        -------
            Radiative forcing in W/m2
        """
        parms = self.parms_etminan

        cm0 = parms.M0
        cn0 = parms.N0
        cm = np.hstack(args)
        cn = np.array(kw.get('cn2o', cn0))

        return (
            parms.a3 * 0.5*(cm + cm0)
            + parms.b3 * 0.5*(cn + cn0)
            + 0.043
        ) * (np.sqrt(cm) - np.sqrt(cm0))

    def c2erf_etminan(self, gas, *args, **kw):
        """
        Apply Etminan scheme to CO2/CH4/N2O concentrations in ppm/ppb/ppb,
        given by non-keyword arguments, to be passed to numpy.hstack.
        Overlapping gas concentrations are optionally given by keyword arguments
        'cco2' for CO2, referenced when the target gas is N2O;
        'cch4' for CH4, referenced when the target gas is N2O;
        and 'cn2o' for N2O, referenced when the target gas is CO2 or CH4.

        Valid ranges are as follows:
            180-2000 ppm for CO2
            340-3500 ppb for CH4
            200-525 ppb for N2O

        Parameters
        ----------
        gas
            Target gas: 'CO2', 'N2O', or 'CH4'

        Returns
        -------
            Radiative forcing in W/m2
        """
        erf = getattr(self, f'_etminan_{gas}')(*args, **kw)
        return erf.item() if len(erf)==1 else erf

    def _sarf_ar6_CO2(self, cc, **kw):
        """
        Apply AR6 scheme to CO2 concentrations.

        Parameters
        ----------
        cc
            CO2 concentrations in ppm.

        Returns
        -------
            Stratospheric-temperature-adjusted radiative forcing in W/m2
        """
        parms = self.parms_ar6_ghg

        cc0 = parms.C0_1750
        cn0 = parms.N0_1750
        a1 = parms.a1
        b1 = parms.b1
        c1 = parms.c1
        d1 = parms.d1

        # scale = parms.CO2_scale
        scale = 1

        cn = np.array(kw.get('cn2o', cn0))
        ccamax = cc0 - b1/(2*a1)
        alphap = np.full_like(cc, d1)
        alphap = np.where(
            cc < cc0,
            alphap,
            np.where(
                cc < ccamax,
                alphap + a1*(cc-cc0)**2 + b1*(cc-cc0),
                alphap - b1**2/(4*a1)
            )
        )
        return (alphap + c1*np.sqrt(cn)) * np.log(cc/cc0) * scale

    def _sarf_ar6_N2O(self, cn, **kw):
        """
        Apply AR6 scheme to N2O concentrations.

        Parameters
        ----------
        cn
            N2O concentrations in ppb.

        Returns
        -------
            Stratospheric-temperature-adjusted radiative forcing in W/m2
        """
        parms = self.parms_ar6_ghg

        cn0 = parms.N0_1750
        cc0 = parms.C0_1750
        cm0 = parms.M0_1750

        cc = np.array(kw.get('cco2', cc0))
        cm = np.array(kw.get('cch4', cm0))

        return (
            parms.a2 * np.sqrt(cc)
            + parms.b2 * np.sqrt(cn)
            + parms.c2 * np.sqrt(cm)
            + parms.d2
        ) * (np.sqrt(cn) - np.sqrt(cn0))

    def _sarf_ar6_CH4(self, cm, **kw):
        """
        Apply AR6 scheme to CH4 concentrations.

        Parameters
        ----------
        cm
            CH4 concentrations in ppb.

        Returns
        -------
            Stratospheric-temperature-adjusted radiative forcing in W/m2
        """
        parms = self.parms_ar6_ghg

        cm0 = parms.M0_1750
        cn0 = parms.N0_1750

        cn = np.array(kw.get('cn2o', cn0))

        return (
            parms.a3 * np.sqrt(cm)
            + parms.b3 * np.sqrt(cn)
            + parms.d3
        ) * (np.sqrt(cm) - np.sqrt(cm0))

    def c2erf_ar6(self, gas, *args, **kw):
        """
        Apply AR6 scheme to CO2/CH4/N2O concentrations in ppm/ppb/ppb,
        given by non-keyword arguments, to be passed to numpy.hstack.
        Overlapping gas concentrations are optionally given by keyword arguments
        'cco2' for CO2, referenced when the target gas is N2O;
        'cch4' for CH4, referenced when the target gas is N2O;
        and 'cn2o' for N2O, referenced when the target gas is CO2 or CH4.

        Valid ranges are as follows:
            180-2000 ppm for CO2
            340-3500 ppb for CH4
            200-525 ppb for N2O

        Parameters
        ----------
        gas
            Target gas: 'CO2', 'N2O', or 'CH4'

        Returns
        -------
            Effective radiative forcing in W/m2
        """
        parms = self.parms_ar6_ghg
        sarf = getattr(self, f'_sarf_ar6_{gas}')

        erf = sarf(np.hstack(args), **kw)

        # Tropospheric adjustment
        names = [name for name in parms() if name.endswith('_adj')]
        if f'{gas}_adj' in names:
            erf *= 1. + getattr(parms, f'{gas}_adj')

        return erf.item() if len(erf)==1 else erf

    def weight2conc(self, din, units_in, molweight, units_out):
        """
        Convert mass input to concentration changes in the atmosphere

        Parameters
        ----------
        din
            Mass input

        units_in
            Units of mass input, such as 'kt' and 'Gt'

        molweight
            Molecular weight
            or gas name defined in `self.parms['molecular_weight']`

        units_out
            Units of concentration, such as 'ppm', 'ppb', and 'ppt'

        Returns
        -------
            Concentration changes

        Examples
        --------
        >>> self.weight2conc(1, 'kt', 'CFC-11', 'ppt')
        0.04124424823180752
        """
        map_conv2gram = {
            'Gt': 1e15,
            'Mt': 1e12,
            'kt': 1e9,
        }
        map_conv2ppx = {
            'ppm': 1e-3,
            'ppb': 1.,
            'ppt': 1e3,
        }
        dryair = 0.1765 # Teramoles per ppb of dry air
        if isinstance(molweight, str):
            molweight = self.ghgs[molweight].weight

        dout = din * (
            map_conv2gram[units_in] / (molweight * dryair)
            * 1e-12 * map_conv2ppx[units_out]
        )

        return dout

    def gascycle(self, x, tau, dt=1., yinit=0.):
        """
        Perform time integration of a simple gas cycle
        for given emissions, and return concentrations.

        Parameters
        ----------
        x
            Time series of emissions
        tau
            Time constant
        dt, optional
            Time step in year, by default 1.
        yinit, optional
            Initial concentration, by default 0.

        Returns
        -------
            Time series of concentrations
        """
        rd = np.exp(-dt/tau)
        rs = (1 - rd)*tau
        rr = (dt - rs)*(tau/dt)
        y = np.zeros(len(x))
        y[0] = yinit
        for i in range(len(x)-1):
            y[i+1] = rd*y[i] + rs*x[i] + rr*(x[i+1]-x[i])
        return y

    def gascycle_inv(self, y, tau, dt=1., xinit=0.):
        """
        Perform time integration of a simple gas cycle
        for given concentrations, and return emissions.

        Parameters
        ----------
        x
            Time series of concentrations
        tau
            Time constant
        dt, optional
            Time step in year, by default 1.
        yinit, optional
            Initial emissions, by default 0.

        Returns
        -------
            Time series of emissions
        """
        rd = np.exp(-dt/tau)
        rs = (1 - rd)*tau
        rr = (dt - rs)*(tau/dt)
        x = np.zeros(len(y))
        x[0] = xinit
        for i in range(len(y)-1):
            x[i+1] = (rd*dt-rs)/(dt-rs)*x[i] + (y[i+1]-rd*y[i])/rr
        return x

    def ozone(self, emis, conc, emis_pi, conc_pi, **kw):
        """
        Compute effective radiative forcing of ozone based on the AR6 method.
        Temperature dependency can be considered when keyword argument 'tsa'
        is given as a surface temperature anomaly.
        Uncertainty can be considered when keyword argument 'perturb' is given
        as a perturbation factor in terms of 90% uncertainty ranges

        Parameters
        ----------
        emis
            CO, VOC, and NOx emissions in Mt CO/yr, Mt VOC/yr, and Mt NOx/yr

        conc
            CH4, N2O, and ODSs concentrations in ppb, ppb, and ppt

        emis_pi
            Same as emis, but preindustrial emissions

        conc_pi
            Same as conc, but preindustrial concentrations

        Returns
        -------
            ERF of ozone
            Categorized values are returned when keyword argument 'categorized'
            is given as True.
        """
        tsa = kw.get('tsa', 0.)
        perturb = kw.get('perturb', 0.)
        categorized = kw.get('categorized', False)
        feedback = self.parms_ar6_ghg.fb_o3_forcing

        coeff = {
            k: v.mean + v.u90 * perturb
            for k, v in self.o3_coeff.items()
        }

        ret = {
            v: coeff[v] * (conc[v] - conc_pi[v])
            for v in ['CH4', 'N2O']
        }

        def eesc(v):
            # Equivalent effective stratospheric chlorine for ozone depleting compounds
            parm = self.ods[v]
            return (conc[v] - conc_pi[v]) * parm.fracrel * (parm.cl_atoms + 45 * parm.br_atoms)

        ods = sum([eesc(v) for v in self.ods])
        ret['ODS'] = coeff['ODS'] * ods

        ret.update({
            v: coeff[v] * (emis[v] - emis_pi[v])
            for v in ['CO', 'VOC', 'NOx']
        })

        if not categorized:
            ret = sum(ret.values()) + tsa * feedback
        else:
            ret['feedback'] = tsa * feedback

        return ret

