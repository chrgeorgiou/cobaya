import numpy as np
from cobaya.theory import Theory


class CCL(Theory):

    def initialize(self):
        self._zmax = 0
        self._kmax = 0
        self._z_sampling = []
        self._var_pairs = set()

    def get_requirements(self):
        # These are required to construct a CCL cosmology object.
        # 'As' could be substituted by sigma8.
        return {'omch2', 'ombh2', 'H0', 'ns', 'As'}

    def needs(self, **requirements):
        # requirements is dictionary of things requested by likelihoods
        # Note this may be called more than once
        self._var_pairs = {('delta_tot', 'delta_tot')}  # fix for now

        cl = requirements.get("angular_cl")
        if cl:
            self._kmax = max(10, self._kmax)
            self._zmax = max(5, self._zmax)
        """
            # This is for when more power spectra can be read into CCL.
            if cl.get('WeakLensingTracer'):
                self._var_pairs.add(('Weyl', 'Weyl'))
            if cl.get('NumberCountsTracer'):
                self._var_pairs.add(('delta_nonu', 'delta_nonu'))
        # similarly for cross-spectra etc
        """

        # Dictionary of the things CCL needs from CAMB/CLASS
        needs = {}
        # sampling_step = self._zmax * 0.1
        # self._z_sampling = np.arange(0, self._zmax, sampling_step)
        self._z_sampling = np.linspace(0, self._zmax, 100) # Fixed at 100 steps to z max for now
        if self._kmax:
            needs['Pk_grid'] = {'vars_pairs': [('delta_tot', 'delta_tot')],
                                'nonlinear': (True, False),
                                'z': self._z_sampling,
                                'k_max': self._kmax
                                }
        #    needs['baryon_fudge1'] = None  # nuisance parameter needed
        if self._zmax:
            needs['Hubble'] = {'z': self._z_sampling}
            needs['comoving_radial_distance'] = {'z': self._z_sampling}

        return needs

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return ['baryon_fudge1', 'baryon_fudge2']

    def calculate(self, state, want_derived=True, **params_values_dict):
        # calculate the things that were requested and save into state, ready to be
        # retrieved as needed by get_angular_cl() and other methods
        # get our requirements from self.provider
        if self._zmax:
            # Read in the distance and Hubble parameter arrays:
            distance = self.provider.get_comoving_radial_distance(self._z_sampling)
            hubble_z = self.provider.get_Hubble(self._z_sampling)
            E_of_z = hubble_z/self.provider.get_param('H0')
            # Array z is sorted in ascending order. CCL requires an ascending scale factor as input
            # Flip the arrays to make them a function of the increasing scale factor.
            # If redshift sampling is changed, check that it is monotonically increasing
            distance = np.flip(distance)
            E_of_z = np.flip(E_of_z)

        if self._kmax:
            state['angular_cl'] = {}
            for pair in self._var_pairs:
                # Get the matter power spectrum:
                k, z, Pk_lin = self.provider.get_Pk_grid(var_pair=pair, nonlinear=False)
                k, z, Pk_nonlin = self.provider.get_Pk_grid(var_pair=pair, nonlinear=True)
                # Array z is sorted in ascending order. CCL requires an ascending scale factor as input
                # np.flip(arr, axis=0) flips the rows of arr, thus making Pk with z in descending order.
                Pk_lin = np.flip(Pk_lin, axis=0)
                Pk_nonlin = np.flip(Pk_nonlin, axis=0)
                a = np.sort(1. / (1 + z))

                # Create a CCL cosmology object
                import pyccl as ccl
                h = self.provider.get_param('H0') / 100.
                Omega_c = self.provider.get_param('omch2') / h ** 2.
                Omega_b = self.provider.get_param('ombh2') / h ** 2.
                #baryon_fudge1 = params_values_dict['baryon_fudge1']
                # Currently, CCL requires the linear matter perturbations growth factor and rate.
                # Compute these from CCL:
                cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h,
                                      n_s=self.provider.get_param('ns'),
                                      A_s=self.provider.get_param('As'))
                growth = ccl.background.growth_factor(cosmo, a)
                fgrowth = ccl.background.growth_rate(cosmo, a)
                # In order to use CCL with input arrays, the cosmology object needs to be reset.
                cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h,
                                      n_s=self.provider.get_param('ns'),
                                      A_s=self.provider.get_param('As'))
                cosmo._set_background_from_arrays(a_array=a,
                                                  chi_array=distance,
                                                  hoh0_array=E_of_z,
                                                  growth_array=growth,
                                                  fgrowth_array=fgrowth)
                cosmo._set_linear_power_from_arrays(a, k, Pk_lin)
                cosmo._set_nonlin_power_from_arrays(a, k, Pk_nonlin)

                # FIXME: CCL angular_cl requires 2 tracers. How to build those?
                z_n = np.linspace(0., 1., 200)
                n = np.ones(z_n.shape)
                tracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
                tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
                ell =  np.logspace(np.log10(3),3) # FIXME: At which ell do we compute C_l? Now it is log-spaced from 3 to 1000
                cls = ccl.cls.angular_cl(cosmo, tracer1, tracer2, ell)
                state['angular_cl'][pair] = {'cls': cls, 'ell': ell}

    def get_angular_cl(self, pair=('delta_tot', 'delta_tot')):
        # function used by likelihoods to get results from CCL
        # (does not include CMB spectra which can be obtained from CAMB/Class directly)
        return self._current_state['angular_cl'].get(pair)


if __name__ == '__main__':
    from collections import OrderedDict
    from cobaya.model import get_model

    ombh2 = 0.022
    omch2 = 0.122
    H0 = 67
    As = 0.2132755716e-8
    ns = 0.965
    mnu = 0.

    h = H0 / 100.
    omb = ombh2 / (h ** 2.)
    omc = omch2 / (h ** 2.)

    camb_params = {
        "ombh2": ombh2,
        "omch2": omch2,
        #    "H0": H0,
        "As": As,
        "ns": ns,
        "mnu": mnu}

    debug = True


    def cl_likelihood(_theory={'angular_cl': {'_kmax': 2}}):
        results = _theory.get_angular_cl()
        # print(results)
        """
        # Test the C_ls computed using a power spectrum against the C_ls computed directly, in CCL:
        import pyccl as ccl
        cosmo = ccl.Cosmology(Omega_c=omc, Omega_b=omb, h=h,
                              n_s=ns, A_s=As, m_nu=mnu, transfer_function='boltzmann_camb', matter_power_spectrum='halofit')
        # Define a simple binned galaxy number density curve as a function of redshift
        z_n = np.linspace(0., 1., 200)
        n = np.ones(z_n.shape)
        # Create objects to represent tracers of the weak lensing signal with this
        # number density (with has_intrinsic_alignment=False)
        lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
        lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
        ell = results['ell']
        # Computing C_l without parsing a power spectrum
        cls = ccl.angular_cl(cosmo, lens1, lens2, ell)

        cls_pk = results['cls']
        from matplotlib import pyplot as plt
        plt.plot(ell, cls / cls_pk)
        plt.xscale('log')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell$(cosmo)/$C_\ell$(from Pk)')
        plt.show()
        """
        return


    info = {'likelihood': {'C_l': cl_likelihood},
            'theory': OrderedDict({
                'camb': {'stop_at_error': True},
                'CCL': {'external': CCL}}),
            'params': dict(H0={'prior': {'min': 0, 'max': 100}}, **camb_params),
            'debug': debug, 'stop_at_error': True}


    model = get_model(info)
    model.loglikes({'H0': H0})
