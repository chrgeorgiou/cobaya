# semi-pseudo-code of what CCL theory might look like

# For Cobaya docs see
# https://cobaya.readthedocs.io/en/devel/theory.html
# https://cobaya.readthedocs.io/en/devel/theories_and_dependencies.html

import numpy as np
from cobaya.theory import Theory


class CCL(Theory):

    def initialize(self):
        self._zmax = 0
        self._kmax = 0
        self._z_sampling = []
        self._var_pairs = set()

    def get_requirements(self):
        # (Should this be moved to "needs"?) Requirements for creating a CCL cosmo object:
        return {'omch2', 'ombh2', 'H0', 'ns', 'As'}

    def needs(self, **requirements):
        # requirements is dictionary of things requested by likelihoods
        # check all requirements, and find what CCL needs to compute them
        # Note this may be called more than once
        self._var_pairs = {('delta_tot', 'delta_tot')}  # fix for now

        cl = requirements.get("angular_cl")
        if cl:
            self._kmax = max(10, self._kmax)
            self._zmax = max(5, self._zmax)
        """
            if cl.get('WeakLensingTracer'):
                self._var_pairs.add(('Weyl', 'Weyl'))
            if cl.get('NumberCountsTracer'):
                self._var_pairs.add(('delta_nonu', 'delta_nonu'))
        # similarly for cross-spectra etc
        """

        # make a dictionary of the things CCL needs from CAMB/CLASS
        # see https://cobaya.readthedocs.io/en/devel/cosmo_theories_likes.html
        needs = {}
        # sampling_step = self._zmax * 0.1
        # self._z_sampling = np.arange(0, self._zmax, sampling_step)
        self._z_sampling = np.linspace(0, self._zmax, 100) # Fixed at 100 steps to z max for now
        if self._kmax:
            needs['Pk_grid'] = {'vars_pairs': [('delta_tot', 'delta_tot')],
                                'nonlinear': False,  # allows both linear and non-linear
                                'z': self._z_sampling,
                                'k_max': self._kmax
                                }
        #            needs['baryon_fudge1'] = None  # nuisance parameter needed
        if self._zmax:
            pass
        # TODO: Are these dictionaries with the arrays required for computing chi(z) and H(z)?
        #   needs['comoving_radial_distance'] = {'z': self._z_sampling}
        #   needs['Hubble'] = {'z': self._z_sampling}

        # if any cosmological parameters needed can add them to requirements
        # TODO: Figure out which parameters are necessary, (e.g. A_s, H0 to go from E(z) to H(z))
        # needs['H0'] = None
        # needs['A_s'] = None

        return needs

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return ['baryon_fudge1', 'baryon_fudge2']

    def calculate(self, state, want_derived=True, **params_values_dict):
        # calculate the things that were requested and save into state, ready to be
        # retrieved as needed by get_angular_cl() and other methods
        # get our requirements from self.provider
        if self._zmax:
            pass
            # TODO: Here the CCL functions that return chi(z), E(z)? Note: CCL accepts scale factor as variable, change z to a.
            # distance = self.provider.get_comoving_radial_distance(self._z_sampling)

        if self._kmax:
            # Create a CCL cosmology object
            h = self.provider.get_param('H0') / 100.
            Omega_c = self.provider.get_param('omch2') / h ** 2.
            Omega_b = self.provider.get_param('ombh2') / h ** 2.

            import pyccl as ccl
            matter_pk = 'linear'
            cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h,
                                   n_s=self.provider.get_param('ns'),
                                   A_s=self.provider.get_param('As'), matter_power_spectrum=matter_pk)
            #baryon_fudge1 = params_values_dict['baryon_fudge1']
            state['angular_cl'] = {}
            for pair in self._var_pairs:
                k, z, Pk = self.provider.get_Pk_grid(var_pair=pair,
                                                          nonlinear=False)
                # import pdb; pdb.set_trace()
                # .. use them and store calculated result
                # TODO: Here create a cosmo object with paramteres read from input.
                # cosmo = CCL(distance=distance, k=k, z=z, PKWell=PK_Weyl ...)
                # Array z is sorted in ascending order. CCL requires an ascending scale factor as input
                # np.flip(arr, axis=0) flips the rows of arr, thus making PK_Weyl with z in descending order.
                Pk = np.flip(Pk, axis=0)
                a = np.sort(1. / (1 + z))
                P_of_k_a = ccl.pk2d.Pk2D(a_arr=a, lk_arr=np.log(k), pk_arr=Pk,
                                          is_logp=False)

                # FIXME: CCL angular_cl requires 2 tracers. How to build those?
                z_n = np.linspace(0., 1., 200)
                n = np.ones(z_n.shape)
                tracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
                tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
                ell =  np.logspace(np.log10(3),3) # FIXME: At which ell do we compute C_l? Now it is log-spaced from 3 to 1000
                cls = ccl.cls.angular_cl(cosmo, tracer1, tracer2, ell, p_of_k_a=P_of_k_a)
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

        """
        # Test the C_ls computed using a power spectrum against the C_ls computed directly, in CCL:
        import pyccl as ccl
        cosmo = ccl.Cosmology(Omega_c=omc, Omega_b=omb, h=h,
                              n_s=ns, A_s=As, m_nu=mnu, transfer_function='boltzmann_camb', matter_power_spectrum='linear')
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
        # print(results)
        return


    info = {'likelihood': {'C_l': cl_likelihood},
            'theory': OrderedDict({
                'camb': {'stop_at_error': True,
                         'requires': {'omch2', 'ombh2', 'H0', 'ns', 'As'}},
                'CCL': {'external': CCL}}),
            'params': dict(H0={'prior': {'min': 0, 'max': 100}}, **camb_params),
            'debug': debug, 'stop_at_error': True}

    model = get_model(info)
    model.loglikes({'H0': H0})
