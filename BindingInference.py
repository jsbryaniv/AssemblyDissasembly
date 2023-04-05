
# Import standard modules
import os
import sys
import h5py
import copy
import time
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy import stats
from types import SimpleNamespace


# Declare class
class StateInference:

    # Declare parameters
    PARAMETERS = {
        # Variables
        "s": None,
        "pi": None,
        "mu_flor": None,
        "mu_back": None,
        "gain": None,
        "P": None,
        # Priors
        "pi_conc": None,
        "mu_flor_shape": None,
        "mu_flor_scale": None,
        "mu_back_shape": None,
        "mu_back_scale": None,
        "gain_shape": None,
        "gain_scale": None,
        # Experiment
        "dt": 1,
        "num_states": 10,
        "num_frames": None,
        # Sampling
        "gain_proposal_shape": 10,
        "mu_flor_proposal_shape": 10,
        "mu_back_proposal_shape": 10,
    }
    
    # Define FFBS
    @nb.njit(cache=True)
    def FFBS(lhood, transition_matrix):

        # Get parameters
        num_states, num_data = lhood.shape
        pi0 = transition_matrix[-1, :]
        pis = transition_matrix[:-1, :]
        states = np.zeros(num_data, dtype=np.int32)

        # Forward filter
        forward = np.zeros((num_states, num_data))
        forward[:, 0] = lhood[:, 0] * pi0
        forward[:, 0] /= np.sum(forward[:, 0])
        for n in range(1, num_data):
            forward[:, n] = lhood[:, n] * (pis.T @ forward[:, n - 1])
            forward[:, n] /= np.sum(forward[:, n])

        # Backward sample
        s = np.searchsorted(np.cumsum(forward[:, -1]), np.random.rand())
        states[-1] = s
        for m in range(1, num_data):
            n = num_data - m - 1
            backward = forward[:, n] * pis[:, s]
            backward /= np.sum(backward)
            s = np.searchsorted(np.cumsum(backward), np.random.rand())
            states[n] = s

        return states


    
    @staticmethod
    def simulate_data():
        pass

    @staticmethod
    def initialize_parameters(data, parameters=None, **kwargs):
        
        # Set up parameters
        if parameters is None:
            parameters = {}
        parameters = copy.deepcopy(parameters)
        parameters = {**StateInference.PARAMETERS, **parameters, **kwargs}

        # Set up variables
        variables = SimpleNamespace(**parameters)
        P = variables.P
        s = variables.s 
        pi = variables.pi
        pi_conc = variables.pi_conc
        mu_flor = variables.mu_flor
        mu_flor_shape = variables.mu_flor_shape
        mu_flor_scale = variables.mu_flor_scale
        mu_back = variables.mu_back
        mu_back_shape = variables.mu_back_shape
        mu_back_scale = variables.mu_back_scale
        gain = variables.gain
        gain_shape = variables.gain_shape
        gain_scale = variables.gain_scale
        dt = variables.dt
        num_states = variables.num_states
        num_frames = variables.num_frames
        mu_flor_proposal_shape = variables.mu_flor_proposal_shape
        mu_back_proposal_shape = variables.mu_back_proposal_shape

        # Initialize constants
        num_frames = data.shape[0]
        variables.num_frames = num_frames

        # Initialize gain
        if gain_shape is None:
            gain_shape = 2
        if gain_scale is None:
            gain_scale = 20
        if gain is None:
            gain = gain_shape * gain_scale
        variables.gain = gain
        variables.gain_shape = gain_shape
        variables.gain_scale = gain_scale

        # Initialize states
        s = np.zeros(num_frames, dtype=int)
        variables.s = s

        # Initialize transition probabilities
        if pi_conc is None:
            pi_conc = np.zeros((num_states+1, num_states))
            pi_conc[-1, :] = 1
            pi_conc[:-1, :] += (
                10 * np.eye(num_states)
                + np.eye(num_states, k=1)
                + np.eye(num_states, k=-1)
            )
            for k in range(num_states + 1):
                pi_conc[k, :] /= np.sum(pi_conc[k, :])
        if pi is None:
            pi = pi_conc.copy()
            for k in range(num_states + 1):
                pi[k, :] /= np.sum(pi[k, :])
        variables.pi = pi
        variables.pi_conc = pi_conc

        # Initialize background 
        if mu_back_shape is None:
            mu_back_shape = 2
        if (mu_back_scale is None) and (data is not None):
            mu_back_scale = np.mean(np.sort(data)[:int(num_frames/10)]) / gain / mu_back_shape
        if mu_back is None:
            mu_back = mu_back_shape * mu_back_scale
        variables.mu_back = mu_back
        variables.mu_back_shape = mu_back_shape
        variables.mu_back_scale = mu_back_scale

        # Initialize fluorophore brightness
        if mu_flor_shape is None:
            mu_flor_shape = 2
        if (mu_flor_scale is None) and (data is not None):
            mu_flor_scale = (np.mean(np.sort(data)[-int(num_frames/10):]) / gain - mu_back) / mu_flor_shape
        if mu_flor is None:
            mu_flor = mu_flor_shape * mu_flor_scale
        variables.mu_flor = mu_flor
        variables.mu_flor_shape = mu_flor_shape
        variables.mu_flor_scale = mu_flor_scale

        # Initialze probability
        P = -np.inf
        variables.P = P
            
        # Return variables
        return variables

    @staticmethod
    def sample_states(data, variables):
        
        # Extract variables
        s = variables.s
        pi = variables.pi
        mu_flor = variables.mu_flor
        mu_back = variables.mu_back
        dt = variables.dt
        gain = variables.gain
        num_states = variables.num_states
        num_frames = variables.num_frames

        # Set up log likelihood matrix
        lhood = np.zeros((num_states, num_frames))
        for k in range(num_states):
            mu = k*mu_flor + mu_back
            lhood[k, :] = stats.gamma.logpdf(data, a=gain, scale=mu)

        # Softmax for numerical stability
        lhood = np.exp(lhood - np.max(lhood, axis=0))

        # Sample states using FFBS
        s[:] = FFBS(lhood, pi)

        # Update variables
        variables.s = s

        # Return variables
        return variables

    @staticmethod
    def sample_transitions(data, variables):
        
        # Extract variables
        s = variables.s
        pi = variables.pi
        pi_conc = variables.pi_conc
        num_states = variables.num_states
        num_frames = variables.num_frames

        # Get counts
        counts = np.zeros((num_states+1, num_states))
        s_old = -1
        for n in range(num_frames):
            s_new = s[n]
            counts[s_old, s_new] += 1
            s_old = s_new

        # Sample transition probabilities
        for k in range(num_states + 1):
            ids = pi_conc[k, :] > 0
            pi[k, ids] = stats.dirichlet.rvs(counts[k, ids] + pi_conc[k, ids])
        pi += 1e-100

        # Return variables
        return variables

    @staticmethod
    def sample_brightness(data, variables):

        # Extract variables
        s = variables.s
        mu_flor = variables.mu_flor
        mu_flor_shape = variables.mu_flor_shape
        mu_flor_scale = variables.mu_flor_scale
        mu_back = variables.mu_back
        mu_back_shape = variables.mu_back_shape
        mu_back_scale = variables.mu_back_scale
        dt = variables.dt
        gain = variables.gain
        num_states = variables.num_states
        num_frames = variables.num_frames
        mu_flor_proposal_shape = variables.mu_flor_proposal_shape
        mu_back_proposal_shape = variables.mu_back_proposal_shape

        # Define conditional probability
        def prob(mu_flor_, mu_back_):
            # Calculate prior
            P = (
                stats.gamma.logpdf(mu_flor_, a=mu_flor_shape, scale=mu_flor_scale)
                + stats.gamma.logpdf(mu_back_, a=mu_back_shape, scale=mu_back_scale)
            )
            # Calculate likelihood
            for k in range(num_states):
                ids = np.where(s == k)[0]
                mu = k * mu_flor_ + mu_back_
                P += (
                    np.sum(stats.gamma.logpdf(data[ids], a=gain, scale=mu))
                )
            return P

        # Sample brightnesses
        for _ in range(10):

            # Sample background
            a = mu_back_proposal_shape
            mu_back_old = copy.deepcopy(mu_back)
            mu_back_new = stats.gamma.rvs(a=a, scale=mu_back_old/a)
            P_old = prob(mu_flor, mu_back_old)
            P_new = prob(mu_flor, mu_back_new)
            acc_prob = (
                P_new - P_old
                + stats.gamma.logpdf(mu_back_old, a=a, scale=mu_back_new/a)
                - stats.gamma.logpdf(mu_back_new, a=a, scale=mu_back_old/a)
            )
            if acc_prob > np.log(np.random.rand()):
                mu_back = mu_back_new
            
            # Sample fluorophore brightness
            a = mu_flor_proposal_shape
            mu_flor_old = copy.deepcopy(mu_flor)
            mu_flor_new = stats.gamma.rvs(a=a, scale=mu_flor_old/a)
            P_old = prob(mu_flor_old, mu_back)
            P_new = prob(mu_flor_new, mu_back)
            acc_prob = (
                P_new - P_old
                + stats.gamma.logpdf(mu_flor_old, a=a, scale=mu_flor_new/a)
                - stats.gamma.logpdf(mu_flor_new, a=a, scale=mu_flor_old/a)
            )
            if acc_prob > np.log(np.random.rand()):
                mu_flor = mu_flor_new

        # Update variables
        variables.mu_flor = mu_flor
        variables.mu_back = mu_back

        # Return variables
        return variables

    @staticmethod
    def sample_gain(data, variables):

        # Extract variables
        s = variables.s
        pi = variables.pi
        mu_flor = variables.mu_flor
        mu_back = variables.mu_back
        dt = variables.dt
        gain = variables.gain
        num_states = variables.num_states
        num_frames = variables.num_frames
        gain_proposal_shape = variables.gain_proposal_shape

        # Define conditional probability
        trace = np.zeros_like(data)
        for k in range(num_states):
            ids = np.where(s == k)[0]
            trace[ids] = k * mu_flor + mu_back
        def prob(gain_):
            P = np.sum(stats.gamma.logpdf(data, a=gain_, scale=trace))
            return P
        
        # Sample gain
        a = gain_proposal_shape
        for _ in range(10):
            gain_old = copy.deepcopy(gain)
            gain_new = stats.gamma.rvs(a=a, scale=gain_old/a)
            P_old = prob(gain_old)
            P_new = prob(gain_new)
            acc_prob = (
                P_new - P_old
                + stats.gamma.logpdf(gain_old, a=a, scale=gain_new/a)
                - stats.gamma.logpdf(gain_new, a=a, scale=gain_old/a)
            )
            if acc_prob > np.log(np.random.rand()):
                gain = gain_new
        
        # Update variables
        variables.gain = gain

        # Return variables
        return variables
    
    @staticmethod
    def sample_states_and_brightness(data, variables):

        # Sample new brightnesses
        variables_old = variables
        variables_new = copy.deepcopy(variables)
        mu_flor_old = variables_old.mu_flor
        mu_flor_new = stats.expon.rvs(scale=mu_flor_old)
        variables_new.mu_flor = mu_flor_new
        variables_new = StateInference.sample_states(data, variables_new)

        # Calculate acceptance probability
        P_old = StateInference.calculate_posterior(data, variables_old)
        P_new = StateInference.calculate_posterior(data, variables_new)
        acc_prob = (
            P_new - P_old
            + stats.expon.logpdf(mu_flor_old, scale=mu_flor_new)
            - stats.expon.logpdf(mu_flor_new, scale=mu_flor_old)
        )
        if acc_prob > np.log(np.random.rand()):
            variables = variables_new
        
        # Return variables
        return variables
    
    @staticmethod
    def sample_states_and_background(data, variables):

        # Sample new brightnesses
        variables_old = variables
        variables_new = copy.deepcopy(variables)
        mu_back_old = variables_old.mu_flor
        mu_back_new = stats.expon.rvs(scale=mu_back_old)
        variables_new.mu_flor = mu_back_new
        variables_new = StateInference.sample_states(data, variables_new)

        # Calculate acceptance probability
        P_old = StateInference.calculate_posterior(data, variables_old)
        P_new = StateInference.calculate_posterior(data, variables_new)
        acc_prob = (
            P_new - P_old
            + stats.expon.logpdf(mu_back_old, scale=mu_back_new)
            - stats.expon.logpdf(mu_back_new, scale=mu_back_old)
        )
        if acc_prob > np.log(np.random.rand()):
            variables = variables_new
        
        # Return variables
        return variables
    
    @staticmethod
    def calculate_posterior(data, variables, **kwargs):

        # Set up variables
        if len(kwargs) > 0:
            variables = copy.copy(variables)
            for key, value in kwargs.items():
                setattr(variables, key, value)
        
        # Get variables
        s = variables.s
        pi = variables.pi
        pi_conc = variables.pi_conc
        mu_flor = variables.mu_flor
        mu_flor_shape = variables.mu_flor_shape
        mu_flor_scale = variables.mu_flor_scale
        mu_back = variables.mu_back
        mu_back_shape = variables.mu_back_shape
        mu_back_scale = variables.mu_back_scale
        gain = variables.gain
        gain_shape = variables.gain_shape
        gain_scale = variables.gain_scale
        dt = variables.dt
        num_states = variables.num_states
        num_frames = variables.num_frames

        # Calculate prior
        prior = (
            stats.gamma.logpdf(mu_flor, a=mu_flor_shape, scale=mu_flor_scale)
            + stats.gamma.logpdf(mu_back, a=mu_back_shape, scale=mu_back_scale)
            + stats.gamma.logpdf(gain, a=gain_shape, scale=gain_scale)
        )
        for k in range(num_states + 1):
            ids = pi_conc[k, :] > 0
            prior += stats.dirichlet.logpdf(pi[k, ids], pi_conc[k, ids])

        # Calculate likelihood
        likelihood = 0
        for k in range(num_states):
            ids = np.where(s == k)[0]
            mu = k * mu_flor + mu_back
            likelihood += (
                np.sum(stats.gamma.logpdf(data[ids], a=gain, scale=mu))
            )

        # Calculate dynamics
        dynamics = 0
        s_old = -1
        for n in range(num_frames):
            s_new = s[n]
            dynamics += np.log(pi[s_old, s_new])
            s_old = s_new

        # Calculate posterior
        P = prior + likelihood + dynamics

        return P

    @staticmethod
    def plot_data(data, variables=None):

        # Set up figure
        fig = plt.gcf()
        fig.clf()
        plt.ion()
        plt.show()
        ax = fig.add_subplot(111)

        # Plot data
        ax.plot(data, "r", label="Data")

        # Plot variables
        if variables is not None:
            # Get variables
            s = variables.s
            mu_flor = variables.mu_flor
            mu_back = variables.mu_back
            gain = variables.gain
            num_states = variables.num_states
            # Calculate trace
            trace = np.zeros_like(data)
            for k in range(num_states):
                ids = np.where(s == k)[0]
                trace[ids] = gain*(k * mu_flor + mu_back)
            # Plot trace
            ax.plot(trace, "b", label=f"Trace\nMax={np.max(s)}")

        # Set up plot
        ax.set_ylabel("Intensity (ADU)")
        ax.set_xlabel("Time (Frame #)")
        ax.legend()
        plt.tight_layout()
        plt.pause(.1)

        # Finish
        return

    @staticmethod
    def analyze_data(data, parameters=None, num_iterations=100, plot=False, verbose=False, **kwargs):

        # Initialize variables
        variables = StateInference.initialize_parameters(data, parameters, **kwargs)
        MAPvariables = copy.deepcopy(variables)

        # Gibbs sampling
        for iteration in range(num_iterations):
            if verbose:
                print(f"Iteration {iteration+1}/{num_iterations}", end='')
            t = time.time()
                
            # Sample states
            variables = StateInference.sample_gain(data, variables)
            variables = StateInference.sample_brightness(data, variables)
            variables = StateInference.sample_transitions(data, variables)
            variables = StateInference.sample_states(data, variables)
            if np.random.rand() < .25:
                variables = StateInference.sample_states_and_brightness(data, variables)

            # Update MAP
            variables.P = StateInference.calculate_posterior(data, variables)
            if variables.P >= MAPvariables.P:
                MAPvariables = copy.deepcopy(variables)

            # Plot
            if plot:
                StateInference.plot_data(data, variables)
                plt.pause(.1)

            # Print
            if verbose:
                print(f" ({time.time()-t:.2f} s) P = {variables.P:.2f}")

        # Return MAP variables
        return MAPvariables



class RateInference:
    

    PARAMETERS = {

        # Variables
        "P": None,
        "k_on": None,
        "k_off": None,
        "k_photo": None,

        # Priors
        "k_on_shape": 2,
        "k_on_scale": None,
        "k_off_shape": 2,
        "k_off_scale": None,
        "k_photo_shape": 2,
        "k_photo_scale": None,

        # Experiment
        "dt": 1,
        "counts": None,
        "laserpowers": None,
        "concentrations": None,
        "settings": None,
        "num_states": None,
        "num_rois": None,
        "num_frames": None,
        "num_settings": None,

        # Sampling
        "proposal_shape": 100,
    }

    @staticmethod
    def initialize_parameters(data, parameters=None, **kwargs):
        
        # Set up parameters
        if parameters is None:
            parameters = {}
        parameters = copy.deepcopy(parameters)
        parameters = {**RateInference.PARAMETERS, **parameters, **kwargs}

        # Set up variables
        variables = SimpleNamespace(**parameters)
        dt = variables.dt
        k_on = variables.k_on
        k_off = variables.k_off
        k_photo = variables.k_photo
        k_on_shape = variables.k_on_shape
        k_on_scale = variables.k_on_scale
        k_off_shape = variables.k_off_shape
        k_off_scale = variables.k_off_scale
        k_photo_shape = variables.k_photo_shape
        k_photo_scale = variables.k_photo_scale
        laserpowers = variables.laserpowers
        concentrations = variables.concentrations
        num_states = variables.num_states
        num_rois = variables.num_rois
        num_frames = variables.num_frames

        # Set up numbers
        num_states = np.max(data) + 1
        num_rois, num_frames = data.shape
        variables.num_states = num_states
        variables.num_rois = num_rois
        variables.num_frames = num_frames

        # Set up experimental parameters
        if dt is None:
            dt = 1
        if concentrations is None:
            concentrations = np.ones(num_rois)
        if laserpowers is None:
            laserpowers = np.ones(num_rois)
        settings = []
        for c in np.unique(concentrations):
            for l in np.unique(laserpowers):
                settings.append((c, l))
        num_settings = len(settings)
        variables.concentrations = concentrations
        variables.laserpowers = laserpowers
        variables.settings = settings

        # Set up counts
        counts = np.zeros((num_settings, num_states, num_states))
        for k, (c, l) in enumerate(settings):
            ids = np.where((concentrations == c) & (laserpowers == l))[0]
            for r in ids:
                counts[k, :, :] += RateInference.calculate_counts(data[r, :], num_states)
        variables.counts = counts
        
        # Set up k_on
        if k_on is None:
            k_on = 1/(dt*num_frames)
        if k_on_scale is None:
            k_on_scale = k_on / k_on_shape
        variables.k_on = k_on
        variables.k_on_shape = k_on_shape
        variables.k_on_scale = k_on_scale

        # Set up k_off
        if k_off is None:
            k_off = 1/(dt*num_frames)
        if k_off_scale is None:
            k_off_scale = k_off / k_off_shape
        variables.k_off = k_off
        variables.k_off_shape = k_off_shape
        variables.k_off_scale = k_off_scale

        # Set up k_photo
        if k_photo is None:
            k_photo = 1/(dt*num_frames)
        if k_photo_scale is None:
            k_photo_scale = k_photo / k_photo_shape
        variables.k_photo = k_photo
        variables.k_photo_shape = k_photo_shape
        variables.k_photo_scale = k_photo_scale

        # Set up P
        P = -np.inf
        variables.P = P

        # Return variables
        return variables
    
    @staticmethod
    def calculate_posterior(data, variables, **kwargs):

        # Set up variables
        if len(kwargs) > 0:
            variables = copy.copy(variables)
            for key, value in kwargs.items():
                setattr(variables, key, value)
        dt = variables.dt
        k_on = variables.k_on
        k_off = variables.k_off
        k_photo = variables.k_photo
        k_on_shape = variables.k_on_shape
        k_on_scale = variables.k_on_scale
        k_off_shape = variables.k_off_shape
        k_off_scale = variables.k_off_scale
        k_photo_shape = variables.k_photo_shape
        k_photo_scale = variables.k_photo_scale
        counts = variables.counts
        laserpowers = variables.laserpowers
        concentrations = variables.concentrations
        settings = variables.settings
        num_states = variables.num_states
        num_rois = variables.num_rois
        num_frames = variables.num_frames
        num_settings = variables.num_settings

        # Calculate P
        P = (
            stats.gamma.logpdf(k_on, k_on_shape, scale=k_on_scale)
            + stats.gamma.logpdf(k_off, k_off_shape, scale=k_off_scale)
            + stats.gamma.logpdf(k_photo, k_photo_shape, scale=k_photo_scale)
        )

        # Calculate likelihood
        for k, (c, l) in enumerate(settings):
            pi_k = RateInference.calculate_transition_matrix(num_states, dt, k_on, k_off, k_photo, c, l)
            counts_k = counts[k, :, :]
            ids = (pi_k > 0) * (counts_k > 0)
            P += np.sum(counts_k[ids] * np.log(pi_k[ids]))

        # Return P
        return P
        
    @staticmethod
    @nb.jit(nopython=True)
    def calculate_counts(trace, num_states):

        # Set up counts
        counts = np.zeros((num_states, num_states))

        # Loop through states
        s_old = trace[1]
        for n in range(1, len(trace)):
            s_new = trace[n]
            counts[s_old, s_new] += 1
            s_old = s_new

        # Return counts
        return counts

    # @nb.jit(nopython=True)
    def calculate_transition_matrix(num_states, dt, k_on, k_off, k_photo, concentration=1, laserpower=1):

        # Set up transition matrix
        pi = np.zeros((num_states, num_states))

        # Loop through states
        for k in range(num_states):

            # Calculate escape rate
            esc_rate = (
                laserpower * k * k_photo
                + (k>0) * k * k_off
                + (k<num_states-1) * concentration * k_on
            )

            # Self transition
            pi[k, k] = np.exp(-dt*esc_rate)

            # Step down
            if k > 0:
                pi[k, k-1] = (1 - pi[k, k]) * k * (laserpower*k_photo + k_off) / esc_rate

            # Step up
            if k < num_states-1:
                pi[k, k+1] = (1 - pi[k, k]) * concentration * k_on / esc_rate
        
        return pi

    @staticmethod
    def sample_rates(data, variables):

        # Extract variables
        dt = variables.dt
        k_on = variables.k_on
        k_off = variables.k_off
        k_photo = variables.k_photo
        k_on_shape = variables.k_on_shape
        k_on_scale = variables.k_on_scale
        k_off_shape = variables.k_off_shape
        k_off_scale = variables.k_off_scale
        k_photo_shape = variables.k_photo_shape
        k_photo_scale = variables.k_photo_scale
        counts = variables.counts
        laserpowers = variables.laserpowers
        concentrations = variables.concentrations
        settings = variables.settings
        num_states = variables.num_states
        num_rois = variables.num_rois
        num_frames = variables.num_frames
        num_settings = variables.num_settings
        proposal_shape = variables.proposal_shape

        # Sample k_on
        k_on_old = k_on
        k_on_new = stats.gamma.rvs(proposal_shape, scale=k_on_old/proposal_shape)
        P_old = RateInference.calculate_posterior(data, variables, k_on=k_on_old)
        P_new = RateInference.calculate_posterior(data, variables, k_on=k_on_new)
        acc_prob = (
            P_new - P_old
            + stats.gamma.logpdf(k_on_old, proposal_shape, scale=k_on_new/proposal_shape)
            - stats.gamma.logpdf(k_on_new, proposal_shape, scale=k_on_old/proposal_shape)
        )
        if acc_prob > np.log(np.random.rand()):
            k_on = k_on_new

        # Sample k_off
        k_off_old = k_off
        k_off_new = stats.gamma.rvs(proposal_shape, scale=k_off_old/proposal_shape)
        P_old = RateInference.calculate_posterior(data, variables, k_off=k_off_old)
        P_new = RateInference.calculate_posterior(data, variables, k_off=k_off_new)
        acc_prob = (
            P_new - P_old
            + stats.gamma.logpdf(k_off_old, proposal_shape, scale=k_off_new/proposal_shape)
            - stats.gamma.logpdf(k_off_new, proposal_shape, scale=k_off_old/proposal_shape)
        )
        if acc_prob > np.log(np.random.rand()):
            k_off = k_off_new

        # Sample k_photo
        k_photo_old = k_photo
        k_photo_new = stats.gamma.rvs(proposal_shape, scale=k_photo_old/proposal_shape)
        P_old = RateInference.calculate_posterior(data, variables, k_photo=k_photo_old)
        P_new = RateInference.calculate_posterior(data, variables, k_photo=k_photo_new)
        acc_prob = (
            P_new - P_old
            + stats.gamma.logpdf(k_photo_old, proposal_shape, scale=k_photo_new/proposal_shape)
            - stats.gamma.logpdf(k_photo_new, proposal_shape, scale=k_photo_old/proposal_shape)
        )
        if acc_prob > np.log(np.random.rand()):
            k_photo = k_photo_new

        # Update variables
        variables.k_on = k_on
        variables.k_off = k_off
        variables.k_photo = k_photo

        # Return variables
        return variables

    @staticmethod
    def plot_rates(MAP, Samples):

        # Set up figure
        fig = plt.gcf()
        fig.clf()
        plt.ion()
        plt.show()
        ax = np.empty(3, dtype=object)
        for i in range(3):
            ax[i] = fig.add_subplot(1, 3, i+1)

        # Select sample ids
        num_iterations = len(Samples["P"])
        last = [*np.where(Samples["P"] == 0)[0], num_iterations][0]
        burn = int(last/2)
        
        # Plot k_on
        ax[0].hist(Samples["k_on"][burn:last])
        ax[0].axvline(MAP.k_on, color="r")
        ax[0].set_title("K on")
        ax[0].set_xlabel("Rate (1/frames)")
        ax[0].set_ylabel("Frequency")

        # Plot k_off
        ax[1].hist(Samples["k_off"][burn:last])
        ax[1].axvline(MAP.k_off, color="r")
        ax[1].set_title("K off")
        ax[1].set_xlabel("Rate (1/frames)")
        ax[1].set_ylabel("Frequency")

        # Plot k_photo
        ax[2].hist(Samples["k_photo"][burn:last], label="Samples")
        ax[2].axvline(MAP.k_photo, color="r", label="MAP")
        ax[2].set_title("Bleach Rate")
        ax[2].set_xlabel("Rate (1/frames)")
        ax[2].set_ylabel("Frequency")

        # Show figure
        ax[-1].legend()
        plt.tight_layout()
        plt.pause(.01)
        return

    @staticmethod
    def analyze_data(data, parameters=None, num_iterations=10000, plot=False, verbose=True, **kwargs):

        # Initialize variables
        if verbose:
            print("Initializing variables...")
        variables = RateInference.initialize_parameters(data, parameters, **kwargs)
        MAPvariables = copy.deepcopy(variables)
        Samples = {
            "P": np.zeros(num_iterations),
            "k_on": np.zeros(num_iterations),
            "k_off": np.zeros(num_iterations),
            "k_photo": np.zeros(num_iterations),
        }

        # Gibbs sampling
        for iteration in range(num_iterations):
            if verbose:
                print(f"Iteration {iteration+1}/{num_iterations}", end='')
            t = time.time()
                
            # Sample variables
            variables = RateInference.sample_rates(data, variables)

            # Save samples
            variables.P = RateInference.calculate_posterior(data, variables)
            if variables.P >= MAPvariables.P:
                MAPvariables = copy.deepcopy(variables)
            Samples["P"][iteration] = variables.P
            Samples["k_on"][iteration] = variables.k_on
            Samples["k_off"][iteration] = variables.k_off
            Samples["k_photo"][iteration] = variables.k_photo

            # Plot
            if plot:
                RateInference.plot_rates(MAPvariables, Samples)
                plt.pause(.01)

            # Print
            if verbose:
                print(f" ({time.time()-t:.2f} s) P = {variables.P:.2f}")

        # Return MAP variables
        return MAPvariables, Samples



