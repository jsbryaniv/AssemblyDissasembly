
# Import standard modules
import os
import sys
import h5py
import copy
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Declare parameters
PARAMETERS = {
    # Variables
    "s": None,
    "pi": None,
    "mu_flor": None,
    "mu_back": None,
    "P": None,
    # Priors
    "pi_conc": None,
    "mu_flor_shape": None,
    "mu_flor_scale": None,
    "mu_back_shape": None,
    "mu_back_scale": None,
    # Experiment
    "dt": 1,
    "num_states": 20,
    "num_frames": None,
}


# Declare class
class BindingInference:
    
    @staticmethod
    def simulate_data():
        pass

    @staticmethod
    def initialize_parameters(data, parameters=None, **kwargs):
        
        # Set up parameters
        if parameters is None:
            parameters = {}
        parameters = copy.deepcopy(parameters)
        parameters = {**parameters, **kwargs}

        # Set up variables
        variables = SimpleNamespace(**parameters)
        s = variables.s 
        pi = variables.pi
        pi_conc = variables.pi_conc
        mu_flor = variables.mu_flor
        mu_flor_shape = variables.mu_flor_shape
        mu_flor_scale = variables.mu_flor_scale
        mu_back = variables.mu_back
        mu_back_shape = variables.mu_back_shape
        mu_back_scale = variables.mu_back_scale
        dt = variables.dt
        num_states = variables.num_states
        num_frames = variables.num_frames


        # Initialize constants
        num_frames = data.shape[0]
        variables.num_frames = num_frames

        # Initialize states
        s = np.zeros(num_frames, dtype=int)
        variables.s = s

        # Initialize transition probabilities
        if pi_conc is None:
            pi_conc = np.ones((num_states+1, num_states))
            pi_conc[:-1, :] += num_frames / 10 * np.eye(num_states)
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
            mu_back_scale = np.mean(data)
        if mu_back is None:
            mu_back = mu_back_shape * mu_back_scale
        variables.mu_back = mu_back
        variables.mu_back_shape = mu_back_shape
        variables.mu_back_scale = mu_back_scale

        # Initialize fluorophore brightness
        if mu_flor_shape is None:
            mu_flor_shape = 2
        if (mu_flor_scale is None) and (data is not None):
            mu_flor_scale = np.std(data)
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


    @staticmethod
    def sample_transitions(data, variables):
        pass

    @staticmethod
    def sample_brightness(data, variables):
        pass

    @staticmethod
    def calculate_posterior(data, variables):
        pass

    @staticmethod
    def analyze(data, parameters=None, num_iterations=1000, **kwargs):

        # Initialize variables
        variables = BindingInference.initialize_parameters(data, parameters, **kwargs)
        MAPvariables = copy.deepcopy(variables)

        # Gibbs sampling
        for iteration in range(num_iterations):
                
            # Sample states
            variables = BindingInference.sample_brightness(data, variables)
            variables = BindingInference.sample_transitions(data, variables)
            variables = BindingInference.sample_states(data, variables)

            # Update MAP
            variables.P = BindingInference.calculate_posterior(data, variables)
            if variables.P >= MAPvariables.P:
                MAPvariables = copy.deepcopy(variables)

        # Return MAP variables
        return MAPvariables

print("Done")




