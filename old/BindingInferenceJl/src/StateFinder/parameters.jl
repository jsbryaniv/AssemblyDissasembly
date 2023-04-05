
PARAMETERS = Dict{String, Any}([

    # Variables
    ("P", nothing),                      # (log)  Probability
    ("pi", nothing),                     # (#)    Microstate transition rates matrix
    ("gain", nothing),                   # (#)    Gain
    ("states", nothing),                 # (#)    State trajectory
    ("mu_flor", nothing),                # (ADU)  Brightness of fluorophore microstates
    ("mu_back", nothing),                # (ADU)  Brightness of background
         
    # Experiment         
    ("dt", 1),                           # (s)    Time between frames
    ("num_frames", nothing),             # (#)    Number of frames
    ("num_states", 20),                  # (#)    Number of states

    # Hyperparameters
    ("gain_shape", 2),                   # Shape hyperparamter for gain
    ("gain_scale", nothing),             # Scale hyperparamter for gain
    ("mu_flor_shape", 2),                # Shape hyperparamter for fluorophore brightness
    ("mu_flor_scale", nothing),          # Scale hyperparamter for fluorophore brightness
    ("mu_back_shape", 2),                # Shape hyperparamter for background brightness
    ("mu_back_scale", nothing),          # Scale hyperparamter for background brightness
    ("pi_concentration", nothing),       # Conentration hyperparamter for microstate transition rates

    # Sampler parameters
    ("seed", 0),                         # RNG seed
    ("gain_prop_shape", 100),            # Shape parameter for gain proposals
    ("mu_flor_prop_shape", 100),         # Shape parameter for photorate proposals
    ("mu_back_prop_shape", 100),         # Shape parameter for binding rate proposals
])

