
PARAMETERS = Dict{String, Any}([

    # Variables
    ("macrostates", nothing),  # (#)    Macrostate trajectory
    ("mu_micro", nothing),     # (ADU)  Brightness of fluorophore microstates
    ("mu_back", nothing),      # (ADU)  Brightness of background
    ("sigma_micro", nothing),  # (ADU)  Noise of fluorophores (SCMOS only)
    ("sigma_back", nothing),   # (ADU)  Noise of background (SCMOS only)
    ("k_micro", nothing),      # (1/ns) Microstate transition rates matrix
    ("k_bind", nothing),       # (1/ns) Binding rate matrix
    ("num_bound", nothing),    # (#)    Highest number of fluorophors bound in each ROI
    ("P", nothing),            # (log)  Probability

    # Constants
    ("dt", 1),                   # (ns)  Time step in seconds
    ("laser_power", 1),          # (mW)  Mask of laser intensities for each ROI  
    ("concentration", 1),        # (pM)  Mask of concentrations for each ROI    
    ("partitions", nothing),     #       State ID to population mapping
    ("degenerate_ids", nothing), #       Array of arrays linking degenerate brightnesses

    # Hyperparameters
    ("mu_micro_mean", nothing),      # Mean hyperparameter on photostate brightness
    ("mu_micro_std", nothing),       # STD hyperparameter on photostate brightness
    ("mu_back_mean", nothing),       # Mean hyperparameter on background brightness
    ("mu_back_std", nothing),        # STD hyperparameter on background brightness
    ("sigma_micro_shape", 2),        # Shape hyperparameter on photostate brightness
    ("sigma_micro_scale", nothing),  # Scale hyperparameter on photostate brightness
    ("sigma_back_shape", 2),         # Shape hyperparameter on background brightness
    ("sigma_back_scale", nothing),   # Scale hyperparameter on background brightness
    ("k_micro_shape", 2),            # Shape hyperparameter on photostate transition rates
    ("k_micro_scale", nothing),      # Scale hyperparameter on photostate transition rates
    ("k_bind_shape", 2),             # Shape hyperparameter on binding transition rates
    ("k_bind_scale", nothing),       # Scale hyperparameter on binding transition rates

    # Numbers
    ("num_rois", nothing),      # Number of ROIs
    ("num_frames", nothing),    # Number of time levels
    ("num_micro", 3),           # Number of photostates
    ("num_macro", nothing),     # Number of macro states
    ("num_max", 10),            # Number of maximum allowed fluorophres
    ("num_unique", nothing),    # Number of unique brightnesses

    # Sampler parameters
    ("seed", 0),                         # RNG seed
    ("flor_brightness_guess", nothing),  # Guess for fluorophore brightness
    ("k_micro_prop_shape", 100),         # Shape parameter for photorate proposals
    ("k_bind_prop_shape", 100),          # Shape parameter for binding rate proposals
    ("sigma_micro_prop_shape", 100),     # Shape parameter for photostate noise proposals
    ("sigma_back_prop_shape", 100),      # Shape parameter for background noise proposals
])
