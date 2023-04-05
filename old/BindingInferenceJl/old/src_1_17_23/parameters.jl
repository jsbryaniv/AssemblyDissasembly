
PARAMETERS = Dict{String, Any}([

    # Variables
    ("macrostates", nothing),  # (#)    Macrostate trajectory
    ("mu_photo", nothing),     # (ADU)  Brightness of fluorophore microstates
    ("mu_back", nothing),      # (ADU)  Brightness of background
    ("sigma_photo", nothing),  # (ADU)  Noise of fluorophores (SCMOS only)
    ("sigma_back", nothing),   # (ADU)  Noise of background (SCMOS only)
    ("k_photo", nothing),      # (1/ns) Microstate transition rates matrix
    ("k_bind", nothing),       # (1/ns) Binding rate matrix
    ("num_bound", nothing),    # (#)    Highest number of fluorophors bound in each ROI
    ("P", nothing),            # (log)  Probability

    # Constants
    ("dt", 1),                   # (ns)  Time step in seconds
    ("gain", nothing),           # (ADU) Gain (REQUIRED FOR EMCCD)
    ("laser_power", 1),          # (mW)  Mask of laser intensities for each ROI  
    ("concentration", 1),        # (pM)  Mask of concentrations for each ROI    
    ("partitions", nothing),     #       State ID to population mapping
    ("degenerate_ids", nothing), #       Array of arrays linking degenerate brightnesses
    ("cameramodel", "scmos"),    #       Noise model

    # Priors for EMCCD noise parameters
    ("mu_photo_shape", 2),           # Shape hyperparameter on photostate brightness
    ("mu_photo_scale", nothing),     # Scale hyperparameter on photostate brightness
    ("mu_back_shape", 10),           # Shape hyperparameter on background brightness
    ("mu_back_scale", nothing),      # Scale hyperparameter on background brightness

    # Priors for SCMOS
    ("mu_photo_mean", nothing),      # Mean hyperparameter on photostate brightness
    ("mu_photo_vars", nothing),      # Variance hyperparameter on photostate brightness
    ("mu_back_mean", nothing),       # Mean hyperparameter on background brightness
    ("mu_back_vars", nothing),       # Variance hyperparameter on background brightness
    ("sigma_photo_shape", 2),        # Shape hyperparameter on photostate brightness
    ("sigma_photo_scale", nothing),  # Scale hyperparameter on photostate brightness
    ("sigma_back_shape", 2),         # Shape hyperparameter on background brightness
    ("sigma_back_scale", nothing),   # Scale hyperparameter on background brightness

    # Priors for rates
    ("k_photo_shape", 2),            # Shape hyperparameter on photostate transition rates
    ("k_photo_scale", nothing),      # Scale hyperparameter on photostate transition rates
    ("k_bind_shape", 2),             # Shape hyperparameter on binding transition rates
    ("k_bind_scale", nothing),       # Scale hyperparameter on binding transition rates

    # Numbers
    ("num_rois", nothing),      # Number of ROIs
    ("num_data", nothing),      # Number of time levels
    ("num_photo", 3),           # Number of photostates
    ("num_macro", nothing),     # Number of macro states
    ("num_max", 25),            # Number of maximum allowed fluorophres
    ("num_unique", nothing),    # Number of unique brightnesses

    # Sampler parameters
    ("seed", 0),                         # RNG seed
    ("flor_brightness_guess", nothing),  # Guess for fluorophore brightness
    ("background_times", nothing),       # Times at which background is measured
    ("k_photo_prop_shape", 100),         # Shape parameter for photorate proposals
    ("k_bind_prop_shape", 100),          # Shape parameter for binding rate proposals
    ("sigma_photo_prop_shape", 100),     # Shape parameter for photostate noise proposals
    ("sigma_back_prop_shape", 100),      # Shape parameter for background noise proposals
])