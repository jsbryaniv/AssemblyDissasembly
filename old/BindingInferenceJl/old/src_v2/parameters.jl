
PARAMETERS = Dict{String, Any}([

    # Variables
    ("macrostates", nothing),  # (#)    Macrostate trajectory
    ("mu_photo", nothing),     # (ADU)  Brightness of fluorophore microstates
    ("mu_back", nothing),      # (ADU)  Brightness of background
    ("sigma_photo", nothing),  # (ADU)  Noise of fluorophores
    ("sigma_back", nothing),   # (ADU)  Noise of background
    ("k_photo", nothing),      # (1/s)  Microstate transition rates matrix
    ("num_bound", nothing),    # (#)    Highest number of fluorophors bound in each ROI
    ("P", nothing),            # (log)  Probability

    # Constants
    ("dt", 1),                   # (s)   Time step
    ("partitions", nothing),     #       State ID to population mapping
    ("degenerate_ids", nothing), #       Array of arrays linking degenerate brightnesses

    # Priors for brightness
    ("mu_photo_mean", nothing),      # Mean hyperparameter on photostate brightness
    ("mu_photo_std", nothing),       # STD hyperparameter on photostate brightness
    ("mu_back_mean", nothing),       # Mean hyperparameter on background brightness
    ("mu_back_std", nothing),        # STD hyperparameter on background brightness
    ("sigma_photo_shape", 2),        # Shape hyperparameter on photostate brightness
    ("sigma_photo_scale", nothing),  # Scale hyperparameter on photostate brightness
    ("sigma_back_shape", 2),         # Shape hyperparameter on background brightness
    ("sigma_back_scale", nothing),   # Scale hyperparameter on background brightness

    # Priors for rates
    ("k_photo_shape", 2),            # Shape hyperparameter on photostate transition rates
    ("k_photo_scale", nothing),      # Scale hyperparameter on photostate transition rates

    # Numbers
    ("num_rois", nothing),      # Number of ROIs
    ("num_frames", nothing),    # Number of time levels
    ("num_photo", 2),           # Number of photostates
    ("num_macro", nothing),     # Number of macro states
    ("num_max", 15),            # Number of maximum allowed fluorophores
    ("num_unique", nothing),    # Number of unique brightnesses

    # Sampler parameters
    ("seed", 0),                         # RNG seed
    ("flor_brightness_guess", nothing),  # Guess for fluorophore brightness
    ("background_times", nothing),       # Times at which background is measured
    ("k_photo_prop_shape", 100),         # Shape parameter for photorate proposals
    ("sigma_photo_prop_shape", 100),     # Shape parameter for photostate noise proposals
    ("sigma_back_prop_shape", 100),      # Shape parameter for background noise proposals
])