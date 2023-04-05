
function plot_rates(samplecache; plot_options=Dict([]), kwargs...)

    # Set up plot plot_options
    plot_options = merge(
        Dict([
            ("fig", nothing),
            ("groundtruth", nothing),
            ("bins", 100),
        ]),
        plot_options,
        Dict([(string(key), val) for (key, val) in kwargs])
    )
    fig = plot_options["fig"]
    groundtruth = plot_options["groundtruth"]
    bins = plot_options["bins"]
    if groundtruth !== nothing
        gt = SimpleNamespace(groundtruth)
    end

    # Get constants
    k_bind_hist = samplecache.get("k_bind")
    k_micro_hist = samplecache.get("k_micro")
    variables = SimpleNamespace(samplecache.get("map"))
    dt = variables.dt
    mu_micro = variables.mu_micro
    mu_back = variables.mu_back
    partitions = variables.partitions
    macrostates = variables.macrostates
    num_bound = variables.num_bound
    num_frames = variables.num_frames
    laser_power = variables.laser_power
    k_bind = variables.k_bind
    k_micro = variables.k_micro

    # Set up figure
    if fig === nothing
        fig = plt.gcf()
    end
    fig.clf()
    ax = Array{Any}(undef, 1, 2)
    ax[1, 1] = fig.add_subplot(size(ax)..., 1)
    for i = 2:size(ax, 2)
        ax[1, i] = fig.add_subplot(size(ax)..., i, )  # sharex=ax[1, 1], sharey=ax[1, 1])
    end
    plt.show()
        
    # Plot bind rate
    ax[1, 1].set_title("Binding rate")
    ax[1, 1].set_ylabel("Probability")
    ax[1, 1].set_xlabel("Rate (s⁻¹)")
    ax[1, 1].hist((k_bind_hist[:, 1, 2]), bins=bins, density=true)

    # Plot unbind rate
    ax[1, 2].set_title("Unbinding rate")
    ax[1, 2].set_ylabel("Probability")
    ax[1, 2].set_xlabel("Rate (s⁻¹)")
    ax[1, 2].hist((k_bind_hist[:, 2, 1]), bins=bins, density=true)

    # Plot ground truth
    if groundtruth !== nothing
        ax[1, 1].axvline(gt.k_bind[1, 2], color="red", label="Ground truth")
        ax[1, 2].axvline(gt.k_bind[2, 1], color="red", label="Ground truth")
    end

    # Tighten plot
    plt.tight_layout()
    plt.pause(.1)

    # Return nothing
    return nothing

end # function plot_variables
