
function plot_variables(data, variables; plot_options=Dict([]), kwargs...)

    # Set up plot plot_options
    plot_options = merge(
        Dict([
            ("rois", 1:minimum((3, size(data, 1)))),
            ("fig", nothing),
            ("times", nothing),
            ("ground_truth", nothing),
        ]),
        plot_options,
        Dict([(string(key), val) for (key, val) in kwargs])
    )
    rois = plot_options["rois"]
    times = plot_options["times"]
    ground_truth = plot_options["ground_truth"]
    if isa(rois, Number)
        rois = [rois]
    end

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )
    if ground_truth !== nothing
        gt = SimpleNamespace(ground_truth)
    end

    # Get constants
    dt = variables.dt
    gain = variables.gain
    mu_photo = variables.mu_photo
    mu_back = variables.mu_back
    partitions = variables.partitions
    macrostates = variables.macrostates
    num_bound = variables.num_bound
    num_data = variables.num_data
    laser_power = variables.laser_power
    cameramodel = variables.cameramodel

    # Calculate values
    if times === nothing
        times = dt .* (1:num_data)
    end

    # Set up figure
    fig = plt.gcf()
    fig.clf()
    ax = Array{Any}(undef, 1, length(rois))
    ax[1, 1] = fig.add_subplot(1, length(rois), 1)
    for i = 2:length(rois)
        ax[1, i] = fig[:add_subplot](1, length(rois), i, sharex=ax[1], sharey=ax[1])
    end
    plt.ion()
    plt.show()

    # Loop through rois
    for (i, r) in enumerate(rois)
        
        # Plot data
        ax[1, i].plot(times, data[r, :], color="green", label="Data")

        # Plot variables
        mu_macro = (partitions * mu_photo) .* laser_power[r] .+ mu_back[r]
        trace = zeros(Float64, num_data)
        for n in 1:num_data
            s = macrostates[r, n]
            trace[n] = mu_macro[s]
        end
        if cameramodel == "emccd"
            trace .*= gain
        end
        ax[1, i].plot(times, trace, color="blue", label="Variables")

        # Plot ground truth
        if ground_truth !== nothing
            gt_trace = zeros(Float64, num_data)
            for n in 1:num_data
                s = gt.macrostates[r, n]
                mu = (
                    sum(gt.mu_photo .* gt.partitions[s, :]) * gt.laser_power[r] 
                    + gt.mu_back[r]
                )
                if cameramodel == "emccd"
                    gt_trace[n] = mu * gt.gain
                end
            end
            ax[1, i].plot(times, gt_trace, color="red", label="Ground truth")
        end

        # Set up labels
        ax[1, i].set_title("ROI $r\n$(num_bound[r]) bound")
        ax[1, i].set_xlabel("Time (s)")
        if i == 1
            ax[1, i].set_ylabel("Brightness (ADU)")
        end
        if i == length(rois)
            ax[1, i].legend()
        end
    end

    # Tighten plot
    plt.tight_layout()
    plt.pause(.1)

    # Return nothing
    return nothing

end # function plot_variables
