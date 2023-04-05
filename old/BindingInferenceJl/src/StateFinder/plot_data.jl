
function plot_data(data, variables=nothing; plot_options=Dict([]), kwargs...)

    # Set up plot plot_options
    plot_options = merge(
        Dict([
            ("fig", nothing),
            ("times", nothing),
            ("groundtruth", nothing),
        ]),
        plot_options,
        Dict([(string(key), val) for (key, val) in kwargs])
    )
    groundtruth = plot_options["groundtruth"]
    times = plot_options["times"]

    # Set up constants
    num_frames = size(data, 1)
    if times === nothing
        dt = 1
        if variables !== nothing
            dt = variables.dt
        elseif groundtruth !== nothing
            dt = groundtruth.dt
        end
        times = dt .* (1:num_frames)
    end


    # Set up figure
    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    plt.ion()
    plt.show()

    # Plot data
    ax.plot(times, data, color="green", label="Data")

    # Plot variables
    if variables !== nothing
        # Set up variables
        variables = SimpleNamespace(
            merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
        )
        mu_flor = variables.mu_flor
        mu_back = variables.mu_back
        states = variables.states
        gain = variables.gain

        # Set up trace
        trace = zeros(Float64, num_frames)
        for n in 1:num_frames
            trace[n] = (mu_flor*(states[n]-1) + mu_back) * gain
        end

        # Plot
        ax.plot(times, trace, color="blue", label="Estimate")

    end

    # Plot ground truth
    if groundtruth !== nothing
        # Set up ground truth
        gt = SimpleNamespace(groundtruth)
        mu_flor = gt.mu_flor
        mu_back = gt.mu_back
        states = gt.states
        gain = gt.gain

        # Set up trace
        trace = zeros(Float64, num_frames)
        for n in 1:num_frames
            trace[n] = (mu_flor*(states[n]-1) + mu_back) * gain
        end
    end
    # Tighten plot 
    ax.set_ylabel("Brightness (ADU)")
    ax.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.pause(.1)

    # Return nothing
    return nothing

end # function plot_variables
