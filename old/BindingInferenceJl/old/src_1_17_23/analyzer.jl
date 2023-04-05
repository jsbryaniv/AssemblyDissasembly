
function analyze(data, parameters=Dict(); num_iterations=1000, 
    plot=true, plot_options=Dict(), 
    saveas=nothing, kwargs...)

    # Print status
    println("\n---------------------------------")
    println("Starting  analysis...")

    # Set up parameters
    println("Setting up parameters.")
    parameters = Dict([
        PARAMETERS...,
        [(string(key), val) for (key, val) in pairs(parameters)]...,
        [(string(key), val) for (key, val) in pairs(kwargs)]...,
    ])

    # Set up variables
    variables = initialize_variables(data, parameters)
    println("Variables initialized with:")
    for (key, val) in pairs(sort(Dict(variables)))
        str = "---" * (string(key) * " "^20)[1:20]
        str = (str * " :: $(repr(val))")
        # str = (str * " :: $(typeof(val))")
        # if hasmethod(size, Tuple{typeof(val)}) 
        #     length(size(val)) > 0 ? str = str * "$(size(val))" : nothing
        # end
        str = str[1:minimum((end, displaysize(stdout)[2]))]
        println(str)
    end

    # Set up output
    println("Setting up output.")
    map_variables = copy(variables)
    if saveas !== nothing
        # samplecache = SampleCache(
        #     variables, num_iterations, saveas,
        #     fields_to_save=[
        #         "mu_photo",
        #         "mu_back",
        #         "k_photo",
        #         "k_bind",
        #         "P",
        #     ],
        # )
    end

    # Run the Gibbs sampler
    for iteration = 1:num_iterations
        start_time = time()
        print("Iteration $(iteration) of $(num_iterations) [")

        # Sample variables
        variables = sample_macrostates(data, variables)
        print('%')
        variables = sample_brightness(data, variables)
        print('%')
        variables = sample_rates(data, variables)
        print('%')
        variables = sample_brightness_and_macrostates(data, variables)
        print('%')

        # Set probability and check for MAP
        variables.P = calculate_posterior(data, variables)
        isMAP = false
        if variables.P >= map_variables.P
            isMAP = true
            map_variables = variables
        end
        print('%')

        # Save variables
        if saveas !== nothing
            # samplecache.update(variables, iteration, isMAP=isMAP)
            # print('%')
        end

        # Plot variables
        if plot > 0
            if (iteration==1) | (mod(iteration, plot)==0) | (iteration == num_iterations)
                plot_variables(data, variables, plot_options=plot_options)
            end
            print('%')
        end
        
        println("] ($(round(time() - start_time, digits=2)) s) (nbound=$(round(mean(variables.num_bound), digits=2))) (prob=$(@sprintf("%.3e", variables.P)))")
    end # for iteration

    # Print status
    println("...Analysis complete.")

    # Return output
    if saveas !== nothing
        return map_variables  # , samplecache
    else
        return map_variables
    end
end # function analyze