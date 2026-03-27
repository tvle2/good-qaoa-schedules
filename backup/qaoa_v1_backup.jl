using JuliQAOA
using Graphs
using Plots
using Random


"""
    # Arguments
    - p::Int: The depth of QAOA.
    - angles::Vector{Float64}: [β_1, ..., β_p, γ_1, ..., γ_p].
    - mixer::Mixer: The mixing Hamiltonian operator (e.g., standard X mixer or Grover mixer).
    - score_vals::AbstractVector:
    # Returns
    - 1: If the expectation value is strictly increasing (improving) with every added layer.
    - 2: If the expectation value is strictly decreasing (worsening) with every added layer.
    - 3: If the expectation value fluctuates (breaks strict monotonicity in either direction).
"""
function categorize_schedule(p::Int, angles::Vector{Float64}, mixer::Mixer, score_vals::AbstractVector)
    prev_S = exp_value([angles[1], angles[p+1]], mixer, score_vals)

    is_inc = true
    is_dec = true

    for k in 2:p
        prefix_angles = vcat(angles[1:k], angles[p+1:p+k])
        S = exp_value(prefix_angles, mixer, score_vals)

        # strict monotonicity
        if S <= prev_S
            is_inc = false
        end
        if S >= prev_S
            is_dec = false
        end

        if !is_inc && !is_dec
            return 3
        end

        prev_S = S
    end

    return is_inc ? 1 : (is_dec ? 2 : 3)
end

function build_problem(N::Int, problem_type::String; p_edge::Float64 = 0.5, seed::Int = 1)
    Random.seed!(seed)

    if problem_type == "SK"
        J = sk_model(N)
        energy_vals = Float64[spin_energy(J, x) for x in states(N)]
        score_vals  = -energy_vals  # maximize negative energy => minimize energy
        meta = (problem = "SK", sense = "minimize energy", instance = J)
        return score_vals, meta

    elseif problem_type == "MaxCut"
        g = erdos_renyi(N, p_edge)
        cut_vals  = Float64[maxcut(g, x) for x in states(N)]
        score_vals = cut_vals        # maximize cut directly
        meta = (problem = "MaxCut", sense = "maximize cut", instance = g)
        return score_vals, meta

    else
        error("Unknown problem type. Use \"SK\" or \"MaxCut\"")
    end
end


function run_experiment(N::Int, max_p::Int, reps::Int, mixer::Mixer, score_vals::AbstractVector)
    prop_inc = zeros(Float64, max_p)
    prop_dec = zeros(Float64, max_p)

    for p in 2:max_p
        inc_count = 0
        dec_count = 0

        for _ in 1:reps
            angles = 2π * rand(2p)
            angles = clean_angles(angles, mixer, score_vals)

            category = categorize_schedule(p, angles, mixer, score_vals)
            if category == 1
                inc_count += 1
            elseif category == 2
                dec_count += 1
            end
        end

        prop_inc[p] = inc_count / reps
        prop_dec[p] = dec_count / reps
        println("  p=$p -> Improving: $(prop_inc[p]), Worsening: $(prop_dec[p])")
    end

    return prop_inc, prop_dec
end

N = 5
max_p = 20
reps = 1_000_000
seed = 42        
p_edge = 0.5

problem_types = ["SK", "MaxCut"]
mixer_types   = ["X", "Grover"]
p_range = 2:max_p

for prob in problem_types
    score_vals, meta = build_problem(N, prob; p_edge=p_edge, seed=seed)
    println("\nProblem=$prob ($(meta.sense)), seed=$seed")

    for mix in mixer_types
        mixer = mix == "X" ? mixer_x(N) :
                mix == "Grover" ? mixer_grover(N) :
                error("Unknown mixer type")

        println("Running mixer=$mix ...")

        inc_props, dec_props = run_experiment(N, max_p, reps, mixer, score_vals)

        # Plot results (now interpretable across problems)
        p_plot = plot(p_range, inc_props[2:end],
                      label="Strictly Improving",
                      marker=:circle,
                      xlabel="p (QAOA rounds)",
                      ylabel="Proportion",
                      title="$prob, $mix Mixer (N=$N, seed=$seed)",
                      lw=2)
        plot!(p_plot, p_range, dec_props[2:end],
              label="Strictly Worsening",
              marker=:square,
              lw=2)

        display(p_plot)
        savefig(p_plot, "Monotonicity_$(prob)_$(mix)_N$(N)_seed$(seed).png")
    end
end
