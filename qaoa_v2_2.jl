# qaoa_v2_2.jl
using JuliQAOA
using Graphs
using Random
using Statistics
using Printf
using Dates

cd(@__DIR__) 
const TWO_PI = 2π

# -------------------------
# Graph / objective helpers
# -------------------------

"""Deterministic Erdos–Renyi graph generator"""
function erdos_renyi_graph(N::Int, p_edge::Float64; rng::AbstractRNG)
    g = SimpleGraph(N)
    @inbounds for u in 1:(N-1)
        for v in (u+1):N
            if rand(rng) < p_edge
                add_edge!(g, u, v)
            end
        end
    end
    return g
end

"""Computes the MaxCut objective value of assignment x on graph g"""
function maxcut_value(g::SimpleGraph, x)::Int
    c = 0
    @inbounds for e in edges(g)
        u = src(e); v = dst(e)
        if x[u] != x[v]
            c += 1
        end
    end
    return c
end

# -------------------------
# Mixer selection
# -------------------------

"""Return a JuliQAOA mixer by kind (:X or :Grover)."""
function get_mixer(N::Int, kind::Symbol)
    if kind == :X
        return mixer_x(N)
    elseif kind == :Grover
        if isdefined(JuliQAOA, :mixer_grover)
            return JuliQAOA.mixer_grover(N)
        end
        try
            return mixer_x(N, 0:N) / 2^N
        catch err
            error("Could not construct Grover mixer. Your JuliQAOA version may not support mixer_grover(n) or mixer_x(n, 0:n). Original error: $(err)")
        end
    else
        error("Unknown mixer kind: $kind. Use :X or :Grover")
    end
end

# -------------------------
# Problem construction
# -------------------------

"""Build objective lookup table `score_vals` aligned with `states(N)`.

Returns (score_vals, meta, states_vec).

- SK: score_vals = -energy (so maximizing score is minimizing energy).
- MaxCut: score_vals = cut value (maximize).
"""
function build_problem(N::Int, problem::Symbol;
                       p_edge::Float64 = 0.5,
                       instance_seed::Int = 1,
                       states_cache = nothing)

    states_vec = states_cache === nothing ? collect(states(N)) : states_cache

    if problem == :SK
        Random.seed!(instance_seed)
        J = sk_model(N)
        energy_vals = Float64[spin_energy(J, x) for x in states_vec]
        score_vals  = -energy_vals
        meta = (problem = :SK, sense = "minimize energy", instance_seed = instance_seed)
        return score_vals, meta, states_vec

    elseif problem == :MaxCut
        g = erdos_renyi_graph(N, p_edge; rng = MersenneTwister(instance_seed))
        cut_vals   = Float64[maxcut_value(g, x) for x in states_vec]
        score_vals = cut_vals
        meta = (problem = :MaxCut, sense = "maximize cut", instance_seed = instance_seed)
        return score_vals, meta, states_vec

    else
        error("Unknown problem: $problem. Use :SK or :MaxCut")
    end
end


"""Fill the first 2k of work with prefix schedule from `betas`, `gammas`."""
@inline function assemble_prefix!(work::Vector{Float64}, betas::Vector{Float64}, gammas::Vector{Float64}, k::Int)
    @inbounds for i in 1:k
        work[i]     = betas[i]
        work[k + i] = gammas[i]
    end
    return nothing
end

"""Run `reps` random schedules at p=max_p and reuse prefixes.

Monotonicity definition with tolerance tol:
  increasing if S_k > S_{k-1} + tol for all k=2..p
  decreasing if S_k < S_{k-1} - tol for all k=2..p
"""
function run_trials_prefix(max_p::Int,
                           reps::Int,
                           mixer::Mixer,
                           score_vals::AbstractVector;
                           rng::AbstractRNG,
                           tol::Float64 = 1e-12,
                           canonicalize::Bool = false)

    inc_counts = zeros(Int, max_p)
    dec_counts = zeros(Int, max_p)

    betas  = Vector{Float64}(undef, max_p)
    gammas = Vector{Float64}(undef, max_p)

    # work holds the assembled angles for a given k; exp_value requires a Vector (not a view)
    work = Vector{Float64}(undef, 2 * max_p)

    for _ in 1:reps
        rand!(rng, betas);  @. betas  *= TWO_PI
        rand!(rng, gammas); @. gammas *= TWO_PI

        # k=1 (Vector, length 2)
        work[1] = betas[1]
        work[2] = gammas[1]
        angles1 = work[1:2]  # slice -> Vector (required by exp_value dispatch)
        if canonicalize
            angles1 = clean_angles(angles1, mixer, score_vals)
        end
        prev_S = exp_value(angles1, mixer, score_vals)

        inc_ok = true
        dec_ok = true

        for k in 2:max_p
            assemble_prefix!(work, betas, gammas, k)

            anglesk = work[1:(2k)]  # slice -> Vector
            if canonicalize
                anglesk = clean_angles(anglesk, mixer, score_vals)
            end
            S = exp_value(anglesk, mixer, score_vals)

            if S <= prev_S + tol
                inc_ok = false
            end
            if S >= prev_S - tol
                dec_ok = false
            end

            if inc_ok
                inc_counts[k] += 1
            elseif dec_ok
                dec_counts[k] += 1
            end

            if !inc_ok && !dec_ok
                break  # cannot recover monotonicity once broken
            end

            prev_S = S
        end
    end

    return inc_counts, dec_counts
end

"""Convert counts -> proportions + (within-instance) binomial standard error."""
function counts_to_props(counts::Vector{Int}, reps::Int)
    props = Float64[c / reps for c in counts]
    se    = Float64[sqrt(p * (1 - p) / reps) for p in props]
    return props, se
end

# -------------------------
# CSV output
# -------------------------

function write_header(io)
    println(io, join([
        "problem",
        "mixer",
        "N",
        "p",
        "instance_seed",
        "angle_seed",
        "reps",
        "tol",
        "canonicalize",
        "inc_count",
        "dec_count",
        "other_count",
        "inc_prop",
        "dec_prop",
        "inc_se_binom",
        "dec_se_binom",
    ], ","))
end

function write_row(io; problem, mixer, N, p, instance_seed, angle_seed, reps, tol, canonicalize,
                   inc_count, dec_count, other_count, inc_prop, dec_prop, inc_se_binom, dec_se_binom)

    @printf(io,
        "%s,%s,%d,%d,%d,%d,%d,%.3e,%s,%d,%d,%d,%.17g,%.17g,%.17g,%.17g\n",
        String(problem), String(mixer), N, p, instance_seed, angle_seed, reps, tol,
        canonicalize ? "true" : "false",
        inc_count, dec_count, other_count,
        inc_prop, dec_prop, inc_se_binom, dec_se_binom
    )
end

# -------------------------
# Main configuration
# -------------------------

# --- experiment knobs ---
N              = 10
max_p          = 20
reps           = 10_000_000
p_edge         = 0.5
instance_seeds = collect(1:25)

mixers   = [:X, :Grover]
problems = [:SK, :MaxCut]

tol          = 1e-12
CANONICALIZE = false

SHARE_ANGLES_ACROSS_MIXERS = true
ANGLE_SEED_BASE = 20260227

# --- output ---
outdir = "results"
mkpath(outdir)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
raw_csv = joinpath(outdir, "monotonicity_raw_N$(N)_pmax$(max_p)_reps$(reps)_$(timestamp).csv")
println("Writing raw results to: $raw_csv")


states_cache = collect(states(N))

open(raw_csv, "w") do io
    write_header(io)

    for problem in problems
        for instance_seed in instance_seeds
            score_vals, meta, _ = build_problem(N, problem;
                p_edge = p_edge,
                instance_seed = instance_seed,
                states_cache = states_cache
            )

            println("\nProblem=$(problem) ($(meta.sense)) | instance_seed=$(instance_seed)")

            problem_id = problem == :SK ? 1 : 2
            base_angle_seed = ANGLE_SEED_BASE + 1_000_000 * problem_id + instance_seed

            for mixer_kind in mixers
                mixer_obj = get_mixer(N, mixer_kind)

                mixer_id  = mixer_kind == :X ? 1 : 2
                angle_seed = SHARE_ANGLES_ACROSS_MIXERS ? base_angle_seed : (base_angle_seed + 10_000 * mixer_id)
                rng = MersenneTwister(angle_seed)

                t0 = time()
                inc_counts, dec_counts = run_trials_prefix(max_p, reps, mixer_obj, score_vals;
                    rng = rng,
                    tol = tol,
                    canonicalize = CANONICALIZE
                )
                elapsed = time() - t0

                inc_props, inc_se = counts_to_props(inc_counts, reps)
                dec_props, dec_se = counts_to_props(dec_counts, reps)

                for p in 2:max_p
                    inc_c = inc_counts[p]
                    dec_c = dec_counts[p]
                    other_c = reps - inc_c - dec_c

                    write_row(io;
                        problem = problem,
                        mixer = mixer_kind,
                        N = N,
                        p = p,
                        instance_seed = instance_seed,
                        angle_seed = angle_seed,
                        reps = reps,
                        tol = tol,
                        canonicalize = CANONICALIZE,
                        inc_count = inc_c,
                        dec_count = dec_c,
                        other_count = other_c,
                        inc_prop = inc_props[p],
                        dec_prop = dec_props[p],
                        inc_se_binom = inc_se[p],
                        dec_se_binom = dec_se[p],
                    )
                end

                println(@sprintf("  mixer=%s | angle_seed=%d | done in %.2fs", String(mixer_kind), angle_seed, elapsed))
                println(@sprintf("    p=%d: inc=%.6g, dec=%.6g", max_p, inc_props[max_p], dec_props[max_p]))
            end
        end
    end
end

println("\nDone. Next step: run fit_monotonicity.py on the CSV in results/.")
