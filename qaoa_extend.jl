using JuliQAOA
using Graphs
using Random
using Statistics
using Printf
using Dates

cd(@__DIR__)
const TWO_PI = 2π

# ============================================================
# Helpers
# ============================================================

function erdos_renyi_graph(N::Int, p_edge::Float64; rng::AbstractRNG)
    """
    # Arguments
    - `N::Int`: Number of vertices.
    - `p_edge::Float64`: Edge inclusion probability.
    - `rng::AbstractRNG`: Random number generator used for reproducible sampling.

    # Returns
    - `SimpleGraph`: A random graph sampled from `G(N, p_edge)`.

    # Notes
    This helper is used to generate random MaxCut and MaxClique benchmark
    instances. Using an explicit `rng` makes the graph generation reproducible
    without mutating Julia's global RNG state.
    """

    g = SimpleGraph(N)
    @inbounds for u in 1:(N - 1)
        for v in (u + 1):N
            if rand(rng) < p_edge
                add_edge!(g, u, v)
            end
        end
    end
    return g
end


function add_term!(H::Dict{Tuple,Float64}, term::Tuple, coeff::Real)
    """
    Convert a binary bit `b ∈ {0,1}` into an Ising spin in `{+1,-1}` using the
    convention `0 ↦ +1` and `1 ↦ -1`.

    # Arguments
    - `b::Integer`: Binary value.

    # Returns
    - `Float64`: `+1.0` if `b == 0`, otherwise `-1.0`.
    """
    @inline bit_to_spin(b::Integer) = b == 0 ? 1.0 : -1.0

    """
        add_term!(H, term, coeff)

    Accumulate a coefficient into a tuple-keyed Hamiltonian dictionary.

    # Arguments
    - `H::Dict{Tuple,Float64}`: Dictionary mapping index tuples to coefficients.
    - `term::Tuple`: Term key, for example `(i,)` for a linear term or `(i,j)` for
    a quadratic term.
    - `coeff::Real`: Coefficient to add.

    # Returns
    - The same dictionary `H`, updated in place.
    """
    
    c = Float64(coeff)
    if abs(c) < 1e-14
        return H
    end
    H[term] = get(H, term, 0.0) + c
    if abs(H[term]) < 1e-14
        delete!(H, term)
    end
    return H
end

function qubo_to_ising_local(Q::Dict{Tuple{Int,Int},Float64}; offset::Float64 = 0.0)
    """
    Convert a QUBO objective
    `Q(x) = ∑_{i≤j} Q[(i,j)] x_i x_j`, with `x_i ∈ {0,1}`,
    into an equivalent Ising objective

    `E(s) = offset + ∑_i h_i s_i + ∑_{i<j} J_{ij} s_i s_j`, with `s_i ∈ {+1,-1}`,

    under the variable transformation `x_i = (1 - s_i)/2`.

    # Arguments
    - `Q::Dict{Tuple{Int,Int},Float64}`: QUBO coefficients.
    - `offset::Float64=0.0`: Optional additive constant.

    # Returns
    A tuple `(h, J, off)` where:
    - `h::Dict{Int,Float64}` contains linear Ising fields,
    - `J::Dict{Tuple{Int,Int},Float64}` contains pairwise Ising couplings,
    - `off::Float64` is the constant energy shift.
    """
    h = Dict{Int,Float64}()
    J = Dict{Tuple{Int,Int},Float64}()
    off = offset

    for ((u, v), q) in Q
        if u == v
            h[u] = get(h, u, 0.0) - q / 2
            off += q / 2
        else
            i, j = u < v ? (u, v) : (v, u)
            J[(i, j)] = get(J, (i, j), 0.0) + q / 4
            h[i] = get(h, i, 0.0) - q / 4
            h[j] = get(h, j, 0.0) - q / 4
            off += q / 4
        end
    end

    return h, J, off
end

function pairwise_ising_dict(h::Dict{Int,Float64}, J::Dict{Tuple{Int,Int},Float64})
    """
    Pack linear and quadratic Ising coefficients into the tuple-keyed Hamiltonian
    format expected by `JuliQAOA.spin_energy`.

    # Arguments
    - `h::Dict{Int,Float64}`: Linear coefficients.
    - `J::Dict{Tuple{Int,Int},Float64}`: Pairwise couplings.

    # Returns
    - `Dict{Tuple,Float64}`: Hamiltonian dictionary with keys `(i,)` and `(i,j)`.
    """
    H = Dict{Tuple,Float64}()
    for (i, coeff) in h
        add_term!(H, (i,), coeff)
    end
    for ((i, j), coeff) in J
        add_term!(H, (i, j), coeff)
    end
    return H
end


function maxcut_value(g::SimpleGraph, x)::Int
    """
    Compute the unweighted MaxCut value of a binary assignment `x` on graph `g`.

    # Arguments
    - `g::SimpleGraph`: Input graph.
    - `x`: Binary assignment indexed by vertex, interpreted as a bipartition label.

    # Returns
    - `Int`: Number of graph edges cut by `x`.

    # Notes
    An edge contributes `1` exactly when its endpoints are assigned different bits.
    """

    c = 0
    @inbounds for e in edges(g)
        u = src(e)
        v = dst(e)
        if x[u] != x[v]
            c += 1
        end
    end
    return c
end

function qubo_energy(Q::Dict{Tuple{Int,Int},Float64}, x)
    """
    Evaluate a QUBO objective on a binary assignment `x ∈ {0,1}^N`.

    # Arguments
    - `Q::Dict{Tuple{Int,Int},Float64}`: QUBO coefficient dictionary.
    - `x`: Binary assignment.

    # Returns
    - `Float64`: Objective value `∑_{(u,v)} Q[(u,v)] x_u x_v`.
    """
    E = 0.0
    for ((u, v), coeff) in Q
        E += coeff * x[u] * x[v]
    end
    return E
end

# ============================================================
# Problem builders
# ============================================================

function build_sk_problem(N::Int, states_vec; instance_seed::Int)
    """
    Build a Sherrington–Kirkpatrick (SK) instance.

    For each computational basis state in `states_vec`, this function evaluates the
    SK energy and then defines the QAOA score as `score = -energy`, so that larger
    score means better performance.

    # Arguments
    - `N::Int`: Number of spins / qubits.
    - `states_vec`: Cached computational basis states.
    - `instance_seed::Int`: Seed used to generate the random SK couplings.

    # Returns
    A tuple `(score_vals, meta)` where:
    - `score_vals` is aligned with `states_vec`,
    - `meta` is a named tuple describing the instance.
    """
    Random.seed!(instance_seed)
    J = sk_model(N)
    energy_vals = Float64[spin_energy(J, x) for x in states_vec]
    score_vals = -energy_vals
    meta = (
        problem = :SK,
        sense = "maximize -H_SK (equiv. minimize H_SK)",
        representation = "pairwise Ising",
        instance_seed = instance_seed,
    )
    return score_vals, meta
end

function build_maxcut_problem(N::Int, states_vec; p_edge::Float64, instance_seed::Int)
    """
    Build an unweighted MaxCut benchmark instance on a random Erdős–Rényi graph.

    # Arguments
    - `N::Int`: Number of graph vertices / qubits.
    - `states_vec`: Cached basis states.
    - `p_edge::Float64`: Edge probability for the random graph.
    - `instance_seed::Int`: Seed used to generate the graph.

    # Returns
    A tuple `(score_vals, meta)` where:
    - `score_vals[i]` is the cut value of `states_vec[i]`,
    - `meta` stores graph and instance metadata.
    """
    g = erdos_renyi_graph(N, p_edge; rng = MersenneTwister(instance_seed))
    score_vals = Float64[maxcut_value(g, x) for x in states_vec]
    meta = (
        problem = :MaxCut,
        sense = "maximize cut value",
        representation = "pairwise Ising / graph cut",
        instance_seed = instance_seed,
        p_edge = p_edge,
        n_edges = ne(g),
    )
    return score_vals, meta
end


function build_number_partitioning_problem(N::Int, states_vec;
                                           max_weight::Int,
                                           instance_seed::Int)

    """
    Build a number partitioning instance in Ising form.

    Given positive integer weights `a_1, …, a_N`, the partition imbalance is

    `(∑_i a_i s_i)^2`, where `s_i ∈ {+1,-1}`.

    Smaller imbalance is better, so this function returns

    `score(s) = -(∑_i a_i s_i)^2`.

    # Arguments
    - `N::Int`: Number of items / spins.
    - `states_vec`: Cached basis states.
    - `max_weight::Int`: Weights are sampled uniformly from `1:max_weight`.
    - `instance_seed::Int`: Seed controlling the sampled weights.

    # Returns
    A tuple `(score_vals, meta)`.
    """
    rng = MersenneTwister(instance_seed)
    weights = rand(rng, 1:max_weight, N)

    # (Σ a_i s_i)^2 = Σ a_i^2 + 2 Σ_{i<j} a_i a_j s_i s_j.
    # The constant Σ a_i^2 is irrelevant for optimization and monotonicity.
    H = Dict{Tuple,Float64}()
    @inbounds for i in 1:(N - 1)
        for j in (i + 1):N
            add_term!(H, (i, j), 2.0 * weights[i] * weights[j])
        end
    end

    energy_vals = Float64[spin_energy(H, x) for x in states_vec]
    score_vals = -energy_vals
    meta = (
        problem = :NumberPartitioning,
        sense = "maximize -(partition imbalance)^2",
        representation = "pairwise Ising (constant dropped)",
        instance_seed = instance_seed,
        weights = join(weights, ";"),
    )
    return score_vals, meta
end

function build_maxclique_problem(N::Int, states_vec;
                                 p_edge::Float64,
                                 instance_seed::Int,
                                 A::Float64 = 1.0,
                                 B::Float64 = 2.0)
    """
    Build a maximum clique instance using the standard QUBO formulation

    `H(x) = -A ∑_i x_i + B ∑_{(i,j) ∈ complement(G)} x_i x_j`.

    The linear term rewards selecting vertices, while the penalty term punishes
    pairs of selected vertices that are not connected in the original graph.
    Because the experiment is phrased in terms of "larger score = better", this
    function returns `score(x) = -H(x)`.

    # Arguments
    - `N::Int`: Number of graph vertices.
    - `states_vec`: Cached basis states.
    - `p_edge::Float64`: Edge probability used to sample the random graph.
    - `instance_seed::Int`: Seed used to generate the graph.
    - `A::Float64=1.0`: Reward coefficient for including a vertex.
    - `B::Float64=2.0`: Penalty coefficient for selecting a non-edge pair.

    # Returns
    A tuple `(score_vals, meta)`.
    """
    g = erdos_renyi_graph(N, p_edge; rng = MersenneTwister(instance_seed))
    Q = Dict{Tuple{Int,Int},Float64}()

    for i in 1:N
        Q[(i, i)] = -A
    end
    @inbounds for i in 1:(N - 1)
        for j in (i + 1):N
            if !has_edge(g, i, j)
                Q[(i, j)] = get(Q, (i, j), 0.0) + B
            end
        end
    end

    h, J, offset = qubo_to_ising_local(Q)
    Hising = pairwise_ising_dict(h, J)

    energy_vals = Float64[qubo_energy(Q, x) for x in states_vec]
    score_vals = -energy_vals
    meta = (
        problem = :MaxClique,
        sense = "maximize clique QUBO score (equiv. minimize QUBO energy)",
        representation = "QUBO converted to pairwise Ising",
        instance_seed = instance_seed,
        p_edge = p_edge,
        n_edges = ne(g),
        n_nonedges = div(N * (N - 1), 2) - ne(g),
        A = A,
        B = B,
        ising_offset = offset,
        n_ising_terms = length(Hising),
    )
    return score_vals, meta
end


function build_three_sat_problem(N::Int, states_vec;
                                 clause_density::Float64,
                                 instance_seed::Int)
    """
    Build a random 3-SAT instance and score each basis state by the number of
    satisfied clauses.

    # Arguments
    - `N::Int`: Number of Boolean variables / qubits.
    - `states_vec`: Cached basis states.
    - `clause_density::Float64`: Clause-to-variable ratio used to set the number of
    clauses.
    - `instance_seed::Int`: Seed controlling the random 3-SAT instance.

    # Returns
    A tuple `(score_vals, meta)`.
    """
    m = max(1, round(Int, clause_density * N))
    Random.seed!(instance_seed)
    instance = kSAT_instance(3, N, m)

    score_vals = Float64[kSAT(instance, x) for x in states_vec]
    meta = (
        problem = :ThreeSAT,
        sense = "maximize satisfied clauses",
        representation = "3-local diagonal cost (higher-order Ising equivalent)",
        instance_seed = instance_seed,
        m_clauses = m,
        clause_density = clause_density,
    )
    return score_vals, meta
end


function build_problem(N::Int, problem::Symbol, states_vec;
                       p_edge::Float64 = 0.5,
                       max_weight::Int = 10,
                       clause_density::Float64 = 4.2,
                       instance_seed::Int = 1,
                       clique_A::Float64 = 1.0,
                       clique_B::Float64 = 2.0)
    """
    Dispatch helper that constructs the requested benchmark problem and returns a
    lookup table of objective values aligned with `states_vec`.

    # Arguments
    - `N::Int`: Number of qubits.
    - `problem::Symbol`: Problem identifier. Supported values are `:SK`, `:MaxCut`,
    `:NumberPartitioning`, `:MaxClique`, and `:ThreeSAT`.
    - `states_vec`: Cached basis states, typically `collect(states(N))`.

    # Keyword Arguments
    - `p_edge::Float64=0.5`: Edge probability for graph-based problems.
    - `max_weight::Int=10`: Maximum sampled weight for number partitioning.
    - `clause_density::Float64=4.2`: Clause density for 3-SAT.
    - `instance_seed::Int=1`: Random seed for instance generation.
    - `clique_A::Float64=1.0`: Vertex reward coefficient for MaxClique.
    - `clique_B::Float64=2.0`: Non-edge penalty coefficient for MaxClique.

    # Returns
    A tuple `(score_vals, meta)`.
    """
    if problem == :SK
        return build_sk_problem(N, states_vec; instance_seed = instance_seed)
    elseif problem == :MaxCut
        return build_maxcut_problem(N, states_vec; p_edge = p_edge, instance_seed = instance_seed)
    elseif problem == :NumberPartitioning
        return build_number_partitioning_problem(N, states_vec;
            max_weight = max_weight,
            instance_seed = instance_seed,
        )
    elseif problem == :MaxClique
        return build_maxclique_problem(N, states_vec;
            p_edge = p_edge,
            instance_seed = instance_seed,
            A = clique_A,
            B = clique_B,
        )
    elseif problem == :ThreeSAT
        return build_three_sat_problem(N, states_vec;
            clause_density = clause_density,
            instance_seed = instance_seed,
        )
    else
        error("Unknown problem: $problem")
    end
end

# ============================================================
# Mixer selection
# ============================================================

function get_mixer(N::Int, kind::Symbol)
    """
    # Arguments
    - `N::Int`: Number of qubits.
    - `kind::Symbol`: Mixer identifier. Supported values:
    - `:X` for the standard transverse-field mixer,
    - `:Grover` for the Grover-style mixer when available.

    # Returns
    - `Mixer`: A JuliQAOA mixer object.

    """
    if kind == :X
        return mixer_x(N)
    elseif kind == :Grover
        if isdefined(JuliQAOA, :mixer_grover)
            return mixer_grover(N)
        end
        return mixer_x(N, 0:N) / 2^N
    else
        error("Unknown mixer kind: $kind. Use :X or :Grover")
    end
end

# ============================================================
# Monotonicity experiment
# ============================================================

@inline function assemble_prefix!(work::Vector{Float64}, betas::Vector{Float64}, gammas::Vector{Float64}, k::Int)
    """
    Write the depth-`k` QAOA prefix into the preallocated workspace vector 
    using JuliQAOA's expected angle ordering:

    `[β₁, β₂, …, β_k, γ₁, γ₂, …, γ_k]`.

    # Arguments
    - `work::Vector{Float64}`: Workspace with length at least `2k`.
    - `betas::Vector{Float64}`: Full sampled mixer-angle sequence.
    - `gammas::Vector{Float64}`: Full sampled cost-angle sequence.
    - `k::Int`: Prefix depth.

    # Returns
    - `nothing`
    """
    @inbounds for i in 1:k
        work[i] = betas[i]
        work[k + i] = gammas[i]
    end
    return nothing
end

function run_trials_prefix(max_p::Int,
                           reps::Int,
                           mixer::Mixer,
                           score_vals::AbstractVector;
                           rng::AbstractRNG,
                           tol::Float64 = 1e-12,
                           sample_from_periods::Bool = true,
                           canonicalize::Bool = false)

    """
    Estimate how rare monotone QAOA schedules are by Monte Carlo sampling.

    A single trial samples one random full-depth schedule of length `max_p`,
    evaluates every prefix depth `k = 1, …, max_p`, and checks whether the
    resulting prefix expectation sequence remains strictly monotone.

    For each `k ≥ 2`, a trial contributes to exactly one of three categories:
    1. increasing: `S₁ < S₂ < ⋯ < S_k`,
    2. decreasing: `S₁ > S₂ > ⋯ > S_k`,
    3. other: neither of the above.

    # Arguments
    - `max_p::Int`: Maximum QAOA depth sampled per trial.
    - `reps::Int`: Number of Monte Carlo samples.
    - `mixer::Mixer`: QAOA mixer object.
    - `score_vals::AbstractVector`: Diagonal objective values in computational basis
    order.

    # Keyword Arguments
    - `rng::AbstractRNG`: RNG used for angle sampling.
    - `tol::Float64=1e-12`: Strict-monotonicity tolerance.
    - `sample_from_periods::Bool=true`: If `true`, sample angles from the canonical
    operator periods reported by JuliQAOA when available; otherwise sample from
    `[0, 2π]`.
    - `canonicalize::Bool=false`: If `true`, map each angle prefix into JuliQAOA's
    canonical period window before computing the expectation value.

    # Returns
    A tuple `(inc_counts, dec_counts, beta_period, gamma_period)` where:
    - `inc_counts[k]` is the number of trials that are strictly increasing through
    depth `k`,
    - `dec_counts[k]` is the number of trials that are strictly decreasing through
    depth `k`,
    - `beta_period` is the mixer-angle sampling interval,
    - `gamma_period` is the cost-angle sampling interval.
    """
    inc_counts = zeros(Int, max_p)
    dec_counts = zeros(Int, max_p)

    beta_period = (sample_from_periods && isfinite(mixer.period)) ? mixer.period : TWO_PI
    obj_period_raw = get_operator_period(score_vals)
    gamma_period = (sample_from_periods && isfinite(obj_period_raw)) ? obj_period_raw : TWO_PI

    betas = Vector{Float64}(undef, max_p)
    gammas = Vector{Float64}(undef, max_p)
    work = Vector{Float64}(undef, 2 * max_p)

    for _ in 1:reps
        rand!(rng, betas)
        rand!(rng, gammas)
        @. betas *= beta_period
        @. gammas *= gamma_period

        # Evaluate the k = 1 prefix first.
        work[1] = betas[1]
        work[2] = gammas[1]
        angles1 = work[1:2]
        if canonicalize
            angles1 = clean_angles(angles1, mixer, score_vals)
        end
        prev_S = exp_value(angles1, mixer, score_vals)

        inc_ok = true
        dec_ok = true

        for k in 2:max_p
            assemble_prefix!(work, betas, gammas, k)
            anglesk = work[1:(2 * k)]
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
                break
            end

            prev_S = S
        end
    end

    return inc_counts, dec_counts, beta_period, gamma_period
end

function counts_to_props(counts::Vector{Int}, reps::Int)

    """
    Convert raw event counts into empirical proportions and within-instance binomial
    standard errors.

    # Arguments
    - `counts::Vector{Int}`: Event counts, typically indexed by depth `p`.
    - `reps::Int`: Number of Monte Carlo trials.

    # Returns
    A tuple `(props, se)` where:
    - `props[p] = counts[p] / reps`,
    - `se[p] = sqrt(props[p] * (1 - props[p]) / reps)`.
    """
    props = Float64[c / reps for c in counts]
    se = Float64[sqrt(p * (1 - p) / reps) for p in props]
    return props, se
end

# ============================================================
# CSV output
# ============================================================

function write_header(io)
    println(io, join([
        "problem",
        "representation",
        "sense",
        "mixer",
        "N",
        "p",
        "instance_seed",
        "angle_seed",
        "reps",
        "tol",
        "sample_from_periods",
        "canonicalize",
        "beta_period",
        "gamma_period",
        "inc_count",
        "dec_count",
        "other_count",
        "inc_prop",
        "dec_prop",
        "inc_se_binom",
        "dec_se_binom",
    ], ","))
end


function csvsafe(x)
    s = replace(string(x), '"' => "'")
    if occursin(',', s) || occursin(';', s)
        return "\"" * s * "\""
    end
    return s
end

function write_row(io; problem, representation, sense, mixer, N, p, instance_seed, angle_seed,
                   reps, tol, sample_from_periods, canonicalize, beta_period, gamma_period,
                   inc_count, dec_count, other_count, inc_prop, dec_prop, inc_se_binom, dec_se_binom)

    @printf(io,
        "%s,%s,%s,%s,%d,%d,%d,%d,%d,%.3e,%s,%s,%.17g,%.17g,%d,%d,%d,%.17g,%.17g,%.17g,%.17g\n",
        csvsafe(problem),
        csvsafe(representation),
        csvsafe(sense),
        csvsafe(mixer),
        N, p, instance_seed, angle_seed, reps, tol,
        sample_from_periods ? "true" : "false",
        canonicalize ? "true" : "false",
        beta_period, gamma_period,
        inc_count, dec_count, other_count,
        inc_prop, dec_prop, inc_se_binom, dec_se_binom,
    )
end

# ============================================================
# Main configuration
# ============================================================

Ns = [4, 6, 8]
max_p = 12
reps = 100_000
instance_seeds = collect(1:10)

mixers = [:X, :Grover]
problems = [:SK, :MaxCut, :NumberPartitioning, :MaxClique, :ThreeSAT]

# Problem-specific parameters.
p_edge = 0.5
max_weight = 10
clause_density = 4.2
clique_A = 1.0
clique_B = 2.0

tol = 1e-12
SAMPLE_FROM_PERIODS = true
CANONICALIZE = false
SHARE_ANGLES_ACROSS_MIXERS = true
ANGLE_SEED_BASE = 20260327

outdir = "results"
mkpath(outdir)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
raw_csv = joinpath(outdir, "monotonicity_extended_$(timestamp).csv")
println("Writing raw results to: $raw_csv")

function problem_id(problem::Symbol)
    ids = Dict(
        :SK => 1,
        :MaxCut => 2,
        :NumberPartitioning => 3,
        :MaxClique => 4,
        :ThreeSAT => 5,
    )
    return ids[problem]
end

open(raw_csv, "w") do io
    write_header(io)

    for N in Ns
        states_cache = collect(states(N))
        println("\n====================")
        println("N = $N")
        println("====================")

        for problem in problems
            for instance_seed in instance_seeds
                score_vals, meta = build_problem(N, problem, states_cache;
                    p_edge = p_edge,
                    max_weight = max_weight,
                    clause_density = clause_density,
                    instance_seed = instance_seed,
                    clique_A = clique_A,
                    clique_B = clique_B,
                )

                println("\nProblem=$(problem) | N=$(N) | instance_seed=$(instance_seed) | $(meta.sense)")

                base_angle_seed = ANGLE_SEED_BASE + 1_000_000 * problem_id(problem) + 1_000 * N + instance_seed

                for mixer_kind in mixers
                    mixer_obj = get_mixer(N, mixer_kind)
                    mixer_id = mixer_kind == :X ? 1 : 2
                    angle_seed = SHARE_ANGLES_ACROSS_MIXERS ? base_angle_seed : (base_angle_seed + 10_000 * mixer_id)
                    rng = MersenneTwister(angle_seed)

                    t0 = time()
                    inc_counts, dec_counts, beta_period, gamma_period = run_trials_prefix(
                        max_p,
                        reps,
                        mixer_obj,
                        score_vals;
                        rng = rng,
                        tol = tol,
                        sample_from_periods = SAMPLE_FROM_PERIODS,
                        canonicalize = CANONICALIZE,
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
                            representation = meta.representation,
                            sense = meta.sense,
                            mixer = mixer_kind,
                            N = N,
                            p = p,
                            instance_seed = instance_seed,
                            angle_seed = angle_seed,
                            reps = reps,
                            tol = tol,
                            sample_from_periods = SAMPLE_FROM_PERIODS,
                            canonicalize = CANONICALIZE,
                            beta_period = beta_period,
                            gamma_period = gamma_period,
                            inc_count = inc_c,
                            dec_count = dec_c,
                            other_count = other_c,
                            inc_prop = inc_props[p],
                            dec_prop = dec_props[p],
                            inc_se_binom = inc_se[p],
                            dec_se_binom = dec_se[p],
                        )
                    end

                    println(@sprintf(
                        "  mixer=%s | beta_period=%.6g | gamma_period=%.6g | done in %.2fs",
                        String(mixer_kind), beta_period, gamma_period, elapsed,
                    ))
                    println(@sprintf(
                        "    p=%d: inc=%.6g, dec=%.6g, mono=%.6g",
                        max_p, inc_props[max_p], dec_props[max_p], inc_props[max_p] + dec_props[max_p],
                    ))
                end
            end
        end
    end
end

println("\nDone. Run fit_monotonicity.py on the CSV in results/ to fit decay models.")
