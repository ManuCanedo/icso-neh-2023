using Dates

include("inputs.jl")
include("pyarray.jl")
include("objects.jl")

const BENCHMARK_NEH = false
const BENCHMARK_RUNS = 20

const t = 0.01

function populate_e!(jobs::Vector{Int}, inputs::Inputs, index::Int, e::Array{Int})
    e[1, 1] = inputs.times[jobs[1], 1]

    for j = 2:inputs.nMachines
        e[1, j] = inputs.times[jobs[1], j] + e[1, j-1]
    end
    for i = 2:index
        e[i, 1] = inputs.times[jobs[i], 1] + e[i-1, 1]

        for j = 2:inputs.nMachines
            e[i, j] = inputs.times[jobs[i], j] + max(e[i-1, j], e[i, j-1])
        end
    end
end

function populate_q!(jobs::Vector{Int}, inputs::Inputs, index::Int, q::Array{Int})
    for j = inputs.nMachines:-1:1
        q[index, j] = 0
    end
    if index == 1
        return
    end
    q[index-1, inputs.nMachines] = inputs.times[jobs[index-1], inputs.nMachines]

    for j = inputs.nMachines-1:-1:1
        q[index-1, j] = inputs.times[jobs[index-1], j] + q[index-1, j+1]
    end
    if index == 2
        return
    end
    for i = index-2:-1:1
        q[i, inputs.nMachines] =
            inputs.times[jobs[i], inputs.nMachines] + q[i+1, inputs.nMachines]

        for j = inputs.nMachines-1:-1:1
            q[i, j] = inputs.times[jobs[i], j] + max(q[i+1, j], q[i, j+1])
        end
    end
end

function populate_f!(kJob::Int, inputs::Inputs, index::Int, e::Array{Int}, f::Array{Int})
    f[1, 1] = inputs.times[kJob, 1]

    for j = 2:inputs.nMachines
        f[1, j] = inputs.times[kJob, j] + f[1, j-1]
    end
    for i = 2:index
        f[i, 1] = inputs.times[kJob, 1] + e[i-1, 1]

        for j = 2:inputs.nMachines
            f[i, j] = inputs.times[kJob, j] + max(e[i-1, j], f[i, j-1])
        end
    end
end

function insertJobIntoSequence(
    solution::Solution,
    inputs::Inputs,
    k::Int,
    kJob::Int,
    eq::Array{Int},
    f::Array{Int},
)
    n = length(solution.jobs)
    # Compute earliest, tail, and relative completion times structures
    populate_e!(solution.jobs, inputs, n, eq)
    populate_f!(kJob, inputs, n + 1, eq, f)
    populate_q!(solution.jobs, inputs, n + 1, eq)
    # Find position of minimum makespan
    index = k
    solution.makespan = typemax(Int)
    for i = 1:k
        max_sum = 0
        for j = 1:inputs.nMachines
            max_sum = max(f[i, j] + eq[i, j], max_sum)
        end
        if max_sum < solution.makespan
            index = i
            solution.makespan = max_sum
        end
    end
    # Insert job in the sequence and update makespan
    insert!(solution.jobs, min(index, n + 1), kJob)
end


function PFSP_Heuristic(
    inputs::Inputs,
    jobIndices::Vector{Int},
    eq::Array{Int},
    f::Array{Int},
)
    solution = Solution()
    solution.jobs = [jobIndices[1]]
    for i = 2:length(jobIndices)
        insertJobIntoSequence(solution, inputs, i, jobIndices[i], eq, f)
    end
    return solution
end

function createBiasedJobsSequence(jobs::Vector{Int}, rng::AbstractRNG)
    jobsCopy = copy(jobs)
    biasedJobs = Int[]
    for _ = 1:length(jobsCopy)
        index = trunc(Int, length(jobsCopy) * (1 - sqrt(1 - rand(rng)))) + 1
        push!(biasedJobs, jobsCopy[index])
        deleteat!(jobsCopy, index)
    end
    return biasedJobs
end

function PFSP_Multistart(inputs::Inputs, rng::AbstractRNG, eq::Array{Int}, f::Array{Int})
    totalTimes = sum(inputs.times, dims=2)
    sortedJobIndices = sortperm(vec(totalTimes), rev=true)
    nehSolution = PFSP_Heuristic(inputs, sortedJobIndices, eq, f)
    if BENCHMARK_NEH
        return nehSolution
    end
    println("NEH makespan: $(nehSolution.makespan)")
    baseSolution = nehSolution
    nIter = 0
    while baseSolution.makespan >= nehSolution.makespan && nIter < inputs.nJobs
        nIter += 1
        biasedJobs = createBiasedJobsSequence(sortedJobIndices, rng)
        newSolution = PFSP_Heuristic(inputs, biasedJobs, eq, f)
        if newSolution.makespan < baseSolution.makespan
            baseSolution = newSolution
        end
    end
    return baseSolution
end

function localSearch(
    solution::Solution,
    inputs::Inputs,
    rng::AbstractRNG,
    eq::Array{Int},
    f::Array{Int},
)
    improve = true
    while improve
        improve = false
        for index in randperm(rng, length(solution.jobs))
            newSolution = Solution(solution.jobs[:], solution.makespan, 0)
            job = popat!(newSolution.jobs, index)
            insertJobIntoSequence(newSolution, inputs, index, job, eq, f)
            if newSolution.makespan < solution.makespan
                solution = newSolution
                improve = true
            end
        end
    end
    return solution
end

function perturbation(
    baseSolution::Solution,
    inputs::Inputs,
    rng::AbstractRNG,
    eq::Array{Int},
    f::Array{Int},
)
    solution = Solution()
    solution.jobs = copy(baseSolution.jobs)
    solution.makespan = baseSolution.makespan
    # Select two random jobs from the sequence
    aIndex, bIndex = rand(rng, 1:length(solution.jobs), 2)
    # Swap the jobs at the two random positions
    solution.jobs[aIndex], solution.jobs[bIndex] =
        solution.jobs[bIndex], solution.jobs[aIndex]
    if bIndex < aIndex
        aIndex, bIndex = bIndex, aIndex
    end
    # Insert the left-most swapped job where the makespan is minimized
    aJob = splice!(solution.jobs, aIndex)
    insertJobIntoSequence(solution, inputs, aIndex, aJob, eq, f)
    # Insert the right-most swapped job where the makespan is minimized
    bJob = splice!(solution.jobs, bIndex)
    insertJobIntoSequence(solution, inputs, bIndex, bJob, eq, f)
    return solution
end

function detExecution(inputs::Inputs, test::TestData, rng::MersenneTwister)
    eq = Array{Int}(undef, inputs.nJobs, inputs.nMachines)
    f = Array{Int}(undef, inputs.nJobs, inputs.nMachines)

    # Create a base solution using a randomized NEH approach
    baseSolution = PFSP_Multistart(inputs, rng, eq, f)
    if BENCHMARK_NEH
        return baseSolution
    end
    println("Multistart makespan: $(baseSolution.makespan)")
    baseSolution = localSearch(baseSolution, inputs, rng, eq, f)
    bestSolution = baseSolution
    println("LS makespan: $(bestSolution.makespan)")

    # Start the iterated local search process
    credit = 0
    elapsedTime = 0
    startTime = time()
    maxTime = inputs.nJobs * inputs.nMachines * t
    while elapsedTime < maxTime
        # Perturb the base solution to find a new solution
        solution = perturbation(baseSolution, inputs, rng, eq, f)
        solution = localSearch(solution, inputs, rng, eq, f)
        # Check if the solution is adept to be the new base solution
        delta = solution.makespan - baseSolution.makespan
        if delta < 0
            credit = -delta
            baseSolution = solution
            if solution.makespan < bestSolution.makespan
                bestSolution = solution
                bestSolution.time = time() - startTime
            end
        elseif 0 < delta <= credit
            credit = 0
            baseSolution = solution
        end
        # Update the elapsed time before evaluating the stopping criterion
        elapsedTime = time() - startTime
    end
    return bestSolution
end

function printSolution(solution, print_solution=false)
    if print_solution
        println("Jobs: " * join([string(job) for job in solution.jobs], ", "))
    end
    println("ILS Makespan: $(round(solution.makespan, digits=2))")
    println("Time: $(round(solution.time, digits=2))")
end


function main()
    base_path = "/Users/mtabares/dev/icso-neh/"

    # Read tests from the file
    tests = readTests(joinpath(base_path, "tests", "test2run.txt"))

    for test in tests
        # Read inputs for the test inputs
        inputs = readInputs(joinpath(base_path, "inputs"), test.instanceName)
        rng = MersenneTwister(test.seed)

        println("Julia Optimised: OBD $(inputs.name)")
        solution = Solution()

        if BENCHMARK_NEH
            elapsed_times = Float64[]
            for _ in 1:BENCHMARK_RUNS
                start_time = time()
                # Compute the best deterministic solution
                solution = detExecution(inputs, test, rng)
                end_time = time()
                push!(elapsed_times, end_time - start_time)
            end

            avg_time = sum(elapsed_times) / length(elapsed_times)
            min_time = minimum(elapsed_times)
            max_time = maximum(elapsed_times)
            println("Average elapsed time: $(avg_time) seconds")
            println("Minimum elapsed time: $(min_time) seconds")
            println("Maximum elapsed time: $(max_time) seconds")
        else
            # Compute the best deterministic solution
            solution = detExecution(inputs, test, rng)
        end
        printSolution(solution)
    end
end

main()
