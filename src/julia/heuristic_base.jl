using CSV
using DataFrames
using Dates

include("inputs.jl")
include("pyarray.jl")
include("objects.jl")

const BENCHMARK_NEH = true
const BENCHMARK_RUNS = 1000

const t = 0.01

function insertJobIntoSequence(solution, inputs, k, kJob)
    n = length(solution.jobs)
    # Create earliest, tail, and relative completion times structures
    e = PythonLikeArray(inputs.nMachines + 1, n + 2)
    q = PythonLikeArray(inputs.nMachines + 1, n + 2)
    f = PythonLikeArray(inputs.nMachines + 1, n + 2)
    # Compute earliest, tail, and relative completion times values
    for j = 1:n+1
        for i = 1:inputs.nMachines
            if j < n + 1
                e[i, j] = max(e[i-1, j], e[i, j-1]) + inputs.times[solution.jobs[j], i]
            end
            if j > 1
                q[inputs.nMachines+1-i, n+2-j] =
                    max(q[inputs.nMachines+2-i, n+2-j], q[inputs.nMachines+1-i, n+3-j]) +
                    inputs.times[solution.jobs[n+2-j], inputs.nMachines+1-i]
            end
            f[i, j] = max(f[i-1, j], e[i, j-1]) + inputs.times[kJob, i]
        end
    end
    # Find position of minimum makespan
    Mi = maximum(f.data + q.data, dims=2)[1:end]
    index = argmin(Mi[1:min(k, end)])
    # Insert job in the sequence and update makespan
    insert!(solution.jobs, index, kJob)
    solution.makespan = Mi[index]
end

function PFSP_Heuristic(inputs, jobIndices)
    solution = Solution()
    push!(solution.jobs, jobIndices[1])
    for i = 2:length(jobIndices)
        insertJobIntoSequence(solution, inputs, i, jobIndices[i])
    end
    return solution
end

function createBiasedJobsSequence(jobs, rng)
    jobsCopy = copy(jobs)
    biasedJobs = Int[]
    for _ = 1:length(jobsCopy)
        index = trunc(Int, length(jobsCopy) * (1 - sqrt(1 - rand(rng)))) + 1
        push!(biasedJobs, jobsCopy[index])
        deleteat!(jobsCopy, index)
    end
    return biasedJobs
end

function PFSP_Multistart(inputs, rng)
    totalTimes = sum(inputs.times, dims=2)
    sortedJobIndices = sortperm(vec(totalTimes), rev=true)
    nehSolution = PFSP_Heuristic(inputs, sortedJobIndices)
    if BENCHMARK_NEH
        return nehSolution
    end
    baseSolution = nehSolution
    nIter = 0
    while baseSolution.makespan >= nehSolution.makespan && nIter < inputs.nJobs
        nIter += 1
        biasedJobs = createBiasedJobsSequence(sortedJobIndices, rng)
        newSolution = PFSP_Heuristic(inputs, biasedJobs)
        if newSolution.makespan < baseSolution.makespan
            baseSolution = newSolution
        end
    end
    return baseSolution
end

function localSearch(solution, inputs, rng)
    improve = true
    while improve
        improve = false
        for index in randperm(rng, length(solution.jobs))
            job = solution.jobs[index]
            newSolution = Solution(solution.jobs[:], solution.makespan, 0)
            deleteat!(newSolution.jobs, index)
            insertJobIntoSequence(newSolution, inputs, index, job)
            if newSolution.makespan < solution.makespan
                solution = newSolution
                improve = true
            end
        end
    end
    return solution
end

function perturbation(baseSolution, inputs, rng)
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
    insertJobIntoSequence(solution, inputs, aIndex, aJob)
    # Insert the right-most swapped job where the makespan is minimized
    bJob = splice!(solution.jobs, bIndex)
    insertJobIntoSequence(solution, inputs, bIndex, bJob)
    return solution
end

function detExecution(inputs, test, rng)
    # Create a base solution using a randomized NEH approach
    baseSolution = PFSP_Multistart(inputs, rng)
    if BENCHMARK_NEH
        return baseSolution
    end
    baseSolution = localSearch(baseSolution, inputs, rng)
    bestSolution = baseSolution
    println("Multistart LS makespan: $(bestSolution.makespan)")

    # Start the iterated local search process
    credit = 0
    elapsedTime = 0
    startTime = time()
    maxTime = inputs.nJobs * inputs.nMachines * t
    while elapsedTime < maxTime
        # Perturb the base solution to find a new solution
        solution = perturbation(baseSolution, inputs, rng)
        solution = localSearch(solution, inputs, rng)
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
    println("Makespan: $(round(solution.makespan, digits=2))")
    println("Time: $(round(solution.time, digits=2))")
end

function benchmark_execution(inputs, test, rng)
    elapsed_times = Float64[]
    for _ in 1:BENCHMARK_RUNS
        start_time = time()
        solution = detExecution(inputs, test, rng)
        end_time = time()
        push!(elapsed_times, end_time - start_time)
    end
    return elapsed_times
end

function write_to_csv(tests_dir, execution_times_dict)
    dict_keys = collect(keys(execution_times_dict))
    dict_values = collect(values(execution_times_dict))
    csv_data = DataFrame(dict_values, Symbol.(dict_keys))
    data_dir = joinpath(tests_dir, "$(basename(PROGRAM_FILE)).csv")
    CSV.write(data_dir, csv_data)
end

function main()
    base_dir = ""
    try
        base_dir = ARGS[1]
    catch
        println("Please provide a base path as a command-line argument.")
        exit(1)
    end

    # Read tests from the file
    tests_dir = joinpath(base_dir, "tests")
    tests = readTests(joinpath(tests_dir, "test2run.txt"))
    execution_times_dict = Dict()

    for test in tests
        # Read inputs for the test inputs
        inputs_dir = joinpath(base_dir, "inputs")
        inputs = readInputs(inputs_dir, test.instanceName)
        rng = MersenneTwister(test.seed)

        println("Julia Base: OBD $(inputs.name)")
        solution = Solution()

        if BENCHMARK_NEH
            # Benchmark NEH execution
            elapsed_times = benchmark_execution(inputs, test, rng)
            if !haskey(execution_times_dict, test.instanceName)
                execution_times_dict[test.instanceName] = Float64[]
            end
            append!(execution_times_dict[test.instanceName], elapsed_times)
        else
            # Compute the best deterministic solution
            solution = detExecution(inputs, test, rng)
            printSolution(solution)
        end
    end
    write_to_csv(tests_dir, execution_times_dict)
end

main()

