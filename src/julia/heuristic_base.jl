using CSV
using DataFrames
using Dates

include("inputs.jl")
include("pyarray.jl")
include("objects.jl")

const BENCHMARK_NEH = false
const BENCHMARK_RUNS = 1000

const t = 0.05

function insertJobIntoSequence(solution, inputs, k, kJob)
    # Create earliest, tail, and relative completion times structures
    n = lastindex(solution.jobs)
    e = PythonLikeArray(n + 2, inputs.nMachines + 1)
    q = PythonLikeArray(n + 2, inputs.nMachines + 1)
    f = PythonLikeArray(n + 2, inputs.nMachines + 1)

    # Compute earliest, tail, and relative completion times values
    for i = 1:n+1
        for j = 1:inputs.nMachines
            if i < n + 1
                e[i, j] =
                    max(e[i, j-1], e[i-1, j]) +
                    inputs.times[solution.jobs[i], j]
            end
            if i > 1
                q[n+2-i, inputs.nMachines+1-j] =
                    max(
                        q[n+2-i, inputs.nMachines+2-j],
                        q[n+3-i, inputs.nMachines+1-j],
                    ) + inputs.times[solution.jobs[n+2-i], inputs.nMachines+1-j]
            end
            f[i, j] = max(f[i, j-1], e[i-1, j]) + inputs.times[kJob, j]
        end
    end

    # Find position of minimum makespan
    Mi = maximum(f.data + q.data, dims = 2)
    index = argmin(Mi[1:k])

    # Insert job in the sequence and update makespan
    insert!(solution.jobs, index, kJob)
    solution.makespan = Mi[index]
end

function PFSP_Heuristic(inputs, jobIndices)
    solution = Solution()
    solution.jobs = [jobIndices[1]]
    for i = 2:lastindex(jobIndices)
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
    totalTimes = sum(inputs.times, dims = 2)
    sortedJobIndices = reverse(sortperm(vec(totalTimes), alg = MergeSort))

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
    startTime = time()
    baseSolution = PFSP_Multistart(inputs, rng)
    baseSolution.time = time() - startTime
    solution_data =
        [(time = baseSolution.time, makespan = baseSolution.makespan)]
    if BENCHMARK_NEH
        return baseSolution, []
    end
    baseSolution = localSearch(baseSolution, inputs, rng)
    bestSolution = baseSolution
    bestSolution.time = time() - startTime
    push!(
        solution_data,
        (time = bestSolution.time, makespan = bestSolution.makespan),
    )

    # Start the iterated local search process
    credit = 0
    elapsedTime = 0
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
                push!(
                    solution_data,
                    (
                        time = bestSolution.time,
                        makespan = bestSolution.makespan,
                    ),
                )
            end
        elseif 0 < delta <= credit
            credit = 0
            baseSolution = solution
        end
        # Update the elapsed time before evaluating the stopping criterion
        elapsedTime = time() - startTime
    end
    return bestSolution, solution_data
end

function benchmark_execution(inputs, test, rng)
    elapsed_times = Float64[]
    for _ = 1:BENCHMARK_RUNS
        start_time = time()
        solution = detExecution(inputs, test, rng)
        end_time = time()
        push!(elapsed_times, end_time - start_time)
    end
    return elapsed_times
end

function write_to_csv(tests_dir, filename, instance_name, best_solutions)
    best_solutions_dir = joinpath(tests_dir, filename)
    open(best_solutions_dir, "a") do io
        for (t, m) in best_solutions
            println(io, "$instance_name,$t,$m")
        end
    end
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

    # Create or overwrite the file with the header
    best_solutions_filename = "julia_base_solutions.csv"
    best_solutions_dir = joinpath(tests_dir, best_solutions_filename)
    open(best_solutions_dir, "w") do io
        println(io, "Instance,Time,Makespan")
    end

    for test in tests
        # Read inputs for the test inputs
        inputs_dir = joinpath(base_dir, "inputs")
        inputs = readInputs(inputs_dir, test.instanceName)
        rng = MersenneTwister(test.seed)

        if BENCHMARK_NEH
            # Benchmark NEH execution
            elapsed_times = benchmark_execution(inputs, test, rng)
            if !haskey(execution_times_dict, test.instanceName)
                execution_times_dict[test.instanceName] = Float64[]
            end
            append!(execution_times_dict[test.instanceName], elapsed_times)
        else
            # Compute the best deterministic solution
            _, best_solutions_data = detExecution(inputs, test, rng)
            write_to_csv(
                tests_dir,
                best_solutions_filename,
                test.instanceName,
                best_solutions_data,
            )
        end
    end
end

main()
