using Dates

include("inputs.jl")
include("pyarray.jl")
include("objects.jl")

function insertJobIntoSequence(solution, inputs, k, kJob)
	n = length(solution.jobs)
	# Create earliest, tail, and relative completion times structures
	e = toPythonLikeArray(zeros(n + 2, inputs.nMachines + 1))
	q = toPythonLikeArray(zeros(n + 2, inputs.nMachines + 1))
	f = toPythonLikeArray(zeros(n + 2, inputs.nMachines + 1))
	# Compute earliest, tail, and relative completion times values
	for i in 1:(n+1)
		for j in 1:inputs.nMachines
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
					) +
					inputs.times[solution.jobs[n+2-i], inputs.nMachines+1-j]
			end
			f[i, j] = max(f[i, j-1], e[i-1, j]) + inputs.times[kJob, j]
		end
	end
	# Find position of minimum makespan
	Mi = maximum(f.data + q.data, dims = 2)[1:end-1]
	index = argmin(Mi[1:min(k + 1, end)])
	# Insert job in the sequence and update makespan
	insert!(solution.jobs, index, kJob)
	solution.makespan = Mi[index]
end


function PFSP_Heuristic(inputs::Inputs, jobs::Vector{Int})
	solution = Solution()
	solution.jobs = [jobs[1]]
	for (i, job) in enumerate(jobs[2:end])
		insertJobIntoSequence(solution, inputs, i, job)
	end
	return solution
end

function createBiasedJobsSequence(jobs, rng::AbstractRNG)
	jobsCopy = copy(jobs)
	biasedJobs = Int[]
	for _ in 1:length(jobsCopy)
		index = trunc(Int, length(jobsCopy) * (1 - sqrt(1 - rand(rng)))) + 1
		push!(biasedJobs, jobsCopy[index])
		deleteat!(jobsCopy, index)
	end
	return biasedJobs
end

function PFSP_Multistart(inputs::Inputs, test::TestData, rng::AbstractRNG)
	totalTimes = sum(inputs.times, dims = 2)
	sortedJobs = reverse(sortperm(vec(totalTimes)))
	nehSolution = PFSP_Heuristic(inputs, sortedJobs)
	baseSolution = nehSolution
	nIter = 0
	while baseSolution.makespan >= nehSolution.makespan && nIter < inputs.nJobs
		nIter += 1
		biasedJobs = createBiasedJobsSequence(sortedJobs, rng)
		newSolution = PFSP_Heuristic(inputs, biasedJobs)
		if newSolution.makespan < baseSolution.makespan
			baseSolution = newSolution
		end
	end
	return baseSolution
end

function localSearch(solution::Solution, inputs::Inputs, rng::AbstractRNG)
	improve = true
	while improve
		improve = false
		for job in randperm(rng, inputs.nJobs)
			newSolution = Solution(solution.jobs[:], solution.makespan, 0)
			index = findfirst(isequal(job), newSolution.jobs)
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

function perturbation(baseSolution::Solution, inputs::Inputs, rng::AbstractRNG)
	solution = Solution()
	solution.jobs = copy(baseSolution.jobs)
	solution.makespan = baseSolution.makespan
	# Select two random jobs from the sequence
	aIndex, bIndex = rand(rng, 1:inputs.nJobs, 2)
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

function detExecution(inputs::Inputs, test::TestData, rng::MersenneTwister)
	# Create a base solution using a randomized NEH approach
	baseSolution = PFSP_Multistart(inputs, test, rng)
	baseSolution = localSearch(baseSolution, inputs, rng)
	bestSolution = baseSolution
	# Start the iterated local search process
	credit = 0
	elapsedTime = 0
	startTime = time_ns()
	t = parse(Int, test.maxTime)
	while elapsedTime < (inputs.nJobs * inputs.nMachines * t)
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
				bestSolution.time = time_ns()
			end
		elseif 0 < delta <= credit
			credit = 0
			baseSolution = solution
		end
		# Update the elapsed time before evaluating the stopping criterion
		currentTime = time_ns()
		elapsedTime = currentTime - startTime
	end
	return bestSolution
end

function print_solution(solution::Solution)
    println("Jobs: " * join(string(job) for job in solution.jobs, ", "))
    println("Makespan: $(round(solution.makespan, digits=2))")
    println("Time: $(round(solution.time, digits=2))")
end

function main()
    script_dir = dirname(abspath(PROGRAM_FILE))
    base_path = joinpath(script_dir, "..", "..")

    # Read tests from the file
    tests = read_tests(joinpath(base_path, "tests", "test2run.txt"))

    for test in tests
        # Read inputs for the test inputs
        inputs = read_inputs(joinpath(base_path, "inputs"), test.instanceName)
        rng = MersenneTwister(test.seed)

        # Compute the best deterministic solution
        solution = detExecution(inputs, test, rng)
        println("OBD $(inputs.name)")
        print_solution(solution)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end




