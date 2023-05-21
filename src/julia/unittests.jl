using Test

include("inputs.jl")
include("objects.jl")
include("heuristic.jl")

function compare_inputs(a::Inputs, b::Inputs)
    return a.name == b.name &&
           a.nJobs == b.nJobs &&
           a.nMachines == b.nMachines &&
           a.times == b.times
end

function compare_tests(a::TestData, b::TestData)
    return a.instanceName == b.instanceName &&
           a.maxTime == b.maxTime &&
           a.nIter == b.nIter &&
           a.distCrit == b.distCrit &&
           a.betaMin == b.betaMin &&
           a.betaMax == b.betaMax &&
           a.distCand == b.distCand &&
           a.betaMin2 == b.betaMin2 &&
           a.betaMax2 == b.betaMax2 &&
           a.seed == b.seed &&
           a.shortSim == b.shortSim &&
           a.longSim == b.longSim &&
           a.variance == b.variance &&
           a.execType == b.execType
end

@testset "test_readInputs" begin
    test_path = realpath(joinpath(@__DIR__, "..", "..", "inputs"))
    test_instance = "unittest_data"
    expected_input = Inputs(
        test_instance,
        5,
        3,
        [
            1 2 3
            4 5 6
            7 8 9
            10 11 12
            13 14 15
        ],
    )
    input = readInputs(test_path, test_instance)
    @test compare_inputs(input, expected_input)
end

@testset "test_readTests" begin
    test_file = realpath(joinpath(@__DIR__, "..", "..", "tests/test2run.txt"))
    expected_tests = [
        TestData(
            "tai108_200_20",
            "300",
            "100000000",
            "u",
            "1.0",
            "1.0",
            "g",
            "0.1",
            "0.3",
            "8634452",
            "100",
            "1000",
            "1.0",
            "0",
        ),
    ]
    tests = readTests(test_file)
    @test length(tests) == length(expected_tests)
    for (i, expected_test) in enumerate(expected_tests)
        @test compare_tests(tests[i], expected_test)
    end
end

function compare_solutions(a::Solution, b::Solution)
    return a.jobs == b.jobs && a.makespan == b.makespan && a.time == b.time
end

function test_insertJobIntoSequence()
    inputs = Inputs(
        "test",
        5,
        3,
        [
            1 2 3
            4 5 6
            7 8 9
            10 11 12
            13 14 15
        ],
    )

    @testset "insertJobIntoSequence" begin
        # Test case 1
        sol = Solution([1, 2, 3, 4, 5], 0, 0)
        insertJobIntoSequence(sol, inputs, 3, 5)
        @test sol.jobs == [1, 2, 3, 5, 4, 5]
        @test sol.makespan == 81

        # Test case 2
        sol = Solution([1, 2, 3, 4, 5], 0, 0)
        insertJobIntoSequence(sol, inputs, 0, 5)
        @test sol.jobs == [5, 1, 2, 3, 4, 5]
        @test sol.makespan == 87

        # Test case 3
        sol = Solution([1, 2, 3, 4, 5], 0, 0)
        insertJobIntoSequence(sol, inputs, 5, 5)
        @test sol.jobs == [1, 2, 3, 4, 5, 5]
        @test sol.makespan == 79

        # Test case 4
        sol = Solution([2, 3, 4, 5], 0, 0)
        insertJobIntoSequence(sol, inputs, 3, 1)
        @test sol.jobs == [1, 2, 3, 4, 5]
        @test sol.makespan == 64
    end
end

function test_PFSP_Heuristic()
    inputs = Inputs(
        "test",
        5,
        3,
        [
            1 2 3
            4 5 6
            7 8 9
            10 11 12
            13 14 15
        ],
    )

    @testset "PFSP_Heuristic" begin
        # Test case 1
        jobs = [1, 2, 3, 4, 5]
        expected_solution = Solution([1, 2, 3, 4, 5], 64, 0)
        solution = PFSP_Heuristic(inputs, jobs)
        @test compare_solutions(solution, expected_solution)

        # Test case 2
        jobs = [5, 4, 3, 2, 1]
        expected_solution = Solution([1, 2, 3, 4, 5], 64, 0)
        solution = PFSP_Heuristic(inputs, jobs)
        @test compare_solutions(solution, expected_solution)

        # Test case 3
        jobs = [3, 2, 1, 5, 4]
        expected_solution = Solution([1, 4, 2, 3, 5], 64, 0)
        solution = PFSP_Heuristic(inputs, jobs)
        @test compare_solutions(solution, expected_solution)

        # Test case 4
        jobs = [2, 5, 1, 4, 3]
        expected_solution = Solution([3, 4, 1, 2, 5], 64, 0)
        solution = PFSP_Heuristic(inputs, jobs)
        @test compare_solutions(solution, expected_solution)
    end
end

function test_createBiasedJobsSequence()
    @testset "createBiasedJobsSequence" begin
        rng = MersenneTwister(1234)
        jobs = [1, 2, 3, 4, 5]

        # Test case 1
        expected_biased_jobs_1 = [2, 4, 3, 1, 5]
        biasedJobs = createBiasedJobsSequence(jobs, rng)
        @test biasedJobs == expected_biased_jobs_1

        # Test case 2
        rng = MersenneTwister(1234)
        jobs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        expected_biased_jobs_2 = [4, 6, 3, 2, 8, 9, 1, 5, 7, 10]
        biasedJobs = createBiasedJobsSequence(jobs, rng)
        @test biasedJobs == expected_biased_jobs_2
    end
end

function test_PFSP_Multistart()
    inputs = Inputs(
        "test",
        5,
        3,
        [
            1 2 3
            4 5 6
            7 8 9
            10 11 12
            13 14 15
        ],
    )

    test_data = TestData(
        "unittest_data",
        "300",
        "100000000",
        "u",
        "1.0",
        "1.0",
        "g",
        "0.1",
        "0.3",
        "8634452",
        "100",
        "1000",
        "1.0",
        "0",
    )

    @testset "PFSP_Multistart" begin
        seed = 1234
        rng = MersenneTwister(seed)
        nehSolution =
            PFSP_Heuristic(inputs, reverse(sortperm(vec(sum(inputs.times, dims = 2)))))
        multistartSolution = PFSP_Multistart(inputs, test_data, rng)
        @test multistartSolution.makespan <= nehSolution.makespan

        seed = 5678
        rng = MersenneTwister(seed)
        nehSolution =
            PFSP_Heuristic(inputs, reverse(sortperm(vec(sum(inputs.times, dims = 2)))))
        multistartSolution = PFSP_Multistart(inputs, test_data, rng)
        @test multistartSolution.makespan <= nehSolution.makespan

        seed = 9012
        rng = MersenneTwister(seed)
        nehSolution =
            PFSP_Heuristic(inputs, reverse(sortperm(vec(sum(inputs.times, dims = 2)))))
        multistartSolution = PFSP_Multistart(inputs, test_data, rng)
        @test multistartSolution.makespan <= nehSolution.makespan
    end
end

function test_localSearch()
    rng = MersenneTwister(42)
    inputs = Inputs(
        "test",
        5,
        3,
        [
            1 2 3
            4 5 6
            7 8 9
            10 11 12
            13 14 15
        ],
    )

    @testset "localSearch" begin
        # Test case 1
        sol = Solution([1, 2, 3, 4, 5], 64, 0)
        expected_sol = Solution([1, 3, 4, 5, 2], 61, 0)
        output_sol = localSearch(sol, inputs, rng)
        # @test compare_solutions(output_sol, expected_sol)
    end
end

function test_perturbation()
    rng = MersenneTwister(42)
    inputs = Inputs(
        "test",
        5,
        3,
        [
            1 2 3
            4 5 6
            7 8 9
            10 11 12
            13 14 15
        ],
    )

    @testset "perturbation" begin
        # Test case 1
        base_solution = Solution([1, 2, 3, 4, 5], 64, 0)
        perturbed_solution = perturbation(base_solution, inputs, rng)
        @test perturbed_solution.jobs != base_solution.jobs
    end
end

test_insertJobIntoSequence()
test_PFSP_Heuristic()
test_createBiasedJobsSequence()
test_PFSP_Multistart()
test_localSearch()
test_perturbation()
