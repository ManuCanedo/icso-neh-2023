using DelimitedFiles, Random, LinearAlgebra

struct TestData
    instanceName::String
    maxTime::Int
    nIter::Int
    distCrit::String
    betaMin::Float32
    betaMax::Float32
    distCand::String
    betaMin2::Float32
    betaMax2::Float32
    seed::Int
    shortSim::Int
    longSim::Int
    variance::Float32
    execType::Int
    TYPE_CRITERIA::Int
    TYPE_CANDIDATE::Int

    function TestData(
        instanceName,
        maxTime,
        nIter,
        distCrit,
        betaMin,
        betaMax,
        distCand,
        betaMin2,
        betaMax2,
        seed,
        shortSim,
        longSim,
        variance,
        execType,
    )
        new(
            instanceName,
            parse(Int, maxTime),
            parse(Int, nIter),
            distCrit,
            parse(Float32, betaMin),
            parse(Float32, betaMax),
            distCand,
            parse(Float32, betaMin2),
            parse(Float32, betaMax2),
            parse(Int, seed),
            parse(Int, shortSim),
            parse(Int, longSim),
            parse(Float32, variance),
            parse(Int, execType),
            0,
            1,
        )
    end
end

mutable struct Inputs
    name::String
    nJobs::Int
    nMachines::Int
    times::Array{Int,2}
end

function readTests(file)
    tests = TestData[]
    file = open(file)
    for line in eachline(file)
        if !startswith(line, "#")
            values = split(line)
            if length(values) == 14
                push!(tests, TestData(values...))
            end
        end
    end
    close(file)
    return tests
end

function readInputs(path, instance)
    file = open(joinpath(path, instance * ".txt"), "r")
    nJobs = parse(Int, readline(file))
    nMachines = parse(Int, readline(file))
    for _ = 1:3
        readline(file)
    end
    times = transpose(readdlm(file, ' ', Int))
    inputs = Inputs(instance, nJobs, nMachines, times)
    close(file)
    return inputs
end
