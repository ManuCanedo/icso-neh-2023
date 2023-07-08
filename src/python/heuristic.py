#!/usr/bin/env python3

import numpy as np
import os
import time
import csv
import sys

from inputs import readTests, readInputs
from objects import Solution

BENCHMARK_NEH = True
BENCHMARK_RUNS = 100

t = 0.01


def insertJobIntoSequence(solution, inputs, k, kJob):
    n = len(solution.jobs)
    # Create earliest, tail, and relative completion times structures
    e = np.zeros((n + 2, inputs.nMachines + 1))
    q = np.zeros((n + 2, inputs.nMachines + 1))
    f = np.zeros((n + 2, inputs.nMachines + 1))
    for i in range(n + 1):
        # Compute earliest, tail, and relative completion times values
        for j in range(inputs.nMachines):
            if i < n:
                e[i, j] = max(e[i, j - 1], e[i - 1, j]) + \
                    inputs.times[solution.jobs[i], j]
            if i > 0:
                q[n - i, inputs.nMachines - j - 1] = max(q[n - i, inputs.nMachines - j], q[n - i + 1,
                                                         inputs.nMachines - j - 1]) + inputs.times[solution.jobs[n - i], inputs.nMachines - j - 1]
            f[i, j] = max(f[i, j - 1], e[i - 1, j]) + inputs.times[kJob, j]
    # Find position of minimum makespan
    Mi = np.amax(f + q, axis=1)[:-1]
    index = np.where(Mi[:k + 1] == np.amin(Mi[:k + 1]))[0][0]
    # Insert job in the sequence and update makespan
    solution.jobs.insert(index, kJob)
    solution.makespan = Mi[index]


def PFSP_Heuristic(inputs, jobs):
    solution = Solution()
    solution.jobs = [jobs[0]]
    for i, job in enumerate(jobs[1:], start=1):
        insertJobIntoSequence(solution, inputs, i, job)
    return solution


def createBiasedJobsSequence(jobs, rng):
    jobsCopy = jobs.copy()
    biasedJobs = []
    for i in range(len(jobsCopy)):
        # Use a decreasing triangular probability distribution
        index = int(len(jobsCopy) * (1 - np.sqrt(1 - rng.random())))
        biasedJobs.append(jobsCopy[index])
        jobsCopy.pop(index)
    return biasedJobs


def PFSP_Multistart(inputs, test, rng):
    totalTimes = np.sum(inputs.times, axis=1)
    sortedJobs = np.flip(np.argsort(totalTimes)).tolist()
    nehSolution = PFSP_Heuristic(inputs, sortedJobs)
    if BENCHMARK_NEH:
        return nehSolution

    baseSolution = nehSolution
    nIter = 0
    while baseSolution.makespan >= nehSolution.makespan and nIter < inputs.nJobs:
        nIter += 1
        biasedJobs = createBiasedJobsSequence(sortedJobs, rng)
        newSolution = PFSP_Heuristic(inputs, biasedJobs)
        if newSolution.makespan < baseSolution.makespan:
            baseSolution = newSolution
    return baseSolution


def localSearch(solution, inputs, rng):
    improve = True
    while improve == True:
        improve = False
        for job in rng.choice(solution.jobs, inputs.nJobs, replace=False):
            newSolution = Solution()
            newSolution.jobs = solution.jobs.copy()
            newSolution.makespan = solution.makespan
            # Remove random job from the solution without repetition
            index = newSolution.jobs.index(job)
            newSolution.jobs.remove(job)
            # Insert removed job in the position where the makespan is minimized
            insertJobIntoSequence(newSolution, inputs, index, job)
            if newSolution.makespan < solution.makespan:
                solution = newSolution
                improve = True
    return solution


def perturbation(baseSolution, inputs, rng):
    solution = Solution()
    solution.jobs = baseSolution.jobs.copy()
    solution.makespan = baseSolution.makespan
    # Select two random jobs from the sequence
    aIndex, bIndex = rng.choice(inputs.nJobs, 2, replace=False)
    # Swap the jobs at the two random positions
    solution.jobs[aIndex], solution.jobs[bIndex] = solution.jobs[bIndex], solution.jobs[aIndex]
    if bIndex < aIndex:
        aIndex, bIndex = bIndex, aIndex
    # Insert the left-most swapped job where the makespan is minimized
    aJob = solution.jobs.pop(aIndex)
    insertJobIntoSequence(solution, inputs, aIndex, aJob)
    # Insert the right-most swapped job where the makespan is minimized
    bJob = solution.jobs.pop(bIndex)
    insertJobIntoSequence(solution, inputs, bIndex, bJob)
    return solution


def detExecution(inputs, test, rng):
    # Create a base solution using a randomized NEH approach
    baseSolution = PFSP_Multistart(inputs, test, rng)
    if BENCHMARK_NEH:
        return baseSolution

    baseSolution = localSearch(baseSolution, inputs, rng)
    bestSolution = baseSolution
    print(f"Multistart LS makespan: {bestSolution.makespan}")

    # Start the iterated local search process
    credit = 0
    elapsedTime = 0
    startTime = time.process_time()
    while elapsedTime < (inputs.nJobs * inputs.nMachines * t):
        # Perturb the base solution to find a new solution
        solution = perturbation(baseSolution, inputs, rng)
        solution = localSearch(solution, inputs, rng)
        # Check if the solution is adept to be the new base solution
        delta = solution.makespan - baseSolution.makespan
        if delta < 0:
            credit = -delta
            baseSolution = solution
            if solution.makespan < bestSolution.makespan:
                bestSolution = solution
                bestSolution.time = time.process_time() - startTime
        elif 0 < delta <= credit:
            credit = 0
            baseSolution = solution
        # Update the elapsed time before evaluating the stopping criterion
        currentTime = time.process_time()
        elapsedTime = currentTime - startTime

    return bestSolution


def printSolution(solution, print_solution=False):
    if print_solution:
        print("Jobs: " + ", ".join("{:d}".format(job)
              for job in solution.jobs))
    print("Makespan: {:.2f}".format(solution.makespan))
    print("Time: {:.2f}".format(solution.time))


def benchmark_execution(inputs, test, rng):
    elapsed_times = []
    for _ in range(BENCHMARK_RUNS):
        start_time = time.time()
        solution = detExecution(inputs, test, rng)
        end_time = time.time()
        elapsed_times.append(end_time - start_time)
    return elapsed_times


def write_to_csv(execution_times_dict):
    with open('execution_times.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Instance Name"] + list(execution_times_dict.keys()))
        writer.writerows(zip(*execution_times_dict.values()))


if __name__ == "__main__":
    try:
        base_path = sys.argv[1]
    except IndexError:
        print("Please provide a base path as a command-line argument.")
        sys.exit(1)

    execution_times_dict = {}

    # Read tests from the file
    tests = readTests(os.path.join(base_path, "tests", "test2run.txt"))

    for test in tests:
        # Read inputs for the test inputs
        inputs = readInputs(os.path.join(
            base_path, "inputs"), test.instanceName)
        rng = np.random.default_rng(test.seed)

        print("Python Base: OBD {:s}".format(inputs.name))
        solution = Solution()

        if BENCHMARK_NEH:
            elapsed_times = benchmark_execution(inputs, test, rng)

            if test.instanceName not in execution_times_dict:
                execution_times_dict[test.instanceName] = []
            execution_times_dict[test.instanceName].extend(elapsed_times)
        else:
            # Compute the best deterministic solution
            solution = detExecution(inputs, test, rng)
            printSolution(solution)

    write_to_csv(execution_times_dict)
