#!/usr/bin/env python3

import numpy as np

class Test:

	def __init__(self, instanceName, maxTime, nIter, distCrit, betaMin, betaMax, distCand, betaMin2, betaMax2, seed, shortSim, longSim, variance, execType):
		self.instanceName = instanceName
		self.maxTime = int(maxTime)
		self.nIter = int(nIter)
		self.distCrit = distCrit
		self.betaMin = float(betaMin)
		self.betaMax = float(betaMax)
		self.distCand = distCand
		self.betaMin2 = float(betaMin2)
		self.betaMax2 = float(betaMax2)
		self.seed = int(seed)
		self.shortSim = int(shortSim)
		self.longSim = int(longSim)
		self.variance = float(variance)
		self.execType = int(execType)
		self.TYPE_CRITERIA = 0
		self.TYPE_CANDIDATE = 1

class Inputs:

	def __init__(self, name, nJobs, nMachines, times):
		self.name = name
		self.nJobs = nJobs
		self.nMachines = nMachines
		self.times = times

def readTests(fileName):
	tests = []
	with open(fileName) as f:
		for line in f:
			tokens = line.split(" ")
			if "#" not in tokens[0]:
				test = Test(*tokens)
				tests.append(test)
	return tests

def readInputs(path, instance):
	with open(path + "/" + instance + ".txt", "r") as f:
		nJobs = int(f.readline())
		nMachines = int(f.readline())
		_ = f.readline()
		_ = f.readline()
		_ = f.readline()
		times = np.loadtxt(f).transpose()
		inputs = Inputs(instance, nJobs, nMachines, times)
		return inputs
