mutable struct Solution
	jobs::Vector{Int}
	makespan::Int
	time::Float32

	function Solution()
		new([], 0, 0)
	end
	function Solution(jobs, makespan, time)
		new(jobs, makespan, time)
	end
end
