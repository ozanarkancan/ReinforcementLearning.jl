include("Environment.jl")

type Policy
	mapping
	Policy() = new(Dict())
	Policy(mapping) = new(mapping)
end

actionsAndProbs(policy::Policy, state::AbsState) = policy.mapping[state]

function iterative_policy_evaluation(env::AbsEnvironment, policy::Policy; Ɣ=0.9, V=nothing, verbose=false)
	states = getAllStates(env)
	if V == nothing
		V = Dict()
		for s in states; V[s] = 0; end
	end
	delta = 1.0
	threshold = 1e-5
	iteration = 0

	while delta > threshold
		delta = 0.0
		iteration += 1
		for s in states
			v = V[s]
			total = 0.0
			for (a,p) in actionsAndProbs(policy, s)
				for (sprime, r, pp) in getSuccessors(s, a, env)
					total += p * pp * (r + Ɣ * V[sprime])
				end
			end
			V[s] = total
			verbose && println("State: $(s), v: $(v), V: $(V[s])")
			delta = max(delta, abs(v - V[s]))
			verbose && println("Delta: $(delta)\n")
		end
	end

	verbose && println("Number of iterations: $(iteration)")
	return V
end

#Returns optimal policy and state values
function policy_iteration(env::AbsEnvironment; Ɣ=0.9)
	states = getAllStates(env)
	
	### INITIALIZATION ###
	
	#An arbitrary value function and policy
	V = Dict()
	policy = Policy()
	
	for s in states
		V[s] = rand(-5:5)
		actionSet = getActions(s)
		a = shuffle(actionSet)[1]
		policy.mapping[s] = (a, 1.0)#action & probability, i.e. deterministic policy
	end
	
	policyStable = false
	
	while !policyStable
		### POLICY EVALUATION ###
		iterative_policy_evaluation(env, policy; Ɣ, V)

		### GREEDY POLICY IMPROVEMENT ###
		policyStable = true
	end
	policy, V
end

function value_iteration(env::AbsEnvironment)
	error("TODO: value iteration")
end
