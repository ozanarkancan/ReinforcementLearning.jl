include("Environment.jl")

type Policy
	mapping
	Policy() = new(Dict{AbsState, Array{Tuple{AbsAction, Float64}}}())
	Policy(mapping) = new(mapping)
end

actionsAndProbs(policy::Policy, state::AbsState) = policy.mapping[state]

function iterative_policy_evaluation(env::AbsEnvironment, policy::Policy; Ɣ=0.9, V=nothing, verbose=false)
	states = getAllStates(env)
	if V == nothing
		V = Dict{AbsState, Float64}()
		for s in states; V[s] = 0; end
	end
	delta = 1.0
	threshold = 1e-7
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
function policy_iteration(env::AbsEnvironment; Ɣ=0.9, verbose=false)
	states = getAllStates(env)
	
	### INITIALIZATION ###
	
	#An arbitrary value function and policy
	V = Dict{AbsState, Float64}()
	policy = Policy()
	
	for s in states
		V[s] = rand(-5:5)
		actionSet = getActions(s, env)
		a = shuffle(collect(actionSet))[1]
		policy.mapping[s] = [(a, 1.0)]#action & probability, i.e. deterministic policy
	end
	
	policyStable = false
	iteration = 1
	while !policyStable
		### POLICY EVALUATION ###
		iterative_policy_evaluation(env, policy; Ɣ=Ɣ, V=V)

		### GREEDY POLICY IMPROVEMENT ###
		policyStable = true
		verbose && println("Iteration: $(iteration)")
		iteration += 1
		for s in states
			aa = policy.mapping[s][1][1]
			m = -100000000.0
			verbose && println("Before Update - State: $(s), Action: $(aa)")
			for a in getActions(s, env)
				total = 0.0
				for (sprime, r, pp) in getSuccessors(s, a, env)
					total += pp * (r + Ɣ * V[sprime])
				end
				if total > m
					policy.mapping[s] = [(a, 1.0)]
					m = total
				end
			end
			if !(policy.mapping[s][1][1] == aa)
				policyStable = false
			end

			verbose && println("After Update - State: $(s), Action: $(policy.mapping[s][1][1])")
		end
	end
	policy, V
end

function value_iteration(env::AbsEnvironment; Ɣ=0.9, verbose=false)
	#Initialize V
	V = Dict{AbsState, Float64}()
	states = getAllStates(env)
	for s in states; V[s] = rand(-5:5); end

	#Value Iteration
	delta = 1.0
	threshold = 1e-7
	iteration = 0

	while delta > threshold
		delta = 0.0
		iteration += 1
		for s in states
			v = V[s]
			m = -100000000.0
			for a in getActions(s, env)
				total = 0.0
				for (sprime, r, pp) in getSuccessors(s, a, env)
					total += pp * (r + Ɣ * V[sprime])
				end

				if total > m
					m = total
					V[s] = total
				end
			end
			verbose && println("State: $(s), v: $(v), V: $(V[s])")
			delta = max(delta, abs(v - V[s]))
			verbose && println("Delta: $(delta)\n")
		end
	end

	verbose && println("Number of iterations: $(iteration)")

	#Deterministic Policy
	policy = Policy()
	for s in states
		m = -100000000.0
		for a in getActions(s, env)
			total = 0.0
			for (sprime, r, pp) in getSuccessors(s, a, env)
				total += pp * (r + Ɣ * V[sprime])
			end
			if total > m
				policy.mapping[s] = [(a, 1.0)]
				m = total
			end
		end
	end
	return policy, V
end
