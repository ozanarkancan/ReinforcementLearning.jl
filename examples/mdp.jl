using ReinforcementLearning

function random_mdp()
	g = Dict()
	s = State(1)

	mdp = MDP(20, 20)
	printmdp(mdp)

	println("\nPolicy Iteration")
	policy, V = policy_iteration(mdp; Ɣ=0.8, verbose=false)

	ss = getAllStates(mdp)
	for s in ss
		println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
	end

	println("\nValue Iteration")
	policy, V = synchronous_value_iteration(mdp; Ɣ=0.8, verbose=true)
	for s in ss
		println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
	end
	
	println("\nGauss-Seidel Value Iteration")
	policy, V = gauss_seidel_value_iteration(mdp; Ɣ=0.8, verbose=true)
	for s in ss
		println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
	end

	V = iterative_policy_evaluation(mdp, policy, Ɣ=0.8; verbose=false)
	
	println("\nPolicy Evaluation")
	for s in ss
		println("State: $(s), Value: $(V[s])")
	end

end

function twostates()
	ss = [State(1), State(2)]
	as = [Action(1), Action(2), Action(3)]

	graph = Dict()
	graph[ss[1]] = Dict()
	graph[ss[2]] = Dict()
	graph[ss[1]][as[1]] = [(ss[1], 5.0, 0.5), (ss[2], 5.0, 0.5)]
	graph[ss[1]][as[2]] = [(ss[2], 10.0, 1.0)]
	graph[ss[1]][as[3]] = [(ss[1], 0.0, 1.0)]
	graph[ss[2]][as[1]] = [(ss[2], -1.0, 1.0)]

	mdp = MDP(2, 3, graph)
	printmdp(mdp)

	mapping = Dict()
	mapping[ss[1]] = [(as[1], 1.0)]
	mapping[ss[2]] = [(as[1], 1.0)]
	policy = Policy(mapping)
	V = iterative_policy_evaluation(mdp, policy, Ɣ=0.8; verbose=false)
	
	println("Policy Evaluation")
	for s in ss
		println("State: $(s), Value: $(V[s])")
	end

	println("Policy Iteration")
	policy, V = policy_iteration(mdp; Ɣ=0.8, verbose=false)
	for s in ss
		println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
	end

	println("Value Iteration")
	policy, V = synchronous_value_iteration(mdp; Ɣ=0.8, verbose=false)
	for s in ss
		println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
	end

end

function main()
	println("*** RANDOM MDP ***")
	random_mdp()
	
	#=
	println("*** Example MDP ***")
	twostates()
	=#
end

main()
