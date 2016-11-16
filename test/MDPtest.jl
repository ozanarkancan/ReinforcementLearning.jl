using ReinforcementLearning, Base.Test

mdp = MDP(4, 6)
@test mdp.ns == 4
@test mdp.na == 6

states = getAllStates(mdp)

for i=1:4
	@test (State(i) in states) == true
end

for s in states
	for a in getActions(s, mdp)
		total = 0.0
		for (sprime, reward, prob) in getSuccessors(s, a, mdp)
			@test (prob >= 0.0 && prob <= 1.0)
			total += prob
		end
		@test abs(total - 1.0) <= 1e4
	end
end

ss = [State(1), State(2)]
as = [Action(1), Action(2), Action(3)]

graph = Dict{State, Dict{Action, Array{Tuple{State, Float64, Float64}}}}()
graph[ss[1]] = Dict{Action, Array{Tuple{State, Float64, Float64}}}()
graph[ss[2]] = Dict{Action, Array{Tuple{State, Float64, Float64}}}()
graph[ss[1]][as[1]] = [(ss[1], 5.0, 0.5), (ss[2], 5.0, 0.5)]
graph[ss[1]][as[2]] = [(ss[2], 10.0, 1.0)]
graph[ss[1]][as[3]] = [(ss[1], 0.0, 1.0)]
graph[ss[2]][as[1]] = [(ss[2], -1.0, 1.0)]

mdp = MDP(2, 3, graph)

@test mdp.ns == 2
@test mdp.na == 3

mapping = Dict{State, Array{Tuple{Action, Float64}}}()
mapping[ss[1]] = [(as[1], 1.0)]
mapping[ss[2]] = [(as[1], 1.0)]
policy = Policy(mapping)

V = iterative_policy_evaluation(mdp, policy, Ɣ=0.8; verbose=false)
@test abs(V[ss[1]] - 5) < 1e-4
@test abs(V[ss[2]] - -5) < 1e-4

policy.mapping[ss[1]] = [(as[2], 1.0)]

p, V = policy_iteration(mdp; Ɣ=0.8, verbose=false)
@test policy.mapping[ss[1]][1][1] == p.mapping[ss[1]][1][1]
@test policy.mapping[ss[2]][1][1] == p.mapping[ss[2]][1][1]

p, V = synchronous_value_iteration(mdp; Ɣ=0.8, verbose=false)
@test policy.mapping[ss[1]][1][1] == p.mapping[ss[1]][1][1]
@test policy.mapping[ss[2]][1][1] == p.mapping[ss[2]][1][1]

p, V = gauss_seidel_value_iteration(mdp; Ɣ=0.8, verbose=false)
@test policy.mapping[ss[1]][1][1] == p.mapping[ss[1]][1][1]
@test policy.mapping[ss[2]][1][1] == p.mapping[ss[2]][1][1]
