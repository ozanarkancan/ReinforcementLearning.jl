include("Environment.jl")

type State <: AbsState; id; end
type Action <: AbsAction; id; end

==(lhs::State, rhs::State) = lhs.id == rhs.id
isequal(lhs::State, rhs::State) = lhs.id == rhs.id
hash(s::State) = hash(s.id)

==(lhs::Action, rhs::Action) = lhs.id == rhs.id
isequal(lhs::Action, rhs::Action) = lhs.id == rhs.id
hash(a::Action) = hash(a.id)

type MDP <: AbsEnvironment
	ns#number of states
	na#number of actions
	graph#state + action ->  [state, reward, prob]
	MDP(ns, na, graph) = new(ns, na, graph)
	function MDP(ns=3, na=3)
		states = [State(i) for i in 1:ns]
		actions = [Action(i) for i in 1:na]
		graph = Dict{State, Dict{Action,Array{Tuple{State, Float64, Float64}}}}()
		for s in states
			numA = rand(1:na)
			indices = randperm(na)
			graph[s] = Dict{Action, Array{Tuple{State, Float64, Float64}}}()
			
			for k=1:numA
				a = actions[indices[k]]
				numS = rand(1:ns)
				indicesS = randperm(ns)
				probs = rand(numS,)
				probs = probs ./ sum(probs)
				
				for j=1:numS
					sprime = states[indicesS[j]]
					if j == 1
						graph[s][a] = [(sprime, rand(-100:100), probs[j])]
					else
						push!(graph[s][a], (sprime, rand(-100:100), probs[j]))
					end
				end
			end
		end
		new(ns, na, graph)
	end
end

getSuccessors(s::State, a::Action, env::MDP) = env.graph[s][a]
getAllStates(env::MDP) = keys(env.graph)
getActions(s::State, env::MDP) = keys(env.graph[s])

function printmdp(mdp::MDP)
	println("Number of states: $(mdp.ns)")
	println("Number of actions: $(mdp.na)\n")

	for s in keys(mdp.graph)
		for a in keys(mdp.graph[s])
			for t in mdp.graph[s][a]
				println("$(s) + $(a) => $(t[1]), reward: $(t[2]), probability: $(t[3])")
			end
			println("")
		end
	end
end
