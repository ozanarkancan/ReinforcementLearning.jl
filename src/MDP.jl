include("Environment.jl")
import Base.==
import Base.isequal
import Base.hash

type State <: AbsState; id; end
type Action <: AbsAction; id; end

==(lhs::State, rhs::State) = lhs.id == rhs.id
isequal(lhs::State, rhs::State) = lhs.id == rhs.id
hash(a::State) = hash(a.id)

==(lhs::Action, rhs::Action) = lhs.id == rhs.id
isequal(lhs::Action, rhs::Action) = lhs.id == rhs.id
hash(a::Action) = hash(a.id)

type MDP <: AbsEnvironment
	ns#number of states
	na#number of actions
	graph#state + action ->  [state, reward, prob]
	function MDP(ns::Int=3, na::Int=3)
		states = [State(i) for i in 1:ns]
		actions = [Action(i) for i in 1:na]
		graph = Dict()
		for s in states
			numA = rand(1:na)
			indices = randperm(na)
			for i=1:numA
				numS = rand(1:ns)
				indicesS = randperm(ns)
				probs = rand(numS,)
				probs = probs ./ sum(probs)
				if i == 1
					graph[s] = Dict()
				end
				
				for j=1:numS
					if j == 1
						graph[s][actions[indices[i]]] = [(states[indicesS[j]], rand(-100:100), probs[j])]
					else
						push!(graph[s][actions[indices[i]]], (states[indicesS[j]], rand(-100:100), probs[j]))
					end
				end
			end
		end
		new(ns, na, graph)
	end
end

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
		println("")
	end
end

#mdp = MDP(4, 2)
#printmdp(mdp)
