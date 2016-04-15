include("Environment.jl")
import Base.==
import Base.isEqual
import Base.hash

type State <: AbsState; id; end
type Action <: AbsAction; id; end

==(lhs::State, rhs::State) = lhs.id == rhs.id
isEqual(lhs::State, rhs::State) = lhs.id == rhs.id
hash(a::State) = hash(a.id)

==(lhs::Action, rhs::Action) = lhs.id == rhs.id
isEqual(lhs::Action, rhs::Action) = lhs.id == rhs.id
hash(a::Action) = hash(a.id)

type MDP <: AbsEnvironment
	ns#number of states
	na#number of actions
	graph#state + action ->  [state, reward, prob]
	function MDP(ns=3, na=3)
		states = [State(i) for i in 1:ns]
		actions = [Action(i) for i in 1:na]
		graph = {}
		for s in states
			numA = rand(1:na)
			indices = randperm(na)
			probs = rand(numA)
			probs = probs ./ sum(probs)
			for i=1:numA
				if i == 1
					graph[s] = [(states[rand(1:ns), rand(-100:100), probs[i]])]
				else
					push!(graph[s], )
				end
			end
		end
	end
end
