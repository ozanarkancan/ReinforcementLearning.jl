include("Environment.jl")
include("DP.jl")

abstract AbsAgent

#returns an action
play(agent::AbsAgent, state::AbsState, env::AbsEnvironment; learn=true) = error("play is unimplemented")
observe(agent::AbsAgent, state::AbsState, reward::Float64, env::AbsEnvironment; learn=true, terminal=false) = nothing

type RandomAgent <: AbsAgent end
function play(agent::RandomAgent, state::AbsState, env::AbsEnvironment)
	actionSet = getActions(state, env)
	return shuffle(actionSet)[1]
end

#Q-learning
type QLearner <: AbsAgent
	Qtable::Dict{AbsState, Dict{AbsAction, Float64}}
	Ɣ::Float64
	α::Float64
	ε::Float64

	function QLearner(env::AbsEnvironment; Ɣ=0.9, α=0.8, ε=0.05)
		#Initialize Qtable
		Qtable = Dict{AbsState, Dict{AbsAction, Float64}}()
		for s in getAllStates(env)
			Qtable[s] = Dict{AbsAction, Float64}()
			terminal = isTerminal(s, env)

			for a in getActions(s, env)
				Qtable[s][a] = terminal ? 0 : rand(-5:5)
			end
		end
		new(Qtable, Ɣ, α, ε)
	end
end

function maxQ(agent::QLearner, state::AbsState, env::AbsEnvironment)
	m = -1e32
	action = nothing
	for a in getActions(state, env)
		if agent.Qtable[state][a] > m
			action = a
			m = agent.Qtable[state][a]
		end
	end
	(action, m)
end

let
	lastState = nothing
	lastAction = nothing
	global play
	global observe

	function play(agent::QLearner, state::AbsState, env::AbsEnvironment; learn=true)
		r = rand()
		action = nothing
		if r < agent.ε && learn
			actionSet = getActions(state, env)
			action = shuffle(actionSet)[1]
		else
			action, q = maxQ(agent, state, env)
		end

		lastState = state
		lastAction = action
		return action
	end

	function observe(agent::QLearner, state::AbsState, reward::Float64, env::AbsEnvironment; learn=true, terminal=false)
		if learn
			action, q = maxQ(agent, state, env)
			agent.Qtable[lastState][lastAction] = agent.Qtable[lastState][lastAction] + agent.α * (reward + agent.Ɣ * q - agent.Qtable[lastState][lastAction])
		end
	end
end

type SarsaLearner <: AbsAgent; 
	qlearner
	
	function SarsaLearner(env::AbsEnvironment; Ɣ=0.9, α=0.8, ε=0.05)
		qlearner = QLearner(env; Ɣ=Ɣ, α=α, ε=ε)
	end
end

let
	S = nothing
	A = nothing
	Sprime = nothing
	Aprime = nothing
	
	global play
	global observe

	function play(agent::SarsaLearner, state::AbsState, env::AbsEnvironment; learn=true, observe=false)
		r = rand()
		action = nothing
		if r < agent.ε && learn
			actionSet = getActions(state, env)
			action = shuffle(actionSet)[1]
		else
			action, q = maxQ(agent.qlearner, state, env)
		end

		if observe
			Sprime = state
			Aprime = action
		else
			if S == nothing
				S = state
				A = action
			else
				action = learn ? A : action
			end
		end
		return action
	end

	function observe(agent::SarsaLearner, state::AbsState, reward::Float64, env::AbsEnvironment; learn=true, terminal=false)
		if learn
			action = play(agent, state, env; observe=true)
			agent.qlearner.Qtable[S][A] = agent.Qtable[S][A] + 
				agent.α * (reward + agent.Ɣ * agent.qlearner.Qtable[Sprime][Aprime] - agent.qlearner.Qtable[S][A])

			S = Sprime
			A = Aprime
		end
	end
end

type PolicyAgent <: AbsAgent
	policy::Policy
end

play(agent::PolicyAgent, state::AbsState, env::AbsEnvironment; learn=true) = agent.policy.mapping[state][1][1]
