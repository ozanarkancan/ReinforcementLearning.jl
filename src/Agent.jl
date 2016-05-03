include("Environment.jl")

abstract AbsAgent

#returns an action
play(agent::AbsAgent, state::AbsState, env::AbsEnvironment) = error("play is unimplemented")
observe(agent::AbsAgent, state::AbsState, reward::Float64, env::AbsEnvironment) = nothing

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

	function observe(agent::QLearner, state::AbsState, reward::Float64, env::AbsEnvironment)
		action, q = maxQ(agent, state, env)
		agent.Qtable[lastState][lastAction] = agent.Qtable[lastState][lastAction] + agent.α * (reward + agent.Ɣ * q - agent.Qtable[lastState][lastAction])
	end
end
