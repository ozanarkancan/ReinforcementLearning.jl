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
	lastState
	lastAction

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
		new(Qtable, Ɣ, α, ε, nothing, nothing)
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

function play(agent::QLearner, state::AbsState, env::AbsEnvironment; learn=true)
	r = rand()
	action = nothing
	if r < agent.ε && learn
		actionSet = getActions(state, env)
		action = shuffle(actionSet)[1]
	else
		action, q = maxQ(agent, state, env)
	end

	agent.lastState = state
	agent.lastAction = action
	return action
end

function observe(agent::QLearner, state::AbsState, reward::Float64, env::AbsEnvironment; learn=true, terminal=false)
	if learn
		action, q = maxQ(agent, state, env)
		agent.Qtable[agent.lastState][agent.lastAction] = agent.Qtable[agent.lastState][agent.lastAction] + 
			agent.α * (reward + agent.Ɣ * q - agent.Qtable[agent.lastState][agent.lastAction])
		if terminal
			agent.lastState = nothing
			agent.lastAction = nothing
		end
	end
end

type SarsaLearner <: AbsAgent; 
	qlearner::QLearner
	S
	A
	Sprime
	Aprime
	
	function SarsaLearner(env::AbsEnvironment; Ɣ=0.9, α=0.8, ε=0.05)
		ql = QLearner(env; Ɣ=Ɣ, α=α, ε=ε)
		new(ql, nothing, nothing, nothing, nothing)
	end
end

function play(agent::SarsaLearner, state::AbsState, env::AbsEnvironment; learn=true, observe=false)
	r = rand()
	action = nothing
	if r < agent.qlearner.ε && learn
		actionSet = getActions(state, env)
		action = shuffle(actionSet)[1]
	else
		action, q = maxQ(agent.qlearner, state, env)
	end

	if observe
		agent.Sprime = state
		agent.Aprime = action
	else
		if agent.S == nothing
			agent.S = state
			agent.A = action
		else
			action = learn ? agent.A : action
		end
	end
	return action
end

function observe(agent::SarsaLearner, state::AbsState, reward::Float64, env::AbsEnvironment; learn=true, terminal=false)
	if learn
		action = play(agent, state, env; observe=true)
		agent.qlearner.Qtable[agent.S][agent.A] = agent.qlearner.Qtable[agent.S][agent.A] + 
			agent.qlearner.α * (reward + agent.qlearner.Ɣ * agent.qlearner.Qtable[agent.Sprime][agent.Aprime] - agent.qlearner.Qtable[agent.S][agent.A])
		if terminal
			agent.S = nothing
			agent.A = nothing
			agent.Sprime = nothing
			agent.Aprime = nothing
		else
			agent.S = agent.Sprime
			agent.A = agent.Aprime
		end
	end
end

type PolicyAgent <: AbsAgent
	policy::Policy
end

play(agent::PolicyAgent, state::AbsState, env::AbsEnvironment; learn=true) = agent.policy.mapping[state][1][1]

type MonteCarloAgent <: AbsAgent
	Qtable::Dict{AbsState, Dict{AbsAction, Float64}}
	policy::Policy
	returns::Dict{AbsState, Dict{AbsAction, Array{Float64, 1}}}
	rewards::Array{Float64, 1}
	path::Array{Tuple{AbsState, AbsAction}, 1}

	function MonteCarloAgent(env::AbsEnvironment)
		Qtable = Dict{AbsState, Dict{AbsAction, Float64}}()
		returns = Dict{AbsState, Dict{AbsAction, Array{Float64, 1}}}()
		policy = Policy()
		rewards = Array{Float64, 1}()
		path = Array{Tuple{AbsState, AbsAction}, 1}()

		for s in getAllStates(env)
			Qtable[s] = Dict{AbsAction, Float64}()
			returns[s] = Dict{AbsAction, Array{Float64, 1}}()

			terminal = isTerminal(s, env)
			as = getActions(s, env)
			a = shuffle(collect(as))[1]
			policy.mapping[s] = [(a, 1.0)]

			for a in as
				returns[s][a] = Array{Float64, 1}()
				Qtable[s][a] = rand(-5:5)
			end
		end

		new(Qtable, policy, returns, rewards, path)
	end
end


function play(agent::MonteCarloAgent, state::AbsState, env::AbsEnvironment; learn=true)
	action = nothing
	if length(agent.path) == 0 && learn
		as = getActions(state, env)
		action = shuffle(collect(as))[1]
	else
		action = agent.policy.mapping[state][1][1]
	end
	push!(agent.path, (state, action))
	return action
end

function observe(agent::MonteCarloAgent, state::AbsState, reward::Float64, env::AbsEnvironment; learn=true, terminal=false)
	if learn
		push!(agent.rewards, reward)

		if terminal
			visited = Set{Tuple{AbsState, AbsAction}}()
			g = sum(agent.rewards)
			for t in agent.path
				if ! (t in visited)
					push!(agent.returns[t[1]][t[2]], g)
					agent.Qtable[t[1]][t[2]] = mean(agent.returns[t[1]][t[2]])
				end
			end

			#update the policy
			for t in agent.path
				m = -1e32
				action = nothing
				for a in getActions(t[1], env)
					if agent.Qtable[t[1]][a] > m
						m = agent.Qtable[t[1]][a]
						action = a
					end
				end
				agent.policy.mapping[t[1]] = [(action, 1.0)]
			end

			agent.rewards = Array{Float64, 1}()
			agent.path = Array{Tuple{AbsState, AbsAction}, 1}()
		end
	end
end
