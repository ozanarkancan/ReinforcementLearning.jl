include("Agent.jl")
include("Environment.jl")

function playEpisode(env::AbsEnvironment, agent::AbsAgent; verbose=false)
	state = getInitialState(env)
	totalRewards = 0.0
	numOfStates = 1

	verbose && println("Initial State: $(state)")

	while !isTerminal(state, env)
		action = play(agent, state, env)
		verbose && println("Action: $(action)")
		state, reward = transfer(env, state, action)
		verbose && println("State: $(state)")
		verbose && println("Reward: $(reward)")
		observe(agent, state, reward, env)
		totalRewards += reward
		numOfStates += 1
	end

	return (totalRewards, numOfStates)
end
