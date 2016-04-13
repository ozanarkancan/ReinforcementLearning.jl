include("Agent.jl")
include("Environment.jl")

function playEpisode(env::AbsEnvironment, agent::AbsAgent)
	state = getInitialState(env)
	currentInfo = EnvironmentInfo(state, 0, false)
	totalReward = 0.0
	numOfStates = 1

	while !currentInfo.isTerminal
		action = play(agent, currentInfo)
		currentInfo = transfer(currentInfo.state, action)
		totalReward += currentInfo.reward
		numOfStates += 1
	end

	play(agent, currentInfo)
	return (totalreward, numOfStates)
end
