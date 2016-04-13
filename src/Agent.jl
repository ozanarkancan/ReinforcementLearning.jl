include("Environment.jl")

abstract AbsAgent

#returns an action
play(agent::AbsAgent, envInfo::EnvironmentInfo) = error("unimplemented")

type RandomAgent <: AbsAgent end
function play(agent::RandomAgent, state::AbsState, reward)
	actionSet = getActions(state)
	return shuffle(actionSet)[1]
end
