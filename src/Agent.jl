include("Environment.jl")

abstract AbsAgent

#returns an action
play(agent::AbsAgent, state::AbsState) = error("unimplemented")

type RandomAgent <: AbsAgent end
function play(agent::RandomAgent, state::AbsState)
	actionSet = getActions(state)
	return shuffle(actionSet)[1]
end
