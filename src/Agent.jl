include("Environment.jl")

abstract AbsAgent

#returns an action
play(agent::AbsAgent, state::AbsState, env::AbsEnvironment) = error("unimplemented")

type RandomAgent <: AbsAgent end
function play(agent::RandomAgent, state::AbsState, env::AbsEnvironment)
	actionSet = getActions(state, env)
	return shuffle(actionSet)[1]
end
