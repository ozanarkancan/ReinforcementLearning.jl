module ReinforcementLearning

include("Agent.jl"); export AbsAgent, play, RandomAgent
include("Environment.jl"); export AbsEnvironment, AbsAction, transfer, getInitialState, AbsState, getActions, isTerminal
include("Engine.jl"); export playEpisode
end # module
