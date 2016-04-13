module ReinforcementLearning

include("Agent.jl"); export AbsAgent, play, RandomAgent
include("Environment.jl"); export AbsEnvironment, AbsAction, transfer, getInitialState, AbsState, getActions
include("Engine"); export playEpisode
end # module
