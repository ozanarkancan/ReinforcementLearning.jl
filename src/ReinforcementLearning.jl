module ReinforcementLearning

include("Agent.jl"); export AbsAgent, play, RandomAgent
include("Environment.jl"); export AbsEnvironment, AbsAction, transfer, getInitialState, getAllStates, AbsState, getActions, isTerminal
include("Engine.jl"); export playEpisode
include("MDP.jl"); export MDP, State, Action, printmdp
include("DP.jl"); export Policy, actionsAndProbs, iterative_policy_evaluation, policy_iteration, synchronous_value_iteration
end # module
