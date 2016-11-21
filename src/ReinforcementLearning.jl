module ReinforcementLearning

include("Environment.jl"); export AbsEnvironment, AbsAction, transfer, getInitialState, getAllStates, AbsState, getActions, isTerminal, getSuccessors
include("Policy.jl"); export Policy, actionsAndProbs
include("Agent.jl"); export AbsAgent, play, observe, RandomAgent, QLearner, SarsaLearner, SarsaLambdaLearner, PolicyAgent, MonteCarloAgent
include("Engine.jl"); export playEpisode
include("MDP.jl"); export MDP, State, Action, printmdp
include("DP.jl"); export Policy, actionsAndProbs, iterative_policy_evaluation, policy_iteration, synchronous_value_iteration, gauss_seidel_value_iteration
include("GymEnvironment.jl"); export GymState, GymAction, GymEnv, render_gym, sample, monitor_start, monitor_close
end # module
