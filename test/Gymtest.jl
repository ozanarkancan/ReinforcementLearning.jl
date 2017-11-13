using ReinforcementLearning, Base.Test

env = GymEnv("Breakout-v0")

@test length(env.actions) == 4

start = getInitialState(env)
@test isTerminal(start, env) == false
@test size(start.data) == (210, 160, 3)
@test (sample(env) in env.actions) == true
