importall ReinforcementLearning

type GymRandomAgent <: AbsAgent end

function play(agent::GymRandomAgent, state::GymState, env::GymEnv; learn=false)
	return sample(env)
end

#env = GymEnv("Pong-v0")
env = GymEnv("MountainCarContinuous-v0")
agent = GymRandomAgent()
threshold = 1000
for i=1:2
	state = getInitialState(env)
	totalRewards = 0.0
	numOfStates = 1.0

	while !isTerminal(state, env) && numOfStates < threshold
		render_env(env)
		action = play(agent, state, env; learn=false)
		state, reward = transfer(env, state, action)
		totalRewards += reward
		numOfStates += 1
	end

	println("Episode i: $i, rewards: $(totalRewards)")
end
