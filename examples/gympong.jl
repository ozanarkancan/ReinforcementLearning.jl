using ReinforcementLearning

env = GymEnv("Pong-v0")
agent = RandomAgent()
threshold = 10000
for i=1:5
	state = getInitialState(env)
	totalRewards = 0.0
	numOfStates = 1.0

	
	while !isTerminal(state, env) && numOfStates < threshold
		display(env.env)
		action = play(agent, state, env; learn=false)
		state, reward = transfer(env, state, action)
		totalRewards += reward
		numOfStates += 1
	end

	println("Episode i: $i, rewards: $(totalRewards)")
end
