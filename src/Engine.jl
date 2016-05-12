include("Agent.jl")
include("Environment.jl")

function playEpisode(env::AbsEnvironment, agent::AbsAgent; verbose=false, randomInitial = false, learn=true, threshold=100000)
	state = randomInitial ? rand(getAllStates(env)) : getInitialState(env)
	totalRewards = 0.0
	numOfStates = 1

	verbose && println("Initial State: $(state)")

	while !isTerminal(state, env) && numOfStates < threshold
		action = play(agent, state, env; learn=learn)
		verbose && println("Action: $(action)")
		state, reward = transfer(env, state, action)
		verbose && println("State: $(state)")
		verbose && println("Reward: $(reward)")
		observe(agent, state, reward, env; learn=learn, terminal = (numOfStates + 1 == threshold || isTerminal(state, env)))
		totalRewards += reward
		numOfStates += 1
	end

	return (totalRewards, numOfStates)
end
