importall ReinforcementLearning
using Knet

type ReinforceAgent <: AbsAgent
	weights
	prms
	states
	actions
	rewards
	γ
end

function ReinforceAgent(env::GymEnv; α=0.001, γ=0.99)
	w = Dict()
	D = length(getInitialState(env).data)
	O = length(env.actions)

	w["w"] = convert(Array{Float32}, randn(O, D) / sqrt(D))#xavier
	w["b"] = zeros(Float32, O, 1)

	prms = Dict()
	prms["w"] = Adam(w["w"]; lr=α)
	prms["b"] = Adam(w["b"]; lr=α)

	ReinforceAgent(w, prms; γ=γ)
end

ReinforceAgent(w, prms; γ=0.99) = ReinforceAgent(w, prms, Any[], Any[], Array{Float64, 1}(), γ)

function predict(w, b, x)
	linear =  w * x .+ b
end

function sample_action(linear)
	probs = exp(linear) ./ sum(exp(linear), 1)
	c_probs = cumsum(probs)
	return indmax(c_probs .> rand())
end

function loss(w, x, actions)
	linear = predict(w["w"], w["b"], x)
	ynorm = logp(linear,1)
	-sum(actions .* ynorm) / size(actions, 2)
end

lossgradient = grad(loss)

function discount(rewards; γ=0.9)
	discounted = zeros(Float64, length(rewards), 1)
	discounted[end] = rewards[end]

	for i=(length(rewards)-1):-1:1
		discounted[i] = rewards[i] + γ * discounted[i+1]
	end
	return discounted
end

function play(agent::ReinforceAgent, state::GymState, env::GymEnv; learn=false)

	linear = predict(agent.weights["w"], agent.weights["b"], state.data)
	a = sample_action(linear)

	if learn
		push!(agent.states, convert(Array{Float32}, state.data))
		y = zeros(Float32, size(linear, 1), 1)
		y[a] = 1.0
		push!(agent.actions, y)
	end
	return env.actions[a]
end


function observe(agent::ReinforceAgent, state::AbsState, reward::Float64, env::AbsEnvironment; learn=true, terminal=false)
	if learn
		push!(agent.rewards, reward)
		if terminal
			if length(agent.rewards) < 195
				agent.rewards[end] = -50;
			end

			disc_rewards = discount(agent.rewards; γ=agent.γ)
			for i=1:length(agent.states)
				g = lossgradient(agent.weights, agent.states[i], agent.actions[i])
				for k in keys(g)
					update!(agent.weights[k], disc_rewards[i] * agent.γ * g[k], agent.prms[k])
				end
			end

			agent.states = Any[]
			agent.actions = Any[]
			agent.rewards = Array{Float64, 1}()
		end
	end
end

function main()
	srand(123)
	env = GymEnv("CartPole-v0")
	agent = ReinforceAgent(env; α=0.01, γ=0.9)
	threshold = 1000
	rewards = Array{Float64, 1}()

	monitor_start(env, "/home/cano/gym_experiments/cartpole_pg")
	for i=1:1000
		totalRewards, numberOfStates = playEpisode(env, agent; learn=true, threshold = threshold, render=false)
		push!(rewards, totalRewards)
		msg = string("Episode ", i, " , total rewards: ", totalRewards)
		if i >= 100
			sort!(rewards; rev=true)
			m = mean(rewards[1:100])
			msg = string(msg, " , mean: ", m)
			if m >= 195
				break
			end
		end
		println(msg)
	end
	monitor_close(env)
end

main()
