importall ReinforcementLearning
using Knet, ArgParse

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
	linear = w * x .+ b
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
				agent.rewards[end] = -100;
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

function main(args=ARGS)
	s = ArgParseSettings()
	s.description="Solution with REINFORCE algorithm"
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
		("--lr"; arg_type=Float64; default=0.01; help="learning rate")
		("--gamma"; arg_type=Float64; default=0.9; help="discount rate")
		("--env"; default="CartPole-v0"; help="name of the environment")
		("--render"; action=:store_true; help="display the env")
		("--monitor"; default=""; help="path of the log for the experiment, empty if mointoring is disabled")
		("--epoch"; arg_type=Int; default=300; help="number of epochs")
		("--threshold"; arg_type=Int; default=1000; help="stop the episode even it is not terminal after number of steps exceeds the threshold")
	end
	srand(123)
	o = parse_args(args, s)
	env = GymEnv(o["env"])
	agent = ReinforceAgent(env; α=o["lr"], γ=o["gamma"])
	rewards = Array{Float64, 1}()

	o["monitor"] != "" && monitor_start(env, o["monitor"])
	for i=1:o["epoch"]
		totalRewards, numberOfStates = playEpisode(env, agent; learn=true, threshold = o["threshold"], render=o["render"])
		push!(rewards, totalRewards)
		msg = string("Episode ", i, " , total rewards: ", totalRewards)
		if i >= 100
			m = mean(rewards[(i-100+1):end])
			msg = string(msg, " , mean: ", m)
			if m >= 195
				break
			end
		end
		println(msg)
	end
	o["monitor"] != "" && monitor_close(env)
end

main()
