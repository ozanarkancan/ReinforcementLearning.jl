importall ReinforcementLearning
import Base.==
import Base.isequal
import Base.hash

using ArgParse

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		"--epochs"
			help = "number of epochs"
			default = 20
			arg_type=Int
		"--eps"
			help = "epsilon parameter for the random behaviour"
			default = 0.05
			arg_type = Float64
		"--gamma"
			help = "the discount factor"
			default = 0.9
			arg_type = Float64
		"--alpha"
			help = "learning step"
			default = 0.8
			arg_type = Float64
		"--dims"
			help = "dimensions of the maze"
			nargs='+'
			default = [3,3]
			arg_type = Int
		"--agent"
			help = "Agent type"
			default = "Q"
	end
	return parse_args(s)
end

#recursive backtracking algorithm
function generate_maze(h = 4, w = 4)
	maze = zeros(h, w, 4)
	unvisited = ones(h, w)

	function neighbours(r,c)
		ns = Array{Tuple{Int, Int, Int}, 1}()
		for i=1:4
			if i == 1 && (r - 1) >= 1 && unvisited[r-1, c] == 1
				push!(ns, (r-1, c, 1))
			elseif i == 2 && (c + 1) <= w && unvisited[r, c+1] == 1
				push!(ns, (r, c+1, 2))
			elseif i == 3 && (r+1) <= h && unvisited[r+1, c] == 1
				push!(ns, (r+1, c, 3))
			elseif i == 4 && (c-1) >= 1 && unvisited[r, c-1] == 1
				push!(ns, (r, c-1, 4))
			end
		end
		return shuffle(ns)
	end

	start = rand(1:h*w)
	r = div(start, h) + 1
	r = r > h ? h : r
	c = start % w
	c = c == 0 ? w : c

	stack = Array{Tuple{Int, Int}, 1}()
	curr = (r, c)
	unvisited[r,c] = 0
	while countnz(unvisited) != 0
		ns = neighbours(curr[1], curr[2])
		if length(ns) > 0
			r,c = curr
			rn, cn, d = ns[1]
			push!(stack, (r,c))
			maze[r, c, d] = 1
			dn = d - 2 <= 0 ? d + 2 : d - 2
			maze[rn, cn, dn] = 1
			curr = (rn, cn)
			unvisited[rn, cn] = 0
		elseif length(stack) != 0
			curr = pop!(stack)
		end

	end
	return maze
end

function print_maze(maze)
	h,w,_ = size(maze)
	rows = 2*h + 1
	cols = 2*w + 1

	for i=1:rows
		println("")
		for j=1:cols
			if i == 1 || i == rows || j == 1 || j == cols
				print("#")
			elseif i % 2 == 1 && j % 2 == 1
				print("#")
			elseif i % 2 == 1 && j % 2 == 0
				r = div(i - 1, 2)
				c = div(j, 2)
				if maze[r, c, 3] == 1
					print(" ")
				else
					print("#")
				end
			elseif i % 2 == 0 && j % 2 == 1
				r = div(i, 2)
				c = div(j - 1, 2)
				if maze[r,c,2] == 1
					print(" ")
				else
					print("#")
				end
			else
				print(" ")
			end
			
		end
	end
	print("\n")
end

#=======USAGE==============
h,w=(3,3)
maze = generate_maze(h, w)

print_maze(maze)
=========================#

type MazeState <: AbsState
	loc
end

@enum ActionEnum UP RIGHT DOWN LEFT

type MazeAction <: AbsAction; act::ActionEnum; end

==(lhs::MazeAction, rhs::MazeAction) = lhs.act == rhs.act
isequal(lhs::MazeAction, rhs::MazeAction) = lhs.act == rhs.act
hash(a::MazeAction) = hash(a.act)

==(lhs::MazeState, rhs::MazeState) = (lhs.loc[1] == rhs.loc[1] && lhs.loc[2] == rhs.loc[2])
isequal(lhs::MazeState, rhs::MazeState) = ==(lhs, rhs)
hash(s::MazeState) = hash(s.loc)

type MazeEnv <: AbsEnvironment
	dims
	maze
	start
	goal
	
	MazeEnv(dims) = new(dims, generate_maze(dims[1], dims[2]), (rand(1:dims[1]), 1), (rand(1:dims[1]), dims[2]))
	MazeEnv() = Maze((5, 5))
end

getInitialState(env::MazeEnv) = MazeState(env.start)
getActions(state::MazeState, env::MazeEnv) = [MazeAction(UP), MazeAction(RIGHT), MazeAction(DOWN), MazeAction(LEFT)]

function getAllStates(env::MazeEnv)
	states = MazeState[MazeState((0,0))]
	for i=1:env.dims[1]
		for j=1:env.dims[2]
			push!(states, MazeState((i, j)))
		end
	end
	return states
end

function isTerminal(state::MazeState, env::MazeEnv)
	(state.loc[1] == 0 && state.loc[2] == 0) || 
		(state.loc[1] == env.goal[1] && state.loc[2] == env.goal[2])
end

function getSuccessors(state::MazeState, a::MazeAction, env::MazeEnv)
	successors = Array{Tuple{MazeState, Float64, Float64}, 1}()
	push!(successors, transfer(env, state, a))
	return successors
end

function transfer(env::MazeEnv, state::MazeState, action::MazeAction)
	loc = state.loc
	reward = -1.0
	next_state = nothing

	if (loc[1] == 0 && loc[2] == 0)
		return (state, -10.0 * prod(size(env.maze)[1:2]), 1.0)
	elseif (loc[1] == env.goal[1] && loc[2] == env.goal[2])
		return (state, prod(size(env.maze)[1:2])*10.0, 1.0)
	end

	if action.act == UP
		orientation = 1
	elseif action.act == RIGHT
		orientation = 2
	elseif action.act == DOWN
		orientation = 3
	else
		orientation = 4
	end
	
	if env.maze[loc[1], loc[2], orientation] == 0
		reward = -10.0 * prod(size(env.maze)[1:2])
		next_state = MazeState((0,0))
	else
		if orientation == 1
			loc = (loc[1] - 1, loc[2])
		elseif orientation == 2
			loc = (loc[1], loc[2] + 1)
		elseif orientation == 3
			loc = (loc[1] + 1, loc[2])
		else
			loc = (loc[1], loc[2] - 1)
		end

		next_state = MazeState(loc)
		reward = (loc[1] == env.goal[1] && loc[2] == env.goal[2]) ? prod(size(env.maze)[1:2])*10.0 : -1.0
	end
	return (next_state, reward, 1.0)
end

function agent_experiement(agent, env, optimumReward, threshold, steps, rInitial)
	totalRewards = 0.0
	rewards = Any[]
	nstates = Any[]
	
	@time while totalRewards != optimumReward
		while totalRewards < threshold
			totalRewards, numberOfStates = playEpisode(env, agent; randomInitial=rInitial, threshold = steps)
			push!(rewards, totalRewards)
			push!(nstates, numberOfStates)
		end
		totalRewards, numberOfStates = playEpisode(env, agent; learn=false, threshold = steps)
	end

	println("Training took $(length(rewards)) episodes")
	
	#=
	for i=1:length(rewards)
		println("Epoch: $i, totalReward: $(rewards[i]), nsteps: $(nsteps[i])")
	end
	=#

	totalRewards, numberOfStates = playEpisode(env, agent; learn=false, threshold = steps)
	println("After training: $totalRewards, #states: $numberOfStates")

end

function main()
	args = parse_commandline()
	hm, wm = args["dims"]
	env = MazeEnv((hm, wm))
	
	print_maze(env.maze)
	println("Start: $(env.start)")
	println("Goal: $(env.goal)")
	
	ss = getAllStates(env)

	println("Value Iteration")
	@time policy, V = synchronous_value_iteration(env; Ɣ=args["gamma"], verbose=false)
	
	policyAgent = PolicyAgent(policy)
	optimumReward, numberOfStates = playEpisode(env, policyAgent; threshold = prod(args["dims"]) * 2, verbose=false)

	println("Optimum Reward: $optimumReward")

	#=
	for s in ss
		println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
	end
	=#
	
	println("Policy Iteration")
	@time policy, V = policy_iteration(env::AbsEnvironment; Ɣ=args["gamma"], verbose=false)
	
	policyAgent = PolicyAgent(policy)
	optimumReward, numberOfStates = playEpisode(env, policyAgent; threshold = prod(args["dims"]) * 2)

	println("Optimum Reward: $optimumReward")

	#=
	for s in ss
		println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
	end
	=#

	println("Gauss-Seidel Value Iteration")
	@time policy, V = gauss_seidel_value_iteration(env; Ɣ=args["gamma"], verbose=false)
	
	#=
	for s in ss
		println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
	end
	=#

	policyAgent = PolicyAgent(policy)
	optimumReward, numberOfStates = playEpisode(env, policyAgent; threshold = prod(args["dims"]) * 2)

	println("Optimum Reward: $optimumReward")

	
	threshold = optimumReward - (2 * prod(args["dims"]) * args["eps"])
	steps = prod(args["dims"]) * 2
	
	
	println("\nQ Learning")
	agentQ = QLearner(env;ε=args["eps"], α=args["alpha"], Ɣ=args["gamma"])
	agent_experiement(agentQ, env, optimumReward, threshold, steps, false)

	println("\nSarsa")
	agentSarsa = SarsaLearner(env;ε=args["eps"], α=args["alpha"], Ɣ=args["gamma"])
	agent_experiement(agentSarsa, env, optimumReward, threshold, steps, false)
	
	println("\nMonte Carlo")
	agentMC = MonteCarloAgent(env)
	agent_experiement(agentMC, env, optimumReward, threshold, steps, true)
end

main()
