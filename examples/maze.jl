importall ReinforcementLearning
import Base.==
import Base.isequal
import Base.hash

#recursive backtracking algorithm
function generate_maze(h = 4, w = 4)
	maze = zeros(h, w, 4)
	unvisited = ones(h, w)

	function neighbours(r,c)
		ns = Any[]
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

	stack = Any[]
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
	orientation
end

@enum Action MOVE LEFT RIGHT

type MazeAction <: AbsAction; act::Action; end

==(lhs::MazeAction, rhs::MazeAction) = lhs.act == rhs.act
isequal(lhs::MazeAction, rhs::MazeAction) = lhs.act == rhs.act
hash(a::MazeAction) = hash(a.act)

==(lhs::MazeState, rhs::MazeState) = (lhs.loc[1] == rhs.loc[1] && lhs.loc[2] == rhs.loc[2] && lhs.orientation == rhs.orientation)
isequal(lhs::MazeState, rhs::MazeState) = ==(lhs, rhs)
hash(s::MazeState) = hash([s.loc; s.orientation])

type MazeEnv <: AbsEnvironment
	dims
	maze
	start
	goal
	
	MazeEnv(dims) = new(dims, generate_maze(dims[1], dims[2]), (rand(1:dims[1]), 1), (rand(1:dims[1]), dims[2]))
	MazeEnv() = Maze((5, 5))
end

getInitialState(env::MazeEnv) = MazeState(env.start, 1)
getActions(state::MazeState, env::MazeEnv) = [MazeAction(MOVE), MazeAction(LEFT), MazeAction(RIGHT)]

function getAllStates(env::MazeEnv)
	states = MazeState[MazeState((0,0), 1), MazeState((0,0), 2), MazeState((0,0), 3), MazeState((0,0), 4)]
	for i=1:env.dims[1]
		for j=1:env.dims[2]
			for a=1:4
				push!(states, MazeState((i, j), a))
			end
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
	orientation = state.orientation
	reward = -1.0
	next_state = nothing

	if (loc[1] == 0 && loc[2] == 0)
		return (state, -1000.0, 1.0)
	end

	if action.act == LEFT
		orientation = orientation == 1 ? 4 : orientation - 1
		next_state = MazeState(loc, orientation)
	elseif action.act == RIGHT
		orientation = orientation == 4 ? 1 : orientation + 1
		next_state = MazeState(loc, orientation)
	else#move

		if env.maze[loc[1], loc[2], orientation] == 0
			reward = -500.0
			next_state = MazeState((0, 0), orientation)
		else
			if orientation == 1
				loc = (loc[1] -1, loc[2])
			elseif orientation == 2
				loc = (loc[1], loc[2] + 1)
			elseif orientation == 3
				loc = (loc[1] + 1, loc[2])
			else
				loc = (loc[1], loc[2] - 1)
			end

			next_state = MazeState(loc, orientation)
			reward = (loc[1] == env.goal[1] && loc[2] == env.goal[2]) ? 1000.0 : -1.0
		end
	end
	return (next_state, reward, 1.0)
end

function main()
	env = MazeEnv((5,5))
	print_maze(env.maze)
	
	#==
	ss = getAllStates(env)

	@time policy, V = synchronous_value_iteration(env; Ɣ=0.9, verbose=true)
	
	println("Start: $(env.start)")
	println("Goal: $(env.goal)")

	println("Value Iteration")

	for s in ss
		println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
	end

	@time policy, V = policy_iteration(env::AbsEnvironment; Ɣ=0.9, verbose=false)

	println("Policy Iteration")
	
	for s in ss
		println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
	end
	
	println("Gauss-Seidel Value Iteration")
	@time policy, V = gauss_seidel_value_iteration(env; Ɣ=0.9, verbose=true)

	for s in ss
		println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
	end

	==#
	
	#agent = QLearner(env)
	agent = SarsaLearner(env)

	numberOfEpochs = 50
	rewards = Any[]

	println("Training")
	
	@time for i=1:numberOfEpochs
		#totalRewards, numberOfStates = playEpisode(env, agent; verbose = true)
		totalRewards, numberOfStates = playEpisode(env, agent)
		push!(rewards, totalRewards)
		println("Epoch: $i, totalReward: $totalRewards")
	end

	println("Testing")
	@time for i=1:numberOfEpochs
		#totalRewards, numberOfStates = playEpisode(env, agent; verbose = true)
		totalRewards, numberOfStates = playEpisode(env, agent; learn=false)
		push!(rewards, totalRewards)
		println("Epoch: $i, totalReward: $totalRewards")
	end

end

main()
