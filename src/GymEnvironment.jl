using PyCall
global const gym = PyCall.pywrap(PyCall.pyimport("gym"))

type GymState <: AbsState; data; done; end
type GymAction <: AbsAction; action; end

#=
==(lhs::State, rhs::State) = lhs.id == rhs.id
isequal(lhs::State, rhs::State) = lhs.id == rhs.id
hash(s::State) = hash(s.id)
=#

==(lhs::GymAction, rhs::GymAction) = lhs.action == rhs.action
isequal(lhs::GymAction, rhs::GymAction) = lhs.action == rhs.action
hash(a::GymAction) = hash(a.action)

type GymEnv <: AbsEnvironment
	env
	actions
end

function GymEnv(name::AbstractString)
	env = gym.make(name)
	actions = nothing
	if :n in keys(env[:action_space])
		actions = map(GymAction, 0:(env[:action_space][:n]-1))
	else
		actions = (env[:action_space][:low], env[:action_space][:high])
	end
	GymEnv(env, actions)
end

getActions(s::GymState, env::GymEnv) = env.actions
getInitialState(env::GymEnv) = GymState(env.env[:reset](), false)
isTerminal(state::GymState, env::GymEnv) = state.done

function transfer(env::GymEnv, state::GymState, action::GymAction)
	obs, reward, done, info = env.env[:step](action.action)
	return (GymState(obs, done), reward)
end

monitor_start(env::GymEnv, fname::AbstractString) = env.env[:monitor][:start](fname)
monitor_close(env::GymEnv) = env.env[:monitor][:close]()
render_env(env::GymEnv) = env.env[:render]()
sample(env) = GymAction(env.env[:action_space][:sample]())
