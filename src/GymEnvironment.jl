using PyCall
import Base.size

global const gym = PyCall.pywrap(PyCall.pyimport("gym"))

mutable struct GymState <: AbsState; data; done; end
mutable struct GymAction <: AbsAction; action; end

size(s::GymState) = size(s.data)

==(lhs::GymAction, rhs::GymAction) = lhs.action == rhs.action
isequal(lhs::GymAction, rhs::GymAction) = lhs.action == rhs.action
hash(a::GymAction) = hash(a.action)

struct Spec
	id
	nondeterministic
	reward_threshold
	tags
	timestep_limit
	trials
end

struct GymEnv <: AbsEnvironment
	env
	actions
	spec
end

function GymEnv(name::AbstractString)
	env = gym.make(name)
	actions = nothing
	
	if :n in keys(env[:action_space])
		actions = map(GymAction, 0:(env[:action_space][:n]-1))
	elseif :low in keys(env[:action_space])
		actions = (env[:action_space][:low], env[:action_space][:high])
	else#algorithmic
		t = env[:action_space][:shape]
        actions = []
        for i=0:(t[1][1]-1)
            for j=0:(t[2][1]-1)
                for k=0:(t[3][1]-1)
                    push!(actions, GymAction((i, j, k)))
                end
            end
        end
	end

	s = Spec(env[:spec][:id], env[:spec][:nondeterministic], env[:spec][:reward_threshold],
		env[:spec][:tags], env[:spec][:timestep_limit], env[:spec][:trials])
	GymEnv(env, actions, s)
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
sample(env::GymEnv) = GymAction(env.env[:action_space][:sample]())
