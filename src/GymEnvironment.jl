using OpenAIGym

type GymState <: AbsState; state::OpenAIGym.State; end
type GymAction <: AbsAction; action; end

#=
==(lhs::State, rhs::State) = lhs.id == rhs.id
isequal(lhs::State, rhs::State) = lhs.id == rhs.id
hash(s::State) = hash(s.id)

==(lhs::Action, rhs::Action) = lhs.id == rhs.id
isequal(lhs::Action, rhs::Action) = lhs.id == rhs.id
hash(a::Action) = hash(a.id)
=#

type GymEnv <: AbsEnvironment
	env
	GymEnv(name::AbstractString) = new(Env(name))
end

getActions(s::GymState, env::GymEnv) = map(GymAction, 0:(action_space(env.env)[:n]-1))
getInitialState(env::GymEnv) = GymState(reset(env.env))
isTerminal(state::GymState, env::GymEnv) = state.state.done
function transfer(env::GymEnv, state::GymState, action::GymAction)
	s = step(env.env, action.action)
	return (GymState(s), s.reward)
end
