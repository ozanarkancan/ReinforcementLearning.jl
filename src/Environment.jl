import Base.==
import Base.isequal
import Base.hash

abstract AbsEnvironment
abstract AbsAction
abstract AbsState

#returns successor states with their probabilities and related rewards
getSuccessors(state::AbsState, env::AbsEnvironment) = error("getSuccessors is unimplemented")
getSuccessors(state::AbsState, action::AbsAction, env::AbsEnvironment) = error("getSuccessors is unimplemented")
isTerminal(state::AbsState, env::AbsEnvironment) = error("isTerminal is unimplemented")
getActions(state::AbsState, env::AbsEnvironment) = error("getActions is unimplemented")

#takes a state and an action, returns an environment info
transfer(env::AbsEnvironment, state::AbsState, action::AbsAction) = error("transfer is unimplemented")
getInitialState(env::AbsEnvironment) = error("getInitialState is unimplemented")
getAllStates(env::AbsEnvironment) = error("getAllStates unimplemented")

render_env(env::AbsEnvironment) = error("render_env is unimplemented")

==(l::AbsAction, r::AbsAction) = error("== is unimplemented")
isequal(l::AbsAction, r::AbsAction) = error("isequal is unimplemented")
hash(x::AbsAction) = error("hash is unimplemented")

==(l::AbsState, r::AbsState) = error("== is unimplemented")
isequal(l::AbsState, r::AbsState) = error("isequal is unimplemented")
hash(x::AbsState) = error("hash is unimplemented")
