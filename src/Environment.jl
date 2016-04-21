import Base.==
import Base.isequal
import Base.hash

abstract AbsEnvironment
abstract AbsAction
abstract AbsState

#returns successor states with their probabilities and related rewards
getSuccessors(state::AbsState, env::AbsEnvironment) = error("unimplemented")
getSuccessors(state::AbsState, action::AbsAction, env::AbsEnvironment) = error("unimplemented")
isTerminal(state::AbsState, env::AbsEnvironment) = error("unimplemented")
getActions(state::AbsState, env::AbsEnvironment) = error("unimplemented")

#takes a state and an action, returns an environment info
transfer(env::AbsEnvironment, state::AbsState, action::AbsAction) = error("unimplemented")
getInitialState(env::AbsEnvironment) = error("unimplemented")
getAllStates(env::AbsEnvironment) = error("unimplemented")

==(l::AbsAction, r::AbsAction) = error("unimplemented")
isequal(l::AbsAction, r::AbsAction) = error("unimplemented")
hash(x::AbsAction) = error("unimplemented")

==(l::AbsState, r::AbsState) = error("unimplemented")
isequal(l::AbsState, r::AbsState) = error("unimplemented")
hash(x::AbsState) = error("unimplemented")
