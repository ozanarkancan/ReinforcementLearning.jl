abstract AbsEnvironment
abstract AbsAction
abstract AbsState

getSuccessors(state::AbsState) = error("unimplemented")
isTerminal(state::AbsState) = error("unimplemented")
getActions(state::AbsState) = error("unimplemented")

#takes a state and an action, returns an environment info
transfer(env::AbsEnvironment, state::AbsState, action::AbsAction) = error("unimplemented")
getInitialState(env::AbsEnvironment) = error("unimplemented")
