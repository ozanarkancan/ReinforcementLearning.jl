type Policy
	mapping
	Policy() = new(Dict{AbsState, Array{Tuple{AbsAction, Float64}}}())
	Policy(mapping) = new(mapping)
end

actionsAndProbs(policy::Policy, state::AbsState) = policy.mapping[state]
