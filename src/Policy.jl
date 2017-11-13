mutable struct Policy
    mapping
    Policy() = new(Dict{AbsState, Array{Tuple{AbsAction, Float64}}}())
    Policy(mapping) = new(mapping)
end

actionsAndProbs(policy::Policy, state::AbsState) = policy.mapping[state]

function prettyprint(policy::Policy)
    str = ""
    for (k,v) in policy.mapping
        str = string(str, k, " => ", v, "\n")
    end
    return str
end
