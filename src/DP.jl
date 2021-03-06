function iterative_policy_evaluation(env::AbsEnvironment, policy::Policy; Ɣ=0.9, V=nothing, verbose=false)
    states = getAllStates(env)
    if V == nothing
        V = Dict{AbsState, Float64}()
        for s in states; V[s] = 0.0; end
    end
    delta = 1.0

    eps = 1e-30
    threshold = Ɣ < 1.0 ? eps : eps * (1 - Ɣ) / (2*Ɣ)

    iteration = 0

    while delta > threshold
        delta = 0.0
        iteration += 1
        @inbounds for s in states
            v = V[s]
            total = 0.0
            @inbounds for (a,p) in actionsAndProbs(policy, s)
                @inbounds for (sprime, r, pp) in getSuccessors(s, a, env)
                    total += p * pp * (r + Ɣ * V[sprime])
                end
            end
            V[s] = total
            verbose && println("State: $(s), v: $(v), V: $(V[s])")
            delta = max(delta, abs(v - V[s]))
            verbose && println("Delta: $(delta)\n")
        end
    end

    verbose && println("Number of iterations: $(iteration)")
    return V
end

#Returns optimal policy and state values
function policy_iteration(env::AbsEnvironment; Ɣ=0.9, verbose=false, Vs=nothing, Ps=nothing)
    states = getAllStates(env)

    ### INITIALIZATION ###

    #An arbitrary value function and policy
    V = Dict{AbsState, Float64}()
    policy = Policy()

    for s in states
        V[s] = 0.0
        actionSet = getActions(s, env)
        a = shuffle(collect(actionSet))[1]
        policy.mapping[s] = [(a, 1.0)]#action & probability, i.e. deterministic policy
    end

    if verbose
        for s in states
            println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
        end
    end

    policyStable = false
    iteration = 0
    while !policyStable
        ### POLICY EVALUATION ###
        Ps != nothing && push!(Ps, copy(policy.mapping))
        iterative_policy_evaluation(env, policy; Ɣ=Ɣ, V=V, verbose=verbose)
        Vs != nothing && push!(Vs, copy(V))
        ### GREEDY POLICY IMPROVEMENT ###
        policyStable = true
        verbose && println("Iteration: $(iteration)")
        iteration += 1
        for s in states
            aa = policy.mapping[s][1][1]
            m = -1e35
            verbose && println("Before Update - State: $(s), Action: $(aa)")
            for a in getActions(s, env)
                total = 0.0
                for (sprime, r, pp) in getSuccessors(s, a, env)
                    total += pp * (r + Ɣ * V[sprime])
                end
                if total > m
                    policy.mapping[s] = [(a, 1.0)]
                    m = total
                end
            end
            if !(policy.mapping[s][1][1] == aa)
                policyStable = false
            end

            verbose && println("After Update - State: $(s), Action: $(policy.mapping[s][1][1])")
        end
        if verbose
            for s in states
                println("State: $(s), Value: $(V[s]), Action: $(policy.mapping[s][1][1])")
            end

        end
    end
    println("Number of iterations: $iteration")
    policy, V
end

function synchronous_value_iteration(env::AbsEnvironment; Ɣ=0.9, verbose=false, Vs=nothing)
    #Initialize V
    V = Dict{AbsState, Float64}()
    states = getAllStates(env)
    for s in states; V[s] = 0.0; end

    #Value Iteration
    delta = 1.0
    eps = 1e-30
    threshold = Ɣ < 1.0 ? eps : eps * (1 - Ɣ) / (2*Ɣ)
    iteration = 0

    while delta > threshold
        delta = 0.0
        iteration += 1
        copyV = copy(V)
        for s in states
            v = V[s]
            m = -1e35
            for a in getActions(s, env)
                total = 0.0
                for (sprime, r, pp) in getSuccessors(s, a, env)
                    total += pp * (r + Ɣ * V[sprime])
                end

                if total > m
                    m = total
                end
            end
            copyV[s] = m
            verbose && println("State: $(s), v: $(v), V: $(copyV[s])")
            delta = max(delta, abs(v - copyV[s]))
            verbose && println("Delta: $(delta)\n")
        end
        for s in states; V[s] = copyV[s]; end
        Vs != nothing && push!(Vs, copy(V))
    end

    println("Number of iterations: $(iteration)")

    #Deterministic Policy
    policy = Policy()
    for s in states
        m = -1e35
        for a in getActions(s, env)
            total = 0.0
            for (sprime, r, pp) in getSuccessors(s, a, env)
                total += pp * (r + Ɣ * V[sprime])
            end
            if total > m
                policy.mapping[s] = [(a, 1.0)]
                m = total
            end
        end
    end
    return policy, V
end

function gauss_seidel_value_iteration(env::AbsEnvironment; Ɣ=0.9, verbose=false, Vs=nothing)
    #Initialize V
    V = Dict{AbsState, Float64}()
    states = getAllStates(env)
    for s in states; V[s] = 0.0; end
    #Value Iteration
    delta = 1.0
    eps = 1e-30
    threshold = Ɣ < 1.0 ? eps : eps * (1 - Ɣ) / (2*Ɣ)
    iteration = 0

    while delta > threshold
        delta = 0.0
        iteration += 1
        copyV = copy(V)
        updated = Set{AbsState}()
        for s in states
            v = V[s]
            m = -1e35
            for a in getActions(s, env)
                total = 0.0
                for (sprime, r, pp) in getSuccessors(s, a, env)
                    if sprime in updated
                        total += pp * (r + Ɣ * copyV[sprime])
                    else
                        total += pp * (r + Ɣ * V[sprime])
                    end
                end

                if total > m
                    m = total
                    copyV[s] = total
                end
            end
            push!(updated, s)
            verbose && println("State: $(s), v: $(v), V: $(copyV[s])")
            delta = max(delta, abs(v - copyV[s]))
            verbose && println("Delta: $(delta)\n")
        end
        for s in states; V[s] = copyV[s]; end
        Vs != nothing && push!(Vs, copy(V))
    end

    println("Number of iterations: $(iteration)")

    #Deterministic Policy
    policy = Policy()
    for s in states
        m = -1e35
        for a in getActions(s, env)
            total = 0.0
            for (sprime, r, pp) in getSuccessors(s, a, env)
                total += pp * (r + Ɣ * V[sprime])
            end
            if total > m
                policy.mapping[s] = [(a, 1.0)]
                m = total
            end
        end
    end
    return policy, V
end
