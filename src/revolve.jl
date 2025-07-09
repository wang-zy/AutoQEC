function beta(d, t)
    binomial(d+t, t)
end

function set_ncpts(nsteps, t)
    Int64(floor((nsteps*factorial(t))^(1/t)))
end

function find_one_cpt(start_idx, end_idx, c)
    # find the first cpt index within [start_idx, end_idx], with c total cpts available
    # note that c always count start_idx itself as the first cpt
    # c = total_ncpts - current_cpt_idx(which is for start_idx) + 1
    # return index for the first cpt (after start_idx)
    delta = end_idx-start_idx
    if c < 2 || c > delta
        error("wrong value! c=",c)
    end
    old_start_idx = start_idx
    
    # some magic code, references:
    # http://ftp.mcs.anl.gov/pub/tech_reports/reports/P228.pdf
    # https://dl.acm.org/doi/10.1145/347837.347846
    # https://github.com/devitocodes/pyrevolve/blob/master/src/revolve.cpp
    # https://github.com/b45ch1/adol-c/blob/master/ADOL-C/src/revolve.c
    r = 0
    while beta(c,r) < delta
        r += 1
    end
    bino1 = beta(c,r-1)
    bino2 = beta(c-1,r-1)
    if c == 1
        bino3 = 0
    else
        bino3 = beta(c-2,r-1)
    end
    bino4 = Int64(bino2*(r-1)/c)
    if c < 3
        bino5 = 0
    else
        bino5 = beta(c-3,r)
    end

    if delta <= bino1+bino3
        start_idx += bino4
    elseif delta >= beta(c,r)-bino5
        start_idx += bino1
    else
        start_idx = end_idx-bino2-bino3
    end
    
    if start_idx == old_start_idx
        start_idx += 1
    end
    
    start_idx
end

function find_all_cpts(start_idx, end_idx, c)
    # find all cpts index within [start_idx, end_idx], with c total cpts available
    # note that the first cpt is always at start_idx
    # therefore the returned idx_list is only of length c-1
    if c > end_idx-start_idx
        error("wrong value! c=",c)
    end
    
    idx_list = Int64[]
    
    for i = c:-1:2
        start_idx = find_one_cpt(start_idx, end_idx, i)
        push!(idx_list, start_idx)
    end
    idx_list
end

function find_last_cpt(cpt_idx)
    # find index for the last used cpt
    # convension:
    #     cpt being used points to a positive integer time index
    #     cpt not being used points to index -1
    idx = length(cpt_idx)
    for i = length(cpt_idx):-1:1
        if cpt_idx[i] != -1
            break
        else
            idx -= 1
        end
    end
    idx
end

function revolve_schedule(nsteps, t)
    # nsteps: total number of steps
    # t: total number of forward pass, which indicates the time complexity
    # return: schedule sequence [start_cpt_idx, forward_steps, end_cpt_idx, start_idx]
    #         end_cpt_idx == -1 means calculating adjoint in the reverse mode
    #         otherwise save new checkpoint at end_cpt_idx
    
    ncpts = set_ncpts(nsteps, t) # set total number of cpts, which indicates the space complexity
    
    # set all cpts for the first forward pass
    cpt_idx = -1*ones(Int64, ncpts)
    cpt_idx[1] = 0
    cpt_idx[2:end] = find_all_cpts(0, nsteps, ncpts)
    
    seq = Array{Int64,1}[]
    for i = 1:ncpts-1
        push!(seq, [i, cpt_idx[i+1]-cpt_idx[i], i+1, cpt_idx[i]])
    end
    
    final = nsteps
    
    # reverse adjoint to the last cpt iteratively until no more cpts exist
    while find_last_cpt(cpt_idx) != 0
        last_cpt_idx = find_last_cpt(cpt_idx)
        last_cpt = cpt_idx[last_cpt_idx]
        push!(seq, [last_cpt_idx, final-last_cpt, -1, last_cpt])
        
        cpt_idx[last_cpt_idx] = -1
        
        for idx = last_cpt_idx-1:-1:1
            if last_cpt-cpt_idx[idx] == 1
                # if the interval between the second to last cpt and the last cpt is only 1,
                # then the reverse adjoint calculation could be continued
                push!(seq, [idx, 1, -1, cpt_idx[idx]])
                last_cpt = cpt_idx[idx]
                cpt_idx[idx] = -1
            else
                # if the interval is larger than 1, then add more cpts in between before adjoint calculation
                n_add_cpts = min(ncpts-idx+1,last_cpt-cpt_idx[idx]) # >= 2
                cpt_idx[idx+1:idx+n_add_cpts-1] = find_all_cpts(cpt_idx[idx], last_cpt, n_add_cpts)
                for i = idx:idx+n_add_cpts-2
                    push!(seq, [i, cpt_idx[i+1]-cpt_idx[i], i+1, cpt_idx[i]])
                end
                break
            end
        end
        final = last_cpt
    end
    ncpts, seq
end

