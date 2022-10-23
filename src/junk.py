
def get_cheapest_nearby_vertex(penalty_graph, given_vertex, lat_len, long_len):
    row_len, col_len = shape(penalty_graph)
    if (given_vertex[0] == 0):
        if (given_vertex[1] == 0):
            l1r = [0, 1]
            l1t = [1, 0]
            l1rt = [1, 1]

        else:
            l1r = [0, given_vertex[1] + 1]
            l1t = [1, given_vertex[1]]
            l1rt = [1, given_vertex[1] + 1]
            penalty1r = penalty_graph[l1r]
            penalty1t = penalty_graph[l1t]
            peanlty1rt = penalty_graph[l1rt]

    elif (given_vertex[1] == 0):
        l1r = [given_vertex[0], 1]
        l1t = [given_vertex[0] + 1, 0]
        l1rt = [given_vertex[0] + 1, 1]
        penalty1r = penalty_graph[l1r]
        penalty1t = penalty_graph[l1t]
        penalty1rt = penalty_graph[l1rt]
    
    elif (given_vertex[0] == lat_len - 1):
        l1r = [given_vertex[0], given_vertex[1] + 1]
        #no need to account for top right as that is the abort key
    
    elif(given_vertex[1] == long_len - 1):
        l1t = [given_vertex[0] + 1, given_vertex[1]]
    
    else:
        l1r = [given_vertex[0], given_vertex[1] + 1]
        l1t = [given_vertex[0] + 1, given_vertex[1]]
        l1rt = [given_vertex[0] + 1, given_vertex[1] + 1]

    penalty1r = penalty_graph[l1r]
    penalty1t = penalty_graph[l1t]
    penalty1rt = penalty_graph[l1rt]
    penalty_val_array = [penalty1r, penalty1t, penalty1r]
    penalty_ind_array = [l1r, l1t, l1rt]
    min_penalty = min(penalty_val_array)
    min_ind_penalty = penalty_ind_array[penalty_val_array.index(min_penalty)]
    return min_penalty, min_ind_penalty
