import sys

class HMM: 
    """
    - class to represent a Hidden Markov model
    
    Constructor Parameters:
        states : list[str]
        observes : list[str]
        trans_p : dict[Transition : float]
        obs_p : dict[Observation : float]
        init_p = dict[str : float]
    """
    #constructor
    def __init__(self, state_s, obs, trans_to_ps, obs_to_ps, initial_ps):
        self.states = state_s
        self.observes = obs
        self.trans_p = trans_to_ps
        self.obs_p = obs_to_ps
        self.init_p = initial_ps

class Transition:
    """
    - class to represent transitions between states
    - used as keys in transitions_to_probs 
      mapping transitions between states to probabilities 

    Parameters:
        origin : str - name of state at time t
        target : str - name of state at time t + 1
    """
    #constructor
    def __init__(self, i, j):
        self.origin = i
        self.target = j

    def __repr__(self):
        return "Transition(" + self.origin + "," + self.target + ")"

    def __eq__(self, obj):
        return isinstance(obj, Transition)\
            and obj.origin == self.origin\
            and obj.target == self.target

    def __hash__(self):
        return hash(str(self))

class Observation:
    """
    - class to represent observations at given states
    - used as keys in observs_to_probs 
      mapping transitions between states to probabilities 

    Parameters:
        state : str - name of state at time t
        observe : str - name of state at time t + 1
    """
    #constructor
    def __init__(self, st, ob):
        self.state = st
        self.observe = ob
    
    def __repr__(self):
        return "Observation(" + str(self.state) + "," + str(self.observe) + ")"

    def __eq__(self, obj):
        return isinstance(obj, Observation)\
            and obj.state == self.state\
            and obj.observe == self.observe

    def __hash__(self):
        return hash(str(self))

def parse_args():
    """
    Parses input system arguments:
        1 - filepath
        2 - filepath to write result to

    Parameters: NA

    Returns: 
        distances : list(str) representing pairwise distances
        write_to_file : str is filepath to write result to
    """
    config_lines = open(sys.argv[1],'r').readlines()
    obs = [obs.strip() for obs in open(sys.argv[2], 'r').readline()]
    return config_lines, obs

#def config_to_dict(config_line, states, Obs_or_Trans):
    """
    
    """
    rows = config_line.strip().split(";")
    matrix = [row.split(",") for row in rows]
    new_dict = {}
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            arg1 = states[row]
            arg2 = states[col]
            probability = matrix[row][col]
            new_dict[eval(Obs_or_Trans(arg1, arg2))] = float(probability.strip())

    return new_dict

def get_HMM(config_list):
    """
    Consumes the system argument config_list, 
    Returns a HMM object representing this configuration

    Parameter: config_list : list[str]

    Returns: 
        HMM with fields:
            states, 
            observes, 
            inits_to_probs : dict[str : float], 
            transitions_to_probs : dict[Transition : float], 
            observs_to_probs : dict[Observation : float], 

    """
    #states_list = config_list[1].split(",")
    #states = {}
    #for i in range(len(states_list)):
     #   states[states_list[i]] = i
    states = [state.strip() for state in config_list[1].split(",")]
    observations = [obs.strip() for obs in config_list[2].split(",")]
    inits_to_probs = {}

    init_list = config_list[3].strip().split(",")
    for i in range(len(init_list)):
        inits_to_probs[states[i]] = float(init_list[i])

    #transitions_to_probs = config_to_dict(config_list[4], states, "Transition")
    transition_rows = config_list[4].strip().split(";")
    transition_matrix = [row.split(",") for row in transition_rows]
    transitions_to_probs = {}
    for row in range(len(transition_matrix)):
        for col in range(len(transition_matrix[row])):
            origin = states[row]
            target = states[col]
            probability = transition_matrix[row][col]
            transitions_to_probs[Transition(origin, target)] = float(probability.strip())
    #observs_to_probs = config_to_dict(config_list[5], states, "Observation")
    observation_rows = config_list[5].strip().split(";")
    observation_matrix = [row.split(",") for row in observation_rows]
    observs_to_probs = {}
    for row in range(len(observation_matrix)):
        for col in range(len(observation_matrix[row])):
            state = states[row]
            obs = observations[col]
            probability = observation_matrix[row][col]
            observs_to_probs[Observation(state, obs)] = float(probability.strip())
            
    return HMM(states, observations, transitions_to_probs, observs_to_probs, inits_to_probs)

def argmax(list):
    """
    Returns the index of the first occurence of the maximum argument

    Parameter:
        list : list<T>
    Returns: 
        max_i : int
    """
    max_i = 0
    for i in range(len(list)):
        if list[i] > list[max_i]:
            max_i = i
    return max_i

def recursive_scores(t, hmm, state_j, scores):
    """
    Given a current time t, a state to calculate previous scores for, 
    the DP table of scores, and the dictionary of Transitions mapped to their probabilities,
    returns the list of all probabilities of arriving at state_j

    Parameters:
        t : int - the current time of the HMM. indexes the observed sequence.
        hmm : HMM - the hidden markov model
        state_j : str - the name of the current state
        scores : list[list[float]] - each outer list represents the scores for a state, 
                                     each inner list represents the scores indexed by time

    Returns:
        all_recursive_scores : list[float] - a list of the recursive scores for t using scores at t - 1
    """
    prev_score_i = lambda i, t : scores[i][t - 1]
    a_ij = lambda state_i : hmm.trans_p[Transition(state_i, state_j)]
    all_recursive_scores = [prev_score_i(i, t) * a_ij(hmm.states[i]) for i in range(len(hmm.states))]
    return all_recursive_scores

def viterbi_recurrence(hmm, observed_seq, scores, backtracking_mat):
    """
    Updates the DP scores and backtracking_matrix to find the maximum probability sequence of states

    Parameters:
        hmm : HMM - hidden markov model object
        observed_seq : list[str] - the list of observed characters
        scores : list[list[float]] - each outer list represents the scores for a state, 
                                     each inner list represents the scores indexed by time
        backtracking_mat : list[list[int]] - initialized as [[0]...[0]] of len(observed_seq)

    Returns: 
        tuple of (scores, backtracking_mat)
        scores : list[list[float]] - each outer list represents the scores for a state, 
                                    each inner list represents the scores indexed by time
        backtracking_mat : list[list[int]] - 
                                    matrix indicating where each score found its recursive component

    """
    for t in range(1, len(observed_seq)):
        for j in range(len(hmm.states)):
            state_seq_prob = hmm.obs_p[Observation(hmm.states[j], observed_seq[t])]
            all_recursive_scores = recursive_scores(t, hmm, hmm.states[j], scores)

            max_rec_score = max(all_recursive_scores)
            max_i = argmax(all_recursive_scores)

            scores[j].append(max_rec_score * state_seq_prob)
            backtracking_mat[j].append(max_i)

    return scores, backtracking_mat

def backtrack(states, final_i, bt_mat):
    """
    Finds the path of states which led to the highest-probability score

    Parameters:
        states : list[str] - the list of states in the HMM
        final_i : int - index of final state
        bt_mat : list[list[int]] - the backtracking matrix. 
                                    same shape as the scores matrix
    
    Returns: 
        state_sequence : str - string representing 

    """
    path = states[final_i]
    pointer = bt_mat[final_i][-1]
    #print("seq_i" + str(seq_i))
    for t in range(len(bt_mat[0]) - 1, 0, -1):
        path = states[pointer] + path
        pointer = bt_mat[pointer][t - 1]
    return path

def main(config_list, obs_seq):
    """
    Prints the most-likely sequence of hidden states given the observed state emissions
    Prints the probability of this sequence

    Parameters:
        config_list : list[str] - HMM configuration information
        obs : str - the observed sequence string 

    Side effects:
        Prints to stdout the sequence and its probability
    """
    hmm = get_HMM(config_list)
    if len(obs_seq) > 0:
        initial_scores = [[hmm.init_p[state] * hmm.obs_p[Observation(state, obs_seq[0])]] for state in hmm.states]
        #print("initial_scores: " + str(initial_scores))
        backtracking_mat = [[0] for _ in hmm.states]

        scores, backtracking_mat1 = viterbi_recurrence(hmm, obs_seq, initial_scores, backtracking_mat)

        final_scores = [state_scores[-1] for state_scores in scores]
        #print("backtracking: " + str(backtracking_mat1))
        final_state_i = argmax(final_scores)
            
        state_sequence = backtrack(hmm.states, final_state_i, backtracking_mat1)
        seq_prob = max(final_scores)
        rounded_e_prob = "{:.4e}".format(seq_prob)
    else:
        state_sequence = ""
        rounded_e_prob = "1.0000e+00"
    sys.stdout.write(state_sequence + "\n" + rounded_e_prob)
    
if __name__ == "__main__":
    config_list, obs = parse_args()
    main(config_list, obs)