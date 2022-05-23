################################################################################
# 6.0002 Spring 2022
# Problem Set 1
# Name: Yiduo Wang
# Collaborators:
# Time:

from state import State

##########################################################################################################
## Problem 1
##########################################################################################################

def load_election(filename):
    """
    Reads the contents of a file, with data given in the following tab-separated format:
    State[tab]Democrat_votes[tab]Republican_votes[tab]EC_votes

    Please ignore the first line of the file, which are the column headers, and remember that
    the special character for tab is '\t'

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a list of State instances
    """
    with open(filename) as f:
        l = f.readlines()
        # print(l)
        newlist = []
        for i in range(1,len(l)):
            ls = l[i].split("\t")
            newlist.append(State(ls[0],ls[1],ls[2],ls[3]))
    return newlist


##########################################################################################################
## Problem 2: Helper functions
##########################################################################################################

def election_winner(election):
    """
    Finds the winner of the election based on who has the most amount of EC votes.
    Note: In this simplified representation, all of EC votes from a state go
    to the party with the majority vote.

    Parameters:
    election - a list of State instances

    Returns:
    a tuple, (winner, loser) of the election i.e. ('dem', 'rep') if Democrats won, else ('rep', 'dem')
    """
    dem = 0
    rep = 0
    for state in election:
        winner = state.get_winner()
        if winner == "dem": 
            dem += state.get_ecvotes()
        else:
            rep +=  state.get_ecvotes()
    if dem > rep:
        return ('dem', 'rep')
    else:
        return ('rep', 'dem')
    


def winner_states(election):
    """
    Finds the list of States that were won by the winning candidate (lost by the losing candidate).

    Parameters:
    election - a list of State instances

    Returns:
    A list of State instances won by the winning candidate
    """
    winning_states = []
    if election_winner(election) == ('dem', 'rep'):
        for i in election: 
            if i.get_winner() == "dem":
                winning_states.append(i)
    if election_winner(election) == ('rep', 'dem'):
        for i in election: 
            if i.get_winner() == "rep":
                winning_states.append(i)
    return winning_states


def ec_votes_to_flip(election, total=538):
    """
    Finds the number of additional EC votes required by the loser to change election outcome.
    Note: A party wins when they earn half the total number of EC votes plus 1.

    Parameters:
    election - a list of State instances
    total - total possible number of EC votes

    Returns:
    int, number of additional EC votes required by the loser to change the election outcome
    """
    winner_ECvotes = 0
    for i in winner_states(election):
        winner_ECvotes += i.get_ecvotes()
    return int(winner_ECvotes -(total/2 - 1))#number of additional EC votes needed by the loser = number of EC votes that should lost by the winner

##########################################################################################################
## Problem 3: Brute Force approach
##########################################################################################################

def combinations(L):
    """
    Helper function to generate powerset of all possible combinations
    of items in input list L. E.g., if
    L is [1, 2] it will return a list with elements
    [], [1], [2], and [1,2].

    DO NOT MODIFY THIS.

    Parameters:
    L - list of items

    Returns:
    a list of lists that contains all possible
    combinations of the elements of L
    """

    def get_binary_representation(n, num_digits):
        """
        Inner function to get a binary representation of items to add to a subset,
        which combinations() uses to construct and append another item to the powerset.

        DO NOT MODIFY THIS.

        Parameters:
        n and num_digits are non-negative ints

        Returns:
            a num_digits str that is a binary representation of n
        """
        result = ''
        while n > 0:
            result = str(n%2) + result
            n = n//2
        if len(result) > num_digits:
            raise ValueError('not enough digits')
        for i in range(num_digits - len(result)):
            result = '0' + result
        return result

    powerset = []
    for i in range(0, 2**len(L)):
        binStr = get_binary_representation(i, len(L))
        subset = []
        for j in range(len(L)):
            if binStr[j] == '1':
                subset.append(L[j])
        powerset.append(subset)
    return powerset

def brute_force_swing_states(winner_states, ec_votes_needed):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states, these are our swing states. Iterate over
    all possible move combinations using the helper function combinations(L).
    Return the move combination that minimises the number of voters moved. If
    there exists more than one combination that minimises this, return any one of them.

    Parameters:
    winner_states - a list of State instances that were won by the winner
    ec_votes_needed - int, number of EC votes needed to change the election outcome

    Returns:
    * A tuple containing the list of State instances such that the election outcome would change if additional
      voters relocated to those states, as well as the number of voters required for that relocation.
    * A tuple containing the empty list followed by zero, if no possible swing states.
    """
    # initialize variables to hold the best combo and the minimum voters moved(minimum_so_far) *so far*
    best_combo = []
    minimum_so_far = float("inf")#allows the first one to be "minimum_so_far"
    # find possible_move_combinations using helper function "combinations"
    possibilities = combinations(winner_states)
    # for combo in possible_move_combinations:
    for i in possibilities: 
        #every time checking a new combo so needs to go back to 0
        new_sum = 0
        voters_moved = 0
    # 	if the sum of new EC votes >= EC votes required and number voters moved < minimum_so_far
        for state in i:
            new_sum += state.get_ecvotes()
            voters_moved += state.get_margin()+1
            
        # print (new_sum,ec_votes_needed,voters_moved,minimum_so_far)
        if new_sum >= ec_votes_needed and voters_moved < minimum_so_far:
        # update best combo and the new minimum voters 
            best_combo = i
            minimum_so_far = voters_moved
# return a tuple of the best combo as a list of the 'swing states' and the number of voters moved, or ([], 0) if there is no move that can be made
    return (best_combo,minimum_so_far)
    

##########################################################################################################
## Problem 4: Dynamic Programming
## In this section we will define two functions, max_voters_moved and min_voters_moved, that
## together will provide a dynamic programming approach to find swing states. This problem
## is analagous to the complementary knapsack problem, you might find Lecture 1 of 6.0002 useful
## for this section of the pset.
##########################################################################################################


def max_voters_moved(winner_states, max_ec_votes):
    """
    Finds the largest number of voters needed to relocate to get at most max_ec_votes
    for the election loser.

    Analogy to the knapsack problem:
        Given a list of states each with a weight(ec_votes) and value(margin+1),
        determine the states to include in a collection so the total weight(ec_votes)
        is less than or equal to the given limit(max_ec_votes) and the total value(voters displaced)
        is as large as possible.

    Parameters:
    winner_states - a list of State instances that were won by the winner
    max_ec_votes - int, the maximum number of EC votes

    Returns:
    * A tuple containing the list of State instances such that the maximum number of voters need to
      be relocated to these states in order to get at most max_ec_votes, and the number of voters
      required required for such a relocation.
    * A tuple containing the empty list followed by zero, if every state has a # EC votes greater
      than max_ec_votes.
    """
    # def helper(winner_states, max_ec_votes, memo = None):
    #     if memo == None:
    #         memo = {}
    #     if (len(winner_states), max_ec_votes) in memo:
    #         result = memo[(len(winner_states), max_ec_votes)]
    #     elif winner_states == [] or max_ec_votes == 0:
    #         result = ([],0)
    #     elif winner_states[0].get_ecvotes() > max_ec_votes:
    #         #Explore right branch only
    #         result = max_ec_votes(winner_states[1:], max_ec_votes, memo)
    #     else:
    #         next_item = winner_states[0]
    #         #Explore left branch
    #         with_val, with_to_take =\
    #                   max_voters_moved(winner_states[1:],
    #                             max_ec_votes - next_item.get_ecvotes(), memo)
            
    #         with_val += next_item.get_margin()+1
    #         #Explore right branch
    #         without_val, without_to_take = max_ec_votes(winner_states[1:],
    #                                                 max_ec_votes, memo)
    #         #Choose better branch
    #         if with_val > without_val:
    #             result = (with_val, with_to_take + (next_item,))
    #         else:
    #             result = (without_val, without_to_take)
    #     memo[(len(winner_states), max_ec_votes)] = result
    #     return result

    # TODO
    pass
     

def min_voters_moved(winner_states, ec_votes_needed):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states. Should minimize the number of voters being relocated.
    Only return states that were originally won by the winner (lost by the loser)
    of the election.

    Hint: This problem is simply the complement of max_voters_moved. You should call
    max_voters_moved with max_ec_votes set to (#ec votes won by original winner - ec_votes_needed)

    Parameters:
    winner_states - a list of State instances that were won by the winner
    ec_votes_needed - int, number of EC votes needed to change the election outcome

    Returns:
    * A tuple containing the list of State instances (which we can call swing states) such that the
      minimum number of voters need to be relocated to these states in order to get at least
      ec_votes_needed, and the number of voters required for such a relocation.
    * * A tuple containing the empty list followed by zero, if no possible swing states.
    """

    # TODO
    pass


##########################################################################################################
## Problem 5
##########################################################################################################


def relocate_voters(election, swing_states, ideal_states = ['AL', 'AZ', 'CA', 'TX']):
    """
    Finds a way to shuffle voters in order to flip an election outcome. Moves voters
    from states that were won by the losing candidate (states not in winner_states), to
    each of the states in swing_states. To win a swing state, you must move (margin + 1)
    new voters into that state. Any state that voters are moved from should still be won
    by the loser even after voters are moved. Also finds the number of EC votes gained by
    this rearrangement, as well as the minimum number of voters that need to be moved.
    Note: You cannot move voters out of Alabama, Arizona, California, or Texas.

    Parameters:
    election - a list of State instances representing the election
    swing_states - a list of State instances where people need to move to flip the election outcome
                   (result of min_voters_moved or brute_force_swing_states)
    ideal_states - a list of Strings holding the names of states where residents cannot be moved from
                   (default states are AL, AZ, CA, TX)

    Return:
    * A tuple that has 3 elements in the following order:
        - an int, the total number of voters moved
        - an int, the total number of EC votes gained by moving the voters
        - a dictionary with the following (key, value) mapping:
            - Key: a 2 element tuple of str, (from_state, to_state), the 2 letter State names
            - Value: int, number of people that are being moved
    * None, if it is not possible to sway the election
    """

    # TODO
    pass


if __name__ == "__main__":
    pass
    # Uncomment the following lines to test each of the problems

    # tests Problem 1
    year = 2012
    election = load_election(f"{year}_results.txt")
    print(len(election))
    print(election[0])

    # tests Problem 2
    winner, loser = election_winner(election)
    won_states = winner_states(election)
    names_won_states = [state.get_name() for state in won_states]
    reqd_ec_votes = ec_votes_to_flip(election)
    print("Winner:", winner, "\nLoser:", loser)
    print("States won by the winner: ", names_won_states)
    print("EC votes needed:",reqd_ec_votes, "\n")

    # tests Problem 3
    brute_election = load_election("2020_results.txt")
    brute_won_states = winner_states(brute_election)
    brute_ec_votes_to_flip = ec_votes_to_flip(brute_election, total=14)
    brute_swing, voters_brute = brute_force_swing_states(brute_won_states, brute_ec_votes_to_flip)
    names_brute_swing = [state.get_name() for state in brute_swing]
    ecvotes_brute = sum([state.get_ecvotes() for state in brute_swing])
    print("Brute force swing states results:", names_brute_swing)
    print("Brute force voters displaced:", voters_brute, "for a total of", ecvotes_brute, "Electoral College votes.\n")

    # # tests Problem 4a: max_voters_moved
    # print("max_voters_moved")
    # total_lost = sum(state.get_ecvotes() for state in won_states)
    # non_swing_states, max_voters_displaced = max_voters_moved(won_states, total_lost-reqd_ec_votes)
    # non_swing_states_names = [state.get_name() for state in non_swing_states]
    # max_ec_votes = sum([state.get_ecvotes() for state in non_swing_states])
    # print("States with the largest margins (non-swing states):", non_swing_states_names)
    # print("Max voters displaced:", max_voters_displaced, "for a total of", max_ec_votes, "Electoral College votes.", "\n")

    # # tests Problem 4b: min_voters_moved
    # print("min_voters_moved")
    # swing_states, min_voters_displaced = min_voters_moved(won_states, reqd_ec_votes)
    # swing_state_names = [state.get_name() for state in swing_states]
    # swing_ec_votes = sum([state.get_ecvotes() for state in swing_states])
    # print("Complementary knapsack swing states results:", swing_state_names)
    # print("Min voters displaced:", min_voters_displaced, "for a total of", swing_ec_votes, "Electoral College votes. \n")

    # # tests Problem 5: relocate_voters
    # print("relocate_voters")
    # flipped_election = relocate_voters(election, swing_states)
    # print("Flip election mapping:", flipped_election)
