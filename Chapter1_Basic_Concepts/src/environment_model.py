import numpy as np
from typing import List, Tuple, Dict, Union


class GridWorld:
    """
    A modular grid world environment where each component is defined by independent functions.
    This implementation provides a flexible framework for defining grid-based Markov Decision Processes.
    """
    
    def __init__(self, 
                 size: int = 5, 
                 gamma: float = 0.9,
                 actions: List[str] = ['up', 'right', 'down', 'left', 'stay'],
                 forbidden_states: List[int] = [6, 7, 12, 16, 18, 21],
                 target_states: List[int] = [17],
                 r_bound: int = -1,
                 r_forbid: int = -1,
                 r_target: int = 1,
                 r_default: int = 0):
        '''
        Initialize the GridWorld environment with customizable parameters.

        State transition table (size=5 example, grid layout)
        Grid coordinates (row, col) mapping to state index s:
     
        0, 1, 2, 3, 4
        5, 6, 7, 8, 9
        10,11,12,13,14
        15,16,17,18,19
        20,21,22,23,24
        
    
        Parameters:
        -----------
        size : int, default=5
            Grid size (creates a size x size grid)
        gamma : float, default=0.9
            Discount factor for future rewards
        actions : List[str], default=['up', 'right', 'down', 'left', 'stay']
            Available actions in the environment
        forbidden_states : List[int], default=[6, 7, 12, 16, 18, 21]
            States that yield negative reward when entered
        target_states : List[int], default=[17]
            Terminal states that yield positive reward
        r_bound : int, default=-1
            Reward for hitting grid boundaries
        r_forbid : int, default=-1
            Reward for entering forbidden states
        r_target : int, default=1
            Reward for reaching target states
        r_default : int, default=0
            Default reward for normal transitions
        '''
        
        self.set_dimensions(size)
        self.set_gamma(gamma)
        self.define_actions(actions)
        
        self.validate_state_positions(forbidden_states, target_states)
        
        self.define_special_states(forbidden_states, target_states)
        self.set_reward_values(r_bound, r_forbid, r_target, r_default)
        
        self.build_model()

    def validate_state_positions(self, 
                               forbidden_states: List[int], 
                               target_states: List[int]) -> None:
        """
        Validate state positions for correctness
        
        Parameters:
        -----------
        forbidden_states : List[int]
            List of forbidden states
        target_states : List[int]
            List of target states
            
        Raises:
        -------
        ValueError:
            If any of the following issues are found:
            1. State position conflicts (overlap between forbidden and target states)
            2. States exceeding maximum limit (outside range [0, n_states-1])
        """
        # Check for state position conflicts
        forbidden_set = set(forbidden_states)
        target_set = set(target_states)
        conflict = forbidden_set.intersection(target_set)
        
        if conflict:
            raise ValueError(f"State position conflict: Forbidden and target states overlap {conflict}")
        
        # Check if states exceed maximum limit
        max_state = self.n_states - 1
        all_states = forbidden_states + target_states
        
        invalid_states = [s for s in all_states if s < 0 or s > max_state]
        if invalid_states:
            raise ValueError(f"States exceed limit: Valid range is [0, {max_state}], but found {invalid_states}")
        
        # Check if states are integers
        for state in all_states:
            if not isinstance(state, int):
                raise ValueError(f"States must be integers: {state} is of type {type(state)}")
        
        print(f"State validation passed: Forbidden states {forbidden_states}, Target states {target_states}")

    def set_dimensions(self, size: int) -> None:
        """
        Set grid dimensions and compute total number of states.
        
        Parameters:
        -----------
        size : int
            Grid size (size x size)
        """
        self.size = size
        self.n_states = size * size

    def set_gamma(self, gamma: float) -> None:
        """
        Set the discount factor for future rewards.
        
        Parameters:
        -----------
        gamma : float
            Discount factor (0 ≤ gamma ≤ 1)
        """
        self.gamma = gamma

    def define_actions(self, actions: List[str]) -> None:
        """
        Define the action space and count available actions.
        
        Parameters:
        -----------
        actions : List[str]
            List of valid action names
        """
        self.actions = actions
        self.n_actions = len(self.actions)

    def define_special_states(self, 
                            forbidden_states: List[int], 
                            target_states: List[int]) -> None:
        """
        Define special state types (forbidden and target states).
        
        Parameters:
        -----------
        forbidden_states : List[int]
            States that yield negative reward
        target_states : List[int]
            Terminal states that yield positive reward
        """
        self.forbidden_states = forbidden_states
        self.target_states = target_states

    def set_reward_values(self, 
                         r_bound: int, 
                         r_forbid: int, 
                         r_target: int, 
                         r_default: int) -> None:
        """
        Set reward values for different types of state transitions.
        
        Parameters:
        -----------
        r_bound : int
            Reward for boundary collisions
        r_forbid : int
            Reward for entering forbidden states
        r_target : int
            Reward for reaching target states
        r_default : int
            Default reward for normal transitions
        """
        self.r_bound = r_bound
        self.r_forbid = r_forbid
        self.r_target = r_target
        self.r_default = r_default

    def get_reward(self, next_state: int, is_hit_wall: bool) -> int:
        """
        Core reward function: determines reward based on next state and wall collision.
        
        Parameters:
        -----------
        next_state : int
            The state being transitioned to
        is_hit_wall : bool
            Whether the transition hit a grid boundary
            
        Returns:
        --------
        int
            Reward value for the transition
        """
        if is_hit_wall:
            return self.r_bound
        if next_state in self.forbidden_states:
            return self.r_forbid
        if next_state in self.target_states:
            return self.r_target
        return self.r_default

    def transition_logic(self, state: int, action: str) -> Tuple[int, bool]:
        """
        Core state transition logic: computes next state and wall collision.
        
        Parameters:
        -----------
        state : int
            Current state index
        action : str
            Action to be taken
            
        Returns:
        --------
        Tuple[int, bool]
            (next_state, hit_wall) - next state index and whether wall was hit
        """
        row, col = state // self.size, state % self.size
        next_row, next_col = row, col
        hit_wall = False

        if action == 'up':
            if row == 0: 
                hit_wall = True
            else: 
                next_row -= 1
        elif action == 'right':
            if col == self.size - 1: 
                hit_wall = True
            else: 
                next_col += 1
        elif action == 'down':
            if row == self.size - 1: 
                hit_wall = True
            else: 
                next_row += 1
        elif action == 'left':
            if col == 0: 
                hit_wall = True
            else: 
                next_col -= 1
        # 'stay' action: no coordinate change, no wall collision

        next_state = next_row * self.size + next_col
        return next_state, hit_wall

    def build_model(self) -> None:
        """
        Construct the transition probability matrix (P) and reward matrix (R).
        This function iterates through all states and actions to build the MDP model.
        """
        # Initialize matrices: P[s, a, s'] and R[s, a]
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a_idx, action in enumerate(self.actions):
                # Get next state and wall collision status
                s_next, hit_wall = self.transition_logic(s, action)
                
                # Fill transition probability (deterministic model)
                # In this project, we only consider deterministic cases
                self.P[s, a_idx, s_next] = 1.0
                
                # Compute reward for this transition
                self.R[s, a_idx] = self.get_reward(s_next, hit_wall)

    def get_policy_matrices(self, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute policy-projected transition matrix and reward vector.
        
        Parameters:
        -----------
        policy : np.ndarray
            Policy matrix of shape (n_states, n_actions) representing action probabilities
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (P_pi, r_pi) - policy-projected transition matrix and reward vector
        """
        P_pi = np.zeros((self.n_states, self.n_states))
        r_pi = np.zeros(self.n_states)
        
        for s in range(self.n_states):
            for a_idx in range(self.n_actions):
                prob = policy[s, a_idx]
                P_pi[s] += prob * self.P[s, a_idx]
                r_pi[s] += prob * self.R[s, a_idx]
                
        return P_pi, r_pi