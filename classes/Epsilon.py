class Epsilon:
    def __init__(self, epsilon_func, max_epsilon=1.0, min_epsilon=0.05, eval_epsilon=0.0, num_cycles=100, decrement=0.01):
        self.epsilon_func = epsilon_func
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.eval_epsilon = eval_epsilon
        self.num_cycles = num_cycles
        self.decrement = decrement

    def get_epsilon(self, current_cycle):
        """
        Uses the specified epsilon function to calculate the desired epsilon
        """
        if self.epsilon_func == 'linear_anneal':
            return self.get_epsilon_linear_anneal(current_cycle=current_cycle)
        elif self.epsilon_func == 'constant_decrement':
            return self.get_epsilon_constant_decrement(current_cycle=current_cycle)
        elif self.epsilon_func == 'eval':
            return self.get_eval_epsilon()
        raise Exception("This shouldn't happen")

    def get_epsilon_linear_anneal(self, current_cycle):
        """
        Returns an epsilon (probability of taking random action) based on the current cycle using linear annealing
        """
        return max(self.min_epsilon, self.max_epsilon - current_cycle / self.num_cycles * (self.max_epsilon - self.min_epsilon))

    def get_epsilon_constant_decrement(self, current_cycle):
        """
        Returns an epsilon (probability of taking random action) based on the current cycle using a constant decrement
        Cannot go below the minimum epsilon in config
        """
        return max(self.min_epsilon, self.max_epsilon - (current_cycle - 1) * self.decrement)

    def get_eval_epsilon(self):
        return self.eval_epsilon
