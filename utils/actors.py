class CyclicAgent:
    def __init__(self, n_actions, interaction_interval):
        self.n_actions = n_actions
        self.interaction_interval = interaction_interval
        self.action = 0

    def predict(self, step):
        if step % self.interaction_interval == 0:
            self.action = (self.action + 1) % self.n_actions
        return self.action
