class RewardCalculator:
    """Calculate rewards for the reinforcement learning process."""

    def __init__(self, gamma=0.99, lambda_scale=100):
        self.gamma = gamma  # Discount factor
        self.lambda_scale = lambda_scale

    def compute_reward(self, old_obj, new_obj):
        """
        Compute immediate reward based on objective improvement.
        Formula from paper:
        R_t = (Obj(G,P_t) - Obj(G,P_t+1)) / (Obj(G,P_t) + Obj(G,P_t+1)) · λ
        """
        if (old_obj + new_obj).item() == 0:  # Avoid division by zero
            return 0
        return ((old_obj - new_obj) / (old_obj + new_obj)) * self.lambda_scale