import numpy as np
import torch

def retrieve_policy(policy_file="best_policy.npy"):
    """Loads the stored policy parameters from file without pickling."""
    policy_data = np.load(policy_file, allow_pickle=False)
    
    # For a 0-d structured array, no indexing is needed.
    if policy_data.ndim == 0:
        policy_record = policy_data
    elif policy_data.ndim > 0 and policy_data.shape[0] == 1:
        policy_record = policy_data[0]
    else:
        raise ValueError("Unexpected shape for policy data.")
    
    if not (hasattr(policy_record, "dtype") and policy_record.dtype.names is not None):
        raise ValueError("Expected a structured array with dictionary fields.")
    
    policy_dict = {key: policy_record[key] for key in policy_record.dtype.names}
    return {param: torch.tensor(value, dtype=torch.float32) for param, value in policy_dict.items()}

def policy_action(policy, state):
    """Determines the best action given the policy and current state."""
    # If policy is loaded as a numpy array (or structured array), convert if needed.
    if isinstance(policy, np.ndarray):
        if hasattr(policy, "dtype") and policy.dtype.names is not None:
            policy = {key: policy[key] for key in policy.dtype.names}
        else:
            policy = policy.item()
    
    if not isinstance(policy, dict):
        raise ValueError("Expected a dictionary structure in policy file.")
    
    state_tensor = torch.tensor(state, dtype=torch.float32)
    
    layer1_weights = torch.tensor(policy["mlp_extractor.policy_net.0.weight"], dtype=torch.float32)
    layer1_bias = torch.tensor(policy["mlp_extractor.policy_net.0.bias"], dtype=torch.float32)
    layer2_weights = torch.tensor(policy["mlp_extractor.policy_net.2.weight"], dtype=torch.float32)
    layer2_bias = torch.tensor(policy["mlp_extractor.policy_net.2.bias"], dtype=torch.float32)
    action_layer_weights = torch.tensor(policy["action_net.weight"], dtype=torch.float32)
    action_layer_bias = torch.tensor(policy["action_net.bias"], dtype=torch.float32)
    
    layer1_output = torch.relu(torch.matmul(layer1_weights, state_tensor) + layer1_bias)
    layer2_output = torch.relu(torch.matmul(layer2_weights, layer1_output) + layer2_bias)
    action_logits = torch.matmul(action_layer_weights, layer2_output) + action_layer_bias
    
    chosen_action = torch.argmax(action_logits).item()
    return chosen_action

if __name__ == "__main__":
    policy_data = retrieve_policy()
    sample_input = np.random.randn(8)
    action_result = policy_action(policy_data, sample_input)
    print(f"Selected action: {action_result}")
