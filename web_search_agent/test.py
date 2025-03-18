import torch

# Set requires_grad=True for tensors you want to compute gradients for
original_weight = torch.randn(512, 1024, requires_grad=True)  # Only if you want to update this too

# LoRA Parameters
lora_rank = 8  # Low-rank dimension
lora_alpha = 1.0  # Scaling factor

# LoRA Matrices with requires_grad=True
lora_A = torch.randn(1024, lora_rank, requires_grad=True)
lora_B = torch.randn(lora_rank, 512, requires_grad=True)

# Input tensor (requires_grad=True only if you need gradients w.r.t. input)
input_tensor = torch.randn(1, 1024)  # Usually doesn't need requires_grad=True

# Forward pass
original_output = torch.matmul(input_tensor, original_weight.T)
lora_output = torch.matmul(torch.matmul(input_tensor, lora_A), lora_B) * lora_alpha
combined_output = original_output + lora_output

# Print shapes to verify
print("Original Weight Shape:", original_weight.shape)
print("LoRA Matrix A Shape:", lora_A.shape)
print("LoRA Matrix B Shape:", lora_B.shape)
print("Input Tensor Shape:", input_tensor.shape)
print("Original Output Shape:", original_output.shape)
print("LoRA Output Shape:", lora_output.shape)
print("Combined Output Shape:", combined_output.shape)

# Create optimizer before backward pass
optimizer = torch.optim.SGD([lora_A, lora_B], lr=0.01)

# Compute loss and backpropagate
fake_loss = torch.sum(combined_output)
fake_loss.backward()

# Now gradients should be available
print("Gradient of lora_A:", lora_A.grad)

# Update the LoRA weights
optimizer.step()

# Zero the gradients for the next iteration
optimizer.zero_grad()