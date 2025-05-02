from torch import nn
import torch

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()
    
class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, : -self.chomp_size].contiguous()

def fgsm_attack(model, x, y, epsilon, loss_fn):
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model: The model to attack
        x: Input data
        y: Target labels
        epsilon: Perturbation size
        loss_fn: Loss function
        
    Returns:
        Adversarial examples
    """
    # Set requires_grad attribute of tensor
    x.requires_grad = True
    
    # Forward pass
    outputs = model(x)
    
    # Calculate loss
    loss = loss_fn(outputs, y)
    
    # Zero all existing gradients
    model.zero_grad()
    
    # Calculate gradients of model in backward pass
    loss.backward()
    
    # Collect gradients
    data_grad = x.grad.data
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_data = x + epsilon * torch.sign(data_grad)
    
    # Clipping for maintaining the valid range (assume data is normalized to [0,1])
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data

def pgd_attack(model, x, y, epsilon, alpha, num_iter, loss_fn, rand_init=True):
    """
    Projected Gradient Descent (PGD) attack.
    
    Args:
        model: The model to attack
        x: Input data
        y: Target labels
        epsilon: Maximum perturbation
        alpha: Step size
        num_iter: Number of iterations
        loss_fn: Loss function
        rand_init: Whether to initialize with random noise
        
    Returns:
        Adversarial examples
    """
    # Make a clone of the input to avoid modifying it
    perturbed_data = x.clone().detach()
    
    # Initialize perturbation randomly if specified
    if rand_init:
        perturbed_data = perturbed_data + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).to(x.device)
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    for _ in range(num_iter):
        perturbed_data.requires_grad = True
        
        # Forward pass
        outputs = model(perturbed_data)
        
        # Calculate loss
        model.zero_grad()
        loss = loss_fn(outputs, y)
        
        # Calculate gradients
        loss.backward()
        
        # Create adversarial example with gradient step
        with torch.no_grad():
            gradient = perturbed_data.grad.sign()
            perturbed_data = perturbed_data.detach() + alpha * gradient
            
            # Project back to epsilon ball
            delta = torch.clamp(perturbed_data - x, -epsilon, epsilon)
            perturbed_data = x + delta
            
            # Ensure valid range
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data

def loss_visualization(train_loss, valid_loss, save_dir_path='/home/hwkang/TSB-AD/figures/loss_evolution', TS_Name=None, AD_Name=None, Encoder_Name=None, hyperparameters: list=[]):
    """
    Visualize training and validation loss.
    
    Args:
        train_loss: Training loss values
        valid_loss: Validation loss values
    """
    import matplotlib.pyplot as plt
    
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{TS_Name}')

    # hyperparameters의 모든 값을 _을 구분자로 붙여서 변수명 hps로 문자열로 저장
    hps = '_'.join([str(hp) for hp in hyperparameters])

    if Encoder_Name is not None:
        plt.savefig(f'{save_dir_path}/{AD_Name}_{Encoder_Name}_{TS_Name}_loss_{hps}.png')
    else:
        plt.savefig(f'{save_dir_path}/{AD_Name}_{TS_Name}_loss_{hps}.png')
    plt.close()

def measure_energy_and_time(function):
    import time
    import pyRAPL
    pyRAPL.setup()
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        m = pyRAPL.Measurement('energy')
        m.start()
        result = function(*args, **kwargs)
        m.stop()
        end = time.perf_counter()
        elapsed = end - start
        energy = m.result.pkg  # energy consumption in Joules
        print(f"Time consumed: {elapsed:.6f} seconds, Energy consumed: {energy} Joules")
        return result, elapsed, energy
    return wrapper

def calculate_output_size(input_size, kernel_size, stride, padding, dilation):
    """
    Calculate the output size of a convolutional layer.
    
    Args:
        input_size: Size of the input
        kernel_size: Size of the kernel
        stride: Stride of the convolution
        padding: Padding added to both sides
        dilation: Dilation rate
        
    Returns:
        Output size after convolution
    """
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

if __name__ == "__main__":
    input_size = 100
    kernel_size = 5
    stride = 1
    padding = kernel_size - 1
    dilation = 1
    chomp_size = kernel_size - 1
    output_size = calculate_output_size(input_size, kernel_size, stride, padding, dilation) - chomp_size
    print(f"Output size: {output_size}")