import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class AdversarialAttack:
    def __init__(self, model, epsilon=0.1, alpha=0.01, iters=10):
        """
        :param model: The target model to attack.
        :param epsilon: Maximum perturbation.
        :param alpha: Step size for iterative attacks.
        :param iters: Number of iterations for PGD/BIM.
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters

    def fgsm(self, images, labels):
        """Fast Gradient Sign Method"""
        images = images.clone().detach().to(images.device)
        labels = labels.clone().detach().to(images.device)
        images.requires_grad = True
        
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        
        data_grad = images.grad.data
        perturbed_image = images + self.epsilon * data_grad.sign()
        
        # We don't clip to [0,1] here because images are likely normalized.
        # Clipping should be done based on the normalization mean/std if needed.
        return perturbed_image.detach()

    def pgd(self, images, labels):
        """Projected Gradient Descent (Stronger version of BIM)"""
        images = images.clone().detach().to(images.device)
        labels = labels.clone().detach().to(images.device)
        ori_images = images.clone().detach()
        
        for i in range(self.iters):
            images.requires_grad = True
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            # Move images in the direction of the gradient
            adv_images = images + self.alpha * images.grad.data.sign()
            
            # Constraint: Stay within epsilon-ball around original images
            eta = torch.clamp(adv_images - ori_images, min=-self.epsilon, max=self.epsilon)
            images = (ori_images + eta).detach()
            
        return images

    def bim(self, images, labels):
        """Basic Iterative Method (I-FGSM) - Same as PGD without random start"""
        return self.pgd(images, labels)

def plot_adversarial_examples(images, adv_images, labels, predictions, adv_predictions, classes, n=5):
    """Utility to visualize attacks"""
    plt.figure(figsize=(15, 6))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min()) # Rescale for visualization
        plt.imshow(img)
        plt.title(f"Orig: {classes[labels[i]]}\nPred: {classes[predictions[i]]}")
        plt.axis('off')
        
        # Adversarial
        plt.subplot(2, n, i + 1 + n)
        adv_img = adv_images[i].permute(1, 2, 0).cpu().numpy()
        adv_img = (adv_img - adv_img.min()) / (adv_img.max() - adv_img.min())
        plt.imshow(adv_img)
        plt.title(f"Adv Pred: {classes[adv_predictions[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
