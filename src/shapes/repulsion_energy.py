import torch


def repulsion_energy(x, s=1.0):
    """
    Compute the total Riesz (electrostatic) repulsion energy
    of a system of particles.

    Given particle positions x = (x₁, …, x_N) ⊂ ℝᵈ, the energy is

        E(x) = (1 / N²) ∑_{i≠j} φ_s(|x_i - x_j|)

    where the interaction kernel is:
        - s = 0 : φ₀(r) = −log(r)     (2D Coulomb / logarithmic potential)
        - s > 0 : φ_s(r) = r^{-s}    (Riesz potential)

    Self-interactions (i = j) are excluded.

    Parameters
    ----------
    x : torch.Tensor, shape (N, d)
        Particle positions.
    s : float, optional
        Riesz exponent. Use s=0 for logarithmic interaction.

    Returns
    -------
    torch.Tensor (scalar)
        Mean pairwise repulsion energy.
    """
    device, dtype = x.device, x.dtype

    # Pairwise differences: (N, N, d)
    diff = x.unsqueeze(1) - x.unsqueeze(0)

    # Pairwise distances: (N, N)
    dist = torch.linalg.norm(diff, dim=2)

    # Add small diagonal regularization to avoid log(0) / division by zero
    dist = dist + 1e-6 * torch.eye(dist.shape[0], device=device, dtype=dtype)

    # Interaction kernel
    if s == 0:
        interactions = -torch.log(dist)
    else:
        interactions = dist.pow(-s)

    # Remove self-interactions explicitly
    interactions = interactions * (
        1 - torch.eye(interactions.shape[0], device=device, dtype=dtype)
    )

    # Mean over all particle pairs
    return torch.mean(interactions)


def pointwise_repulsion_energy(x, s=1.0):
    """
    Compute pointwise Riesz repulsion energies.

    For each particle x_i, this returns its average interaction
    with all other particles:

        E_i(x) = (1 / N) ∑_{j≠i} φ_s(|x_i - x_j|)

    Parameters
    ----------
    x : torch.Tensor, shape (N, d)
        Particle positions.
    s : float, optional
        Riesz exponent (s=0 gives logarithmic interaction).

    Returns
    -------
    torch.Tensor, shape (N,)
        Repulsion energy associated with each particle.
    """
    device, dtype = x.device, x.dtype

    diff = x.unsqueeze(1) - x.unsqueeze(0)
    dist = torch.linalg.norm(diff, dim=2)

    dist = dist + 1e-6 * torch.eye(dist.shape[0], device=device, dtype=dtype)

    if s == 0:
        interactions = -torch.log(dist)
    else:
        interactions = dist.pow(-s)

    interactions = interactions * (
        1 - torch.eye(interactions.shape[0], device=device, dtype=dtype)
    )

    # Mean interaction per particle
    return torch.mean(interactions, dim=1)