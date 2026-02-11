import torch


def repulsion_energy(x, s=1.0):
    """
    Compute the total Riesz (electrostatic) repulsion energy
    of a system of particles.

    Given particle positions :math:`x = (x_1, \\dots, x_N) \\subset \\mathbb{R}^d`,
    the energy is

    .. math::

        E(x) = \\frac{1}{N^2} \\sum_{i \\neq j} \\varphi_s(\\|x_i - x_j\\|)

    where the interaction kernel is:

    .. math::

        \\varphi_s(r) =
        \\begin{cases}
        -\\log(r), & s = 0 \\\\
        r^{-s}, & s > 0
        \\end{cases}

    Self-interactions :math:`i = j` are excluded.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N, d)``. Particle positions.
    s : float, optional
        Riesz exponent. Use ``s = 0`` for logarithmic interaction.

    Returns
    -------
    torch.Tensor
        Scalar mean pairwise repulsion energy.
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

    For each particle :math:`x_i`, this returns its average interaction
    with all other particles:

    .. math::

        E_i(x) = \\frac{1}{N} \\sum_{j \\neq i} \\varphi_s\\!\\left(\\|x_i - x_j\\|\\right)

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N, d)``. Particle positions.
    s : float, optional
        Riesz exponent. Use :math:`s = 0` for logarithmic interaction.

    Returns
    -------
    torch.Tensor
        Shape ``(N,)``. Repulsion energy associated with each particle.
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