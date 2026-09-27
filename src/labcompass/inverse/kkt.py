import torch

__all__ = ["KKTConditions"]


class KKTConditions:
    """Initializes :class:`KKTConditions`, implementing (a least-squares approximation of) the KKT stationarity conditions for a set of inequality constraints.

    Given a batch of terminal states, this class estimates the Lagrange multipliers of the active
    inequality constraints and builds the corresponding Lagrangian, so that it can be used as a
    (locally) constrained objective during guided sampling/optimization.

    :param loss_fn: The primal objective function, evaluated on a batch of terminal states and
        returning a per-sample scalar.
    :type loss_fn: class:`Callable[[Tensor], Tensor]`

    :param ineq_constraints: List of inequality constraint functions `g_i`, each evaluated on a batch
        of terminal states and returning a per-sample constraint vector `g_i(x1) <= 0`.
    :type ineq_constraints: class:`list[Callable[[Tensor], Tensor]]`

    :param eps: Small value added to the diagonal of the linear system solved for the Lagrange
        multipliers, for numerical stability, defaults to `1e-6`.
    :type eps: class:`float`

    :param use_lstsq: Whether to solve the multiplier system with :func:`torch.linalg.lstsq` instead
        of explicit matrix inversion, defaults to `True`.
    :type use_lstsq: class:`bool`

    :param g_tol: Tolerance below which a constraint is considered active, i.e. `|g_i(x1)| <= g_tol`,
        defaults to `1e-6`.
    :type g_tol: class:`float`
    """

    def __init__(
        self,
        loss_fn,
        ineq_constraints,
        eps=1e-6,
        use_lstsq=True,
        g_tol=1e-6,
    ):
        self.loss_fn = loss_fn
        self.ineq_constraints = ineq_constraints
        self.eps = eps
        self.use_lstsq = use_lstsq
        self.g_tol = g_tol

    def _solve_least_squares(self, M, v, vector=True):
        eye = torch.eye(M.shape[-1], device=M.device)
        M = M + self.eps * eye
        if self.use_lstsq:
            proj_grads = torch.linalg.lstsq(M, v).solution
        else:
            iM = torch.linalg.inv(M)
            if vector:
                proj_grads = torch.einsum("...nn,...n->...n", iM, v)
            else:
                proj_grads = torch.einsum("...nn,...mn->...nm", iM, v)
        return proj_grads

    def _compute_constraints_and_grads(self, x1, fn):
        def compute_constraints(x1):
            g = fn(x1)
            return g, g.clone().detach()
        return torch.func.vmap(
            torch.func.jacrev(
                compute_constraints, has_aux=True
            )
        )(x1)

    def compute_lagrangian(self, x1, ineq_mul=None):
        """Compute the Lagrangian of `loss_fn` at `x1`, given (optional) multipliers for the inequality constraints.

        The Lagrangian is `loss_fn(x1) + sum_i <ineq_mul[i], ineq_constraints[i](x1)>`, where the inner
        product is taken per-sample over the constraint output dimension. When `ineq_mul` is `None`,
        no constraint term is added and the raw `loss_fn(x1)` is returned.

        :param x1: Batch of terminal states, of shape `(batch_size, flow_dim)`.
        :type x1: class:`Tensor`

        :param ineq_mul: Optional list of Lagrange multiplier tensors, one per entry in
            :attr:`ineq_constraints`, each of shape `(batch_size, constraint_dim)`, defaults to `None`.
        :type ineq_mul: class:`list[Tensor] | None`

        :return: The (per-sample) Lagrangian value.
        :rtype: class:`Tensor`
        """
        loss = self.loss_fn(x1)
        for idx, fn in enumerate(self.ineq_constraints):
            if ineq_mul is not None:
                loss = loss + torch.einsum("bn,bn->b", ineq_mul[idx], fn(x1))
        return loss

    def _get_lhs_block_diagonal_matrix_and_active_constraints(self, x1):
        """
        Computes the LHS matrix for the KKT system and tracks active constraints
        per sample and per dimension.
        """
        batch_size = x1.shape[0] if x1.ndim > 1 else 1

        # Compute inequality constraints and gradients
        ineq_constraints_and_grads = [
            self._compute_constraints_and_grads(x1, fn)
            for fn in self.ineq_constraints
        ]

        # Active constraints per sample and per dimension
        active_ineq_constraints = [
            torch.abs(e[1]) <= self.g_tol  # shape: [batch_size, dim]
            for e in ineq_constraints_and_grads
        ]

        # Flatten gradients only for active dimensions
        ineq_grads_active = []
        for e, active in zip(ineq_constraints_and_grads, active_ineq_constraints):
            grad = e[0]  # [batch, dim, x_dim]
            batch_active_grads = []
            for b in range(batch_size):
                batch_mask = active[b]  # [dim]
                selected = grad[b][batch_mask]  # [num_active, x_dim]
                if selected.numel() == 0:
                    selected = torch.zeros((0, grad.shape[-1]), device=grad.device)
                batch_active_grads.append(selected)
            # Stack along a new batch dimension
            max_active = max([g.shape[0] for g in batch_active_grads])
            # pad tensors to the same size (needed for batching)
            padded = torch.stack([
                torch.nn.functional.pad(g, (0, 0, 0, max_active - g.shape[0]))
                for g in batch_active_grads
            ])
            ineq_grads_active.append(padded)  # shape [batch, max_active, x_dim]

        # Dimensions of constraints (per function)
        ineq_constraints_dims = [e[1].shape[-1] for e in ineq_constraints_and_grads]

        # if len(ineq_grads_active) == 0:
        if all(active.sum() == 0 for active in active_ineq_constraints):
            return None, active_ineq_constraints, ineq_constraints_dims, ineq_grads_active

        # Build LHS block matrix for least squares
        # ineq_grads_active = [torch.concat(e, dim=0) for e in ineq_grads_active]
        row_blocks = []
        for rgrad in ineq_grads_active:
            rgrad = rgrad[..., 0, :]
            col_blocks = []
            for cgrad in ineq_grads_active:
                # block = torch.einsum("bnm,bpm->bp", rgrad, cgrad)
                cgrad = cgrad[..., 0, :]
                block = torch.einsum("bn,bp->bnp", rgrad, cgrad)
                col_blocks.append(block)
            col_blocks = torch.concatenate(col_blocks, dim=-1)
            row_blocks.append(col_blocks)
        lhs = torch.concatenate(row_blocks, dim=-2)

        return lhs, active_ineq_constraints, ineq_constraints_dims, ineq_grads_active

    def compute_multipliers(self, x1):
        """Estimate the Lagrange multipliers of the active inequality constraints at `x1`.

        A constraint (or constraint dimension) is considered active when its absolute value is within
        :attr:`g_tol` of zero. The multipliers for the active constraints are obtained by solving, in a
        least-squares sense, the stationarity condition of the KKT system built from the constraint
        gradients and the gradient of `loss_fn` at `x1`; inactive constraint dimensions are assigned a
        multiplier of zero. If no constraint is active, all multipliers are zero.

        :param x1: Batch of terminal states, of shape `(batch_size, flow_dim)`.
        :type x1: class:`Tensor`

        :return: A list of multiplier tensors, one per entry in :attr:`ineq_constraints`, each of shape
            `(batch_size, constraint_dim)`.
        :rtype: class:`list[Tensor]`
        """
        lhs, active_ineq_constraints, ineq_constraints_dims, ineq_grads_active = self._get_lhs_block_diagonal_matrix_and_active_constraints(x1)

        batch_size = x1.shape[0] if x1.ndim > 1 else 1
        total_dim = sum(ineq_constraints_dims)

        # return zeros if no active constraints
        if lhs is None:
            # create a tensor of zeros with shape [batch, dim] for each constraint
            full_lambdas_list = []
            start = 0
            for dim in ineq_constraints_dims:
                zeros = torch.zeros((batch_size, dim), device=x1.device)
                full_lambdas_list.append(zeros)
                start += dim  # not strictly needed here, just for consistency
            return full_lambdas_list

        # Compute gradient of loss
        loss = self.loss_fn(x1)
        grad_outputs = torch.ones_like(loss)
        rhs_full = -torch.autograd.grad(loss, x1, create_graph=True, grad_outputs=grad_outputs)[0]

        # rhs_active: flatten per-sample, per-active constraint
        rhs_active_list = []
        for b in range(batch_size):
            rhs_active_b = []
            for grad, active in zip(ineq_grads_active, active_ineq_constraints):
                # grad[b]: [max_active, x_dim] after padding
                # active[b]: [dim]
                n_active = active[b].sum().item()
                if n_active > 0:
                    # take the first n_active rows (because of padding)
                    rhs_active_rows = rhs_full[b] @ grad[b, :n_active, :].T  # shape [n_active]
                    rhs_active_b.extend(-rhs_active_rows)
            rhs_active_b = torch.tensor(rhs_active_b, device=x1.device)
            rhs_active_list.append(rhs_active_b)

        # pad to max_active constraints across batch
        max_active = max([len(r) for r in rhs_active_list])
        rhs_active_list = [
            torch.nn.functional.pad(r, (0, max_active - len(r))) for r in rhs_active_list
        ]
        rhs_active = torch.stack(rhs_active_list)  # [batch, max_active]

        # Solve for active multipliers
        active_lambdas = self._solve_least_squares(lhs, rhs_active, vector=True)
        if active_lambdas.ndim == 1 and batch_size > 1:
            active_lambdas = active_lambdas.unsqueeze(0)

        # Fill full multipliers, respecting per-sample, per-dimension activity
        full_lambdas = torch.zeros((batch_size, total_dim), device=x1.device)
        active_idx = torch.zeros(batch_size, dtype=torch.long, device=x1.device)

        start_full = 0
        for dim, active in zip(ineq_constraints_dims, active_ineq_constraints):
            for b in range(batch_size):
                for d in range(dim):
                    if active[b, d]:
                        full_lambdas[b, start_full + d] = active_lambdas[b, active_idx[b]]
                        active_idx[b] += 1
            start_full += dim


        full_lambdas_list = []
        start = 0
        for dim in ineq_constraints_dims:
            full_lambdas_list.append(full_lambdas[:, start:start+dim])
            start += dim

        # Now full_lambdas_list[i] has shape [batch, dim_of_constraint_i]
        # Return as a list instead of trying to reshape
        return full_lambdas_list

