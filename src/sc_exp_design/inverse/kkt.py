import torch


__all__ = ["KKTConditions"]


class KKTConditions:
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

    def compute_lagrangian(self, x1, ineq_mul):
        loss = self.loss_fn(x1)
        for idx, fn in enumerate(self.ineq_constraints):
            loss = loss + ineq_mul[idx]*fn(x1)
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
            grad = e[0]  # shape [batch_size, dim, x_dim]
            batch_indices, dim_indices = torch.nonzero(active, as_tuple=True)
            if len(dim_indices) > 0:
                # select only active gradients
                selected = grad[batch_indices, dim_indices, :]
                # make sure we keep shape [batch, active_dim, x_dim]
                selected = selected.view(batch_size, -1, grad.shape[-1])
                ineq_grads_active.append(selected)

        # Dimensions of constraints (per function)
        ineq_constraints_dims = [e[1].shape[-1] for e in ineq_constraints_and_grads]

        if len(ineq_grads_active) == 0:
            return None, active_ineq_constraints, ineq_constraints_dims

        # Build LHS block matrix for least squares
        row_blocks = []
        for rgrad in ineq_grads_active:
            col_blocks = []
            for cgrad in ineq_grads_active:
                block = torch.einsum("bnm,bpm->bnp", rgrad, cgrad)
                col_blocks.append(block)
            col_blocks = torch.concatenate(col_blocks, dim=-1)
            row_blocks.append(col_blocks)
        lhs = torch.concatenate(row_blocks, dim=-2)

        return lhs, active_ineq_constraints, ineq_constraints_dims
    def compute_multipliers(self, x1):
        lhs, active_ineq_constraints, ineq_constraints_dims = self._get_lhs_block_diagonal_matrix_and_active_constraints(x1)

        batch_size = x1.shape[0] if x1.ndim > 1 else 1
        total_dim = sum(ineq_constraints_dims)

        # return zeros if no active constraints
        if lhs is None:
            return torch.zeros((batch_size, total_dim), device=x1.device)

        # Compute gradient of loss
        loss = self.loss_fn(x1)
        grad_outputs = torch.ones_like(loss)
        rhs_full = -torch.autograd.grad(loss, x1, create_graph=True, grad_outputs=grad_outputs)[0]

        # --- FIX START: slice RHS to only active constraints ---
        rhs_active_list = []
        for b in range(batch_size):
            rhs_active_b = []
            for dim, active in zip(ineq_constraints_dims, active_ineq_constraints):
                rhs_active_b.extend(rhs_full[b, :dim][active[b]])
            rhs_active_list.append(torch.tensor(rhs_active_b, device=x1.device))
        rhs_active = torch.stack(rhs_active_list)
        # --- FIX END ---

        # Solve for active multipliers
        active_lambdas = self._solve_least_squares(lhs, rhs_active, vector=True)
        if active_lambdas.ndim == 1 and batch_size > 1:
            active_lambdas = active_lambdas.unsqueeze(0)

        # Fill full multipliers, respecting per-sample, per-dimension activity
        full_lambdas = torch.zeros((batch_size, total_dim), device=x1.device)
        start_full = 0
        start_active = 0
        for dim, active in zip(ineq_constraints_dims, active_ineq_constraints):
            for b in range(batch_size):
                for d in range(dim):
                    if active[b, d]:
                        full_lambdas[b, start_full + d] = active_lambdas[b, start_active]
                        start_active += 1

        # return torch.nn.functional.relu(full_lambdas)
        return full_lambdas
