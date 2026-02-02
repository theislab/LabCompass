import torch


__all__ = ["KKTConditions"]


class KKTConditions:
    def __init__(
        self,
        loss_fn,
        eq_constraints,
        ineq_constraints,
        eps=1e-6,
        use_lstsq=True,
        g_tol=1e-6,
    ):
        self.loss_fn = loss_fn
        self.eq_constraints = eq_constraints
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

    def compute_lagrangian(self, x1, eq_mul, ineq_mul):
        loss = self.loss_fn(x1)
        for idx, fn in enumerate(self.eq_constraints):
            loss = loss + eq_mul[idx]*fn(x1)
        for idx, fn in enumerate(self.ineq_constraints):
            loss = loss + ineq_mul[idx]*fn(x1)
        return loss

    def _get_lhs_block_diagonal_matrix_and_active_constraints(
        self,
        x1,
    ):
        # compute inequality constraints and gradients
        ineq_constraints_and_grads = [
            self._compute_constraints_and_grads(x1, fn)
            for fn in self.ineq_constraints
        ]
        active_ineq_constraints = [
            torch.any(torch.abs(e[1]) <= self.g_tol)
            for e in ineq_constraints_and_grads
        ]
        ineq_grads = [
            e[0] for idx, e in enumerate(ineq_constraints_and_grads)
            if active_ineq_constraints[idx]
        ]
        if len(ineq_grads) == 0:
            return None, active_ineq_constraints

        # compute equality constraints and gradients
        # eq_grads = [
        #     self._compute_constraints_and_grads(x1, fn)[0]
        #     for fn in self.eq_constraints
        # ]

        # compute all blocks for lhs
        # all_grads = eq_grads + ineq_grads
        all_grads = ineq_grads
        row_blocks = []
        for rgrad in all_grads:
            col_blocks = []
            for cgrad in all_grads:
                block = torch.einsum("...mn,...qn->...mq", rgrad, cgrad)
                col_blocks.append(block)
            col_blocks = torch.concatenate(col_blocks, dim=-1)
            row_blocks.append(col_blocks)
        return torch.concatenate(row_blocks, dim=-2), active_ineq_constraints

    def compute_stationarity_conditions(
        self,
        x1,
        eq_mul,
        ineq_mul,
        dx1_dt,
    ):
        # sanity checks
        assert len(eq_mul) == len(self.eq_constraints)
        assert len(ineq_mul) == len(self.ineq_constraints)

        # compute lhs
        lhs, active_ineq_constraints = self._get_lhs_block_diagonal_matrix_and_active_constraints(x1)

        # compute right hand side
        grad_L = torch.autograd.grad(
            self.compute_lagrangian(x1, eq_mul, ineq_mul), x1, create_graph=True
        )[0]
        rhs = -torch.autograd.grad(
            grad_L, x1, grad_outputs=dx1_dt, retain_graph=True
        )[0]
        return self._solve_least_squares(lhs, rhs, vector=True)

    def compute_multipliers(self, x1):
        # compute inequality constraints and gradients
        lhs, active_ineq_constraints = self._get_lhs_block_diagonal_matrix_and_active_constraints(x1)

        if lhs is None:
            return 0

        # compute rhs
        loss = self.loss_fn(x1)
        grad_outputs = torch.ones_like(loss)
        rhs = - torch.autograd.grad(
            loss, x1, create_graph=True,
            grad_outputs=grad_outputs
        )[0]
        print(rhs)
        active_lambdas = self._solve_least_squares(lhs, rhs, vector=True)
        return active_lambdas
