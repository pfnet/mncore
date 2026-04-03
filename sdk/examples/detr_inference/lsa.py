import torch


def linear_sum_assignment(
    cost_matrix_input: torch.Tensor, original_nr: int, original_nc: int
) -> tuple[torch.Tensor, torch.Tensor]:
    device = cost_matrix_input.device
    dtype = torch.float64
    cost_matrix = cost_matrix_input.double()

    transpose = original_nc < original_nr
    if transpose:
        cost = cost_matrix.t()
        nr, nc = original_nc, original_nr
    else:
        cost = cost_matrix
        nr, nc = original_nr, original_nc

    u = torch.zeros(nr, dtype=dtype, device=device)
    v = torch.zeros(nc, dtype=dtype, device=device)
    col4row = torch.full((nr,), -1, dtype=torch.int64, device=device)
    row4col = torch.full((nc,), -1, dtype=torch.int64, device=device)

    for cur_row in range(nr):
        sink, shortest_path_costs, path, SR, SC, min_val = _augmenting_path(
            nr, nc, cost, u, v, row4col, cur_row
        )

        u[cur_row] += min_val
        indices = torch.arange(nr, device=u.device)
        masked_indices = SR & (indices != cur_row)
        u = torch.where(masked_indices, u + (min_val - shortest_path_costs[col4row]), u)

        v = torch.where(SC, v - (min_val - shortest_path_costs), v)

        j = sink
        is_loop_active = torch.tensor(True)
        for _ in range(nr):
            maybe_neg_i = torch.gather(path, 0, j.unsqueeze(0)).squeeze()
            i = torch.where(maybe_neg_i >= 0, maybe_neg_i, torch.tensor(0))

            row4col_updated = row4col.scatter(0, j, i)
            row4col = torch.where(is_loop_active, row4col_updated, row4col)

            tmp = torch.gather(col4row, 0, i.unsqueeze(0)).squeeze()
            col4row_updated = col4row.scatter(0, i, j)
            col4row = torch.where(is_loop_active, col4row_updated, col4row)
            maybe_neg_j = tmp
            j = torch.where(maybe_neg_j >= 0, maybe_neg_j, torch.tensor(0))

            is_loop_active = is_loop_active & (i != cur_row)

    if transpose:
        sorted_indices = torch.argsort(col4row)
        row_ind = col4row[sorted_indices]
        col_ind = sorted_indices
    else:
        row_ind = torch.arange(nr, device=device)
        col_ind = col4row

    return row_ind, col_ind


def _augmenting_path(  # noqa: CFQ002
    nr: int,
    nc: int,
    cost: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    row4col: torch.Tensor,
    i: int,
):
    device = cost.device
    dtype = cost.dtype

    i_t = torch.tensor(i, dtype=torch.int64, device=device)
    min_val = torch.tensor(0.0, dtype=dtype, device=device)
    sink = torch.tensor(-1, dtype=torch.int64, device=device)
    shortest_path_costs = torch.full((nc,), float("inf"), dtype=dtype, device=device)
    path = torch.full((nc,), -1, dtype=torch.int64, device=device)
    SR = torch.zeros(nr, dtype=torch.bool, device=device)
    SC = torch.zeros(nc, dtype=torch.bool, device=device)

    for _ in range(nc):
        is_loop_active = sink == -1

        SR_updated = SR.scatter(0, i_t, torch.ones_like(SR[0]))
        SR = torch.where(is_loop_active, SR_updated, SR)

        selected_cost_row = torch.index_select(cost, 0, i_t).squeeze()
        selected_u_val = torch.index_select(u, 0, i_t).squeeze()
        r = min_val + selected_cost_row - selected_u_val - v

        update_mask = (r < shortest_path_costs) & (~SC)
        path = torch.where(update_mask & is_loop_active, i_t, path)
        shortest_path_costs = torch.where(
            update_mask & is_loop_active, r, shortest_path_costs
        )

        costs_to_consider = torch.where(~SC, shortest_path_costs, float("inf"))
        lowest = torch.min(costs_to_consider)

        min_val = torch.where(is_loop_active, lowest, min_val)

        j = torch.argmin(costs_to_consider)
        is_j_sink = torch.gather(row4col, 0, j.unsqueeze(0)).squeeze() == -1
        sink = torch.where(is_j_sink & is_loop_active, j, sink)
        next_row = torch.gather(row4col, 0, j.unsqueeze(0)).squeeze()
        i_t = torch.where((~is_j_sink) & is_loop_active, next_row, i_t)

        SC_updated = SC.scatter(0, j, torch.ones_like(SC[0]))
        SC = torch.where(is_loop_active, SC_updated, SC)

    return sink, shortest_path_costs, path, SR, SC, min_val
