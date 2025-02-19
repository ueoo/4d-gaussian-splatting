import torch
from utils.general_utils import knn, safe_state

k = 20

xyz = torch.randn(1000, 3).cuda()
velocity = torch.randn(1000, 3).cuda()

idx, dist = knn(xyz[None].contiguous().detach(), xyz[None].contiguous().detach(), k)
print(idx.shape, dist.shape)
weight = torch.exp(-100 * dist)
print(weight.shape)
# cur_marginal_t = gaussians.get_marginal_t(cur_time).detach().squeeze(-1)
# marginal_weights = cur_marginal_t[idx] * cur_marginal_t[None,:,None]
# weight *= marginal_weights

# mean_t, cov_t = gaussians.get_t, gaussians.get_cov_t(scaling_modifier=1)
# mean_t_nn, cov_t_nn = mean_t[idx], cov_t[idx]
# weight *= torch.exp(-0.5*(mean_t[None, :, None]-mean_t_nn)**2/cov_t[None, :, None]/cov_t_nn*(cov_t[None, :, None]+cov_t_nn)).squeeze(-1).detach()
vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
Lrigid = (weight * vel_dist).sum() / k / xyz.shape[0]
print(Lrigid)
