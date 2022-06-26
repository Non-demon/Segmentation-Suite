import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.nn import ConvTranspose2d
from torch.nn import functional as F
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1,
                 padding = 0, dilation = 1, groups = 1, bias = False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    # 修改这里的实现函数
    def forward(self, input):
        return conv2d_same(input, self.weight, self.bias, self.stride,
                           self.dilation, self.groups)


class ConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1,
                 padding = 0, output_padding = 0, groups = 1, bias = False, dilation = 1):
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)

    def forward(self, input):
        input_size = input.size(2)
        output_size = input_size * self.stride[0]
        pad_l, pad_r = get_same(input_size, self.kernel_size[0], self.stride[0], dilation = 1)
        # print(pad_l,pad_r)
        self.padding = max(pad_l, pad_r)
        input_size = (input_size - 1) * self.stride[0] + self.kernel_size[0] - 2 * self.padding
        # print(input_size)
        output_padding = output_size - input_size
        # print(output_padding)
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


def conv2d_same(input, weight, bias = None, stride = [1, 1], dilation = (1, 1), groups = 1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)
    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.conv2d(input, weight, bias, stride,
                    padding = (padding_rows // 2, padding_cols // 2),
                    dilation = dilation, groups = groups)


def max_pool2d_same(input, kernel_size, stride = 1, dilation = 1, ceil_mode = False, return_indices = False):
    input_rows = input.size(2)
    out_rows = (input_rows + stride - 1) // stride
    padding_rows = max(0, (out_rows - 1) * stride +
                       (kernel_size - 1) * dilation + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    cols_odd = (padding_rows % 2 != 0)
    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.max_pool2d(input, kernel_size = kernel_size, stride = stride, padding = padding_rows // 2,
                        dilation = dilation,
                        ceil_mode = ceil_mode, return_indices = return_indices)


def get_same(size, kernel, stride, dilation):
    out_size = (size + stride - 1) // stride
    padding = max(0, (out_size - 1) * stride +
                  (kernel - 1) * dilation + 1 - size)
    size_odd = (padding % 2 != 0)
    pad_l = padding // 2
    pad_r = padding // 2
    if size_odd:
        pad_l += 1
    return pad_l, pad_r


class CapsuleLayer(nn.Module):
    def __init__(self, t_0, z_0, op, k, s, t_1, z_1, routing):
        super().__init__()
        self.t_1 = t_1
        self.z_1 = z_1
        self.op = op
        self.k = k
        self.s = s
        self.routing = routing
        self.convs = nn.ModuleList()
        self.t_0 = t_0
        for _ in range(t_0):
            if self.op == 'conv':
                self.convs.append(nn.Conv2d(z_0, self.t_1 * self.z_1, self.k, self.s, padding = 2, bias = False))
            else:
                self.convs.append(
                    nn.ConvTranspose2d(z_0, self.t_1 * self.z_1, self.k, self.s, padding = 2, output_padding = 1))

    def forward(self, u):  # input [N,CAPS,C,H,W]
        # t0 means the input cap num and t1 means that of the output
        # z0 means the input cap dim and t1 means that of the output
        if u.shape[1] != self.t_0:
            raise ValueError("Wrong type of operation for capsule")
        op = self.op
        k = self.k
        s = self.s
        t_1 = self.t_1
        z_1 = self.z_1
        routing = self.routing
        N = u.shape[0]
        H_1 = u.shape[3]
        W_1 = u.shape[4]
        t_0 = self.t_0

        u_t_list = [u_t.squeeze(1) for u_t in u.split(1, 1)]  # 将cap分别取出来
        # the squeeze operation reshape u_t into [N,C,H,W] if the CAPS is equal to 1
        # but nothing changes if not
        # item in u.split(1, 1) is like (N,1,C,H,W) in which C = z0

        u_hat_t_list = []

        for i, u_t in zip(range(self.t_0), u_t_list):  # u_t: [N,C,H,W]
            if op == "conv":
                u_hat_t = self.convs[i](u_t)  # 卷积方式
            elif op == "deconv":
                u_hat_t = self.convs[i](u_t)  # u_hat_t: [N,t_1*z_1,H,W]
            else:
                raise ValueError("Wrong type of operation for capsule")
            H_1 = u_hat_t.shape[2]
            W_1 = u_hat_t.shape[3]
            u_hat_t = u_hat_t.reshape(N, t_1, z_1, H_1, W_1).transpose_(1, 3).transpose_(2, 4)
            u_hat_t_list.append(u_hat_t)  # [N,H_1,W_1,t_1,z_1]
        v = self.update_routing(u_hat_t_list, k, N, H_1, W_1, t_0, t_1, routing)
        # 对于输出的每个像素的每个cap 都被t0个对应像素（因为stride的关系，可能对应多个像素）的输入cap
        # 采用卷积的方式（共享参数）映射 也就是u_hat_t_list 有t0个(n,h1,w1,t1,z1)
        # 最后用路由投票的方式决定每个cap 输出的shape为(n,t1,z1,h1,w1)
        return v

    def update_routing(self, u_hat_t_list, k, N, H_1, W_1, t_0, t_1, routing):
        one_kernel = torch.ones(1, t_1, k, k).cuda()  # 不需要学习
        b = torch.zeros(N, H_1, W_1, t_0, t_1).cuda()  # 不需要学习
        b_t_list = [b_t.squeeze(3) for b_t in b.split(1, 3)]
        u_hat_t_list_sg = []
        for u_hat_t in u_hat_t_list:
            u_hat_t_sg = u_hat_t.detach()
            u_hat_t_list_sg.append(u_hat_t_sg)

        for d in range(routing):
            if d < routing - 1:
                u_hat_t_list_ = u_hat_t_list_sg
            else:
                u_hat_t_list_ = u_hat_t_list

            r_t_mul_u_hat_t_list = []
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                # routing softmax (N,H_1,W_1,t_1)
                b_t.transpose_(1, 3).transpose_(2, 3)  # [N,t_1,H_1, W_1]
                b_t_max = torch.nn.functional.max_pool2d(b_t, k, 1, padding = 2)
                b_t_max = b_t_max.max(1, True)[0]
                c_t = torch.exp(b_t - b_t_max)
                sum_c_t = conv2d_same(c_t, one_kernel, stride = (1, 1))  # [... , 1]
                r_t = c_t / sum_c_t  # [N,t_1, H_1, W_1]
                r_t = r_t.transpose(1, 3).transpose(1, 2)  # [N, H_1, W_1,t_1]
                r_t = r_t.unsqueeze(4)  # [N, H_1, W_1,t_1, 1]
                r_t_mul_u_hat_t_list.append(r_t * u_hat_t)  # [N, H_1, W_1, t_1, z_1]
            p = sum(r_t_mul_u_hat_t_list)  # [N, H_1, W_1, t_1, z_1]
            v = squash(p)
            if d < routing - 1:
                b_t_list_ = []
                for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                    # b_t     : [N, t_1,H_1, W_1]
                    # u_hat_t : [N, H_1, W_1, t_1, z_1]
                    # v       : [N, H_1, W_1, t_1, z_1]
                    # [N,H_1,W_1,t_1]
                    b_t.transpose_(1, 3).transpose_(2, 1)
                    b_t_list_.append(b_t + (u_hat_t * v).sum(4))
        v.transpose_(1, 3).transpose_(2, 4)
        # print(v.grad)
        return v

    def squash(self, p):
        p_norm_sq = (p * p).sum(-1, True)
        p_norm = (p_norm_sq + 1e-9).sqrt()
        v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
        return v


def update_routing(u_hat_t_list, k, N, H_1, W_1, t_0, t_1, routing):
    one_kernel = torch.ones(1, t_1, k, k).cuda()  # 不需要学习
    b = torch.zeros(N, H_1, W_1, t_0, t_1).cuda()  # 不需要学习
    b_t_list = [b_t.squeeze(3) for b_t in b.split(1, 3)]
    u_hat_t_list_sg = []
    for u_hat_t in u_hat_t_list:
        u_hat_t_sg = u_hat_t.clone()
        u_hat_t_sg.detach_()
        u_hat_t_list_sg.append(u_hat_t_sg)

    for d in range(routing):
        if d < routing - 1:
            u_hat_t_list_ = u_hat_t_list_sg
        else:
            u_hat_t_list_ = u_hat_t_list

        r_t_mul_u_hat_t_list = []
        for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
            # routing softmax (N,H_1,W_1,t_1)
            b_t.transpose_(1, 3).transpose_(2, 3)
            torch.nn.functional.max_pool2d(b_t, k, )
            b_t_max = max_pool2d_same(b_t, k, 1)
            b_t_max = b_t_max.max(1, True)[0]
            c_t = torch.exp(b_t - b_t_max)
            sum_c_t = conv2d_same(c_t, one_kernel, stride = (1, 1))  # [... , 1]
            r_t = c_t / sum_c_t  # [N,t_1, H_1, W_1]
            r_t = r_t.transpose(1, 3).transpose(1, 2)  # [N, H_1, W_1,t_1]
            r_t = r_t.unsqueeze(4)  # [N, H_1, W_1,t_1, 1]
            r_t_mul_u_hat_t_list.append(r_t * u_hat_t)  # [N, H_1, W_1, t_1, z_1]

        p = sum(r_t_mul_u_hat_t_list)  # [N, H_1, W_1, t_1, z_1]
        v = squash(p)
        if d < routing - 1:
            b_t_list_ = []
            for b_t, u_hat_t in zip(b_t_list, u_hat_t_list_):
                # b_t     : [N, t_1,H_1, W_1]
                # u_hat_t : [N, H_1, W_1, t_1, z_1]
                # v       : [N, H_1, W_1, t_1, z_1]
                b_t = b_t.transpose(1, 3).transpose(1, 2)  # [N,H_1,W_1,t_1]
                b_t_list_.append(b_t + (u_hat_t * v).sum(4))
            b_t_list = b_t_list_
        v.transpose_(1, 3).transpose_(2, 4)
    # print(v.grad)
    return v


def squash(p):
    p_norm_sq = (p * p).sum(-1, True)
    p_norm = (p_norm_sq + 1e-9).sqrt()
    v = p_norm_sq / (1. + p_norm_sq) * p / p_norm
    return v


class SegCaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, padding = 2, bias = False),

        )
        self.step_1 = nn.Sequential(  # 1/2
            CapsuleLayer(1, 16, "conv", k = 5, s = 2, t_1 = 2, z_1 = 16, routing = 1),
            CapsuleLayer(2, 16, "conv", k = 5, s = 1, t_1 = 4, z_1 = 16, routing = 3),
        )
        self.step_2 = nn.Sequential(  # 1/4
            CapsuleLayer(4, 16, "conv", k = 5, s = 2, t_1 = 4, z_1 = 32, routing = 3),
            CapsuleLayer(4, 32, "conv", k = 5, s = 1, t_1 = 8, z_1 = 32, routing = 3)
        )
        self.step_3 = nn.Sequential(  # 1/8
            CapsuleLayer(8, 32, "conv", k = 5, s = 2, t_1 = 8, z_1 = 64, routing = 3),
            CapsuleLayer(8, 64, "conv", k = 5, s = 1, t_1 = 8, z_1 = 32, routing = 3)
        )
        self.step_4 = CapsuleLayer(8, 32, "deconv", k = 5, s = 2, t_1 = 8, z_1 = 32, routing = 3)

        self.step_5 = CapsuleLayer(16, 32, "conv", k = 5, s = 1, t_1 = 4, z_1 = 32, routing = 3)

        self.step_6 = CapsuleLayer(4, 32, "deconv", k = 5, s = 2, t_1 = 4, z_1 = 16, routing = 3)
        self.step_7 = CapsuleLayer(8, 16, "conv", k = 5, s = 1, t_1 = 4, z_1 = 16, routing = 3)
        self.step_8 = CapsuleLayer(4, 16, "deconv", k = 5, s = 2, t_1 = 2, z_1 = 16, routing = 3)
        self.step_10 = CapsuleLayer(3, 16, "conv", k = 5, s = 1, t_1 = 1, z_1 = 16, routing = 3)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 1, 5, 1, padding = 2),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x.unsqueeze_(1)

        skip_1 = x  # [N,1,16,H,W]

        x = self.step_1(x)

        skip_2 = x  # [N,4,16,H/2,W/2]
        x = self.step_2(x)

        skip_3 = x  # [N,8,32,H/4,W/4]

        x = self.step_3(x)  # [N,8,32,H/8,W/8]

        x = self.step_4(x)  # [N,8,32,H/4,W/4]
        x = torch.cat((x, skip_3), 1)  # [N,16,32,H/4,W/4]

        x = self.step_5(x)  # [N,4,32,H/4,W/4]

        x = self.step_6(x)  # [N,4,16,H/2,W/2]

        x = torch.cat((x, skip_2), 1)  # [N,8,16,H/2,W/2]
        x = self.step_7(x)  # [N,4,16,H/2,W/2]
        x = self.step_8(x)  # [N,2,16,H,W]

        x = torch.cat((x, skip_1), 1)  # [N,3,16,H,W]
        x = self.step_10(x)  # [N,1,16,H,W]
        x.squeeze_(1)  # [N,16,H,W]
        v_lens = self.compute_vector_length(x)
        v_lens = v_lens.squeeze(1)  # [N,H,W]
        return v_lens

    def compute_vector_length(self, x):
        out = (x.pow(2)).sum(1, True) + 1e-9
        out = out.sqrt()
        return out


def compute_vector_length(x):
    out = (x.pow(2)).sum(1, True) + 1e-9
    out.sqrt_()
    return out


if __name__ == '__main__':
    import os

    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    model = SegCaps()
    model = model.cuda()
    a = torch.ones(1, 3, 256, 256)
    a = a.cuda()
    b = model(a)
    c = b.sum()
    c.backward()
    # for k, v in model.named_parameters():
    #     a = input('s')
    #     print(v.grad, k)
    # from tensorboardX import SummaryWriter
    # with SummaryWriter(comment='LeNet') as w:
    #     w.add_graph(model, a)
    print(b.shape)
    print(b)
