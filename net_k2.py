import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import module as md

class Generator1(nn.Module):
    # 1D convolutional architecture
    def __init__(self, in_ch, n_spk, z_ch, mid_ch, s_ch, normtype='IN', src_conditioning=False):
        super(Generator1, self).__init__()
        add_ch = 0 if src_conditioning==False else s_ch
        self.le1 = md.ConvGLU1D(in_ch+add_ch, mid_ch, 9, 1, normtype)
        self.le2 = md.ConvGLU1D(mid_ch+add_ch, mid_ch, 8, 2, normtype)
        self.le3 = md.ConvGLU1D(mid_ch+add_ch, mid_ch, 8, 2, normtype)
        self.le4 = md.ConvGLU1D(mid_ch+add_ch, mid_ch, 5, 1, normtype)
        self.le5 = md.ConvGLU1D(mid_ch+add_ch, z_ch, 5, 1, normtype)
        self.le6 = md.DeconvGLU1D(z_ch+s_ch, mid_ch, 5, 1, normtype)
        self.le7 = md.DeconvGLU1D(mid_ch+s_ch, mid_ch, 5, 1, normtype)
        self.le8 = md.DeconvGLU1D(mid_ch+s_ch, mid_ch, 8, 2, normtype)
        self.le9 = md.DeconvGLU1D(mid_ch+s_ch, mid_ch, 8, 2, normtype)
        self.le10 = nn.Conv1d(mid_ch+s_ch, in_ch, 9, stride=1, padding=(9-1)//2)
        #nn.init.xavier_normal_(self.le10.weight,gain=0.1)

        if src_conditioning:
            self.eb0 = nn.Embedding(n_spk, s_ch)
        self.eb1 = nn.Embedding(n_spk, s_ch)
        self.src_conditioning = src_conditioning

    def __call__(self, xin, k_t, k_s=None):
        device = xin.device
        B, n_mels, n_frame_ = xin.shape

        kk_t = k_t*torch.ones(B).to(device, dtype=torch.int64)
        trgspk_emb = self.eb1(kk_t)
        if self.src_conditioning:
            kk_s = k_s*torch.ones(B).to(device, dtype=torch.int64)
            srcspk_emb = self.eb0(kk_s)

        out = xin

        if self.src_conditioning: out = md.concat_dim1(out,srcspk_emb)
        out = self.le1(out)
        if self.src_conditioning: 
            out = md.concat_dim1(out,srcspk_emb)
        out = self.le2(out)
        if self.src_conditioning: 
            out = md.concat_dim1(out,srcspk_emb)
        out = self.le3(out)
        if self.src_conditioning: 
            out = md.concat_dim1(out,srcspk_emb)
        out = self.le4(out)
        if self.src_conditioning: 
            out = md.concat_dim1(out,srcspk_emb)
        out = self.le5(out)
        out = md.concat_dim1(out,trgspk_emb)
        out = self.le6(out)
        out = md.concat_dim1(out,trgspk_emb)
        out = self.le7(out)
        out = md.concat_dim1(out,trgspk_emb)
        out = self.le8(out)
        out = md.concat_dim1(out,trgspk_emb)
        out = self.le9(out)
        out = md.concat_dim1(out,trgspk_emb)
        out = self.le10(out)

        return out

class Generator2(nn.Module):
    # Bidirectional LSTM 
    def __init__(self, in_ch, n_spk, z_ch, mid_ch, s_ch, num_layers=2, negative_slope=0.1, src_conditioning=False):
        super(Generator2, self).__init__()
        add_ch = 0 if src_conditioning==False else s_ch

        self.linear0 = md.LinearWN(in_ch+add_ch, mid_ch)
        self.lrelu0 = nn.LeakyReLU(negative_slope)
        self.rnn0 = nn.LSTM(
            mid_ch+add_ch,
            mid_ch//2,
            num_layers,
            dropout=0,
            bidirectional=True,
            batch_first = True
        )
        self.linear1 = md.LinearWN(mid_ch+add_ch, z_ch)
        self.linear2 = md.LinearWN(z_ch+s_ch, mid_ch)
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.rnn1 = nn.LSTM(
            mid_ch+s_ch,
            mid_ch//2,
            num_layers,
            dropout=0,
            bidirectional=True,
            batch_first = True
        )
        self.linear3 = md.LinearWN(mid_ch+s_ch, in_ch)

        if src_conditioning:
            self.eb0 = nn.Embedding(n_spk, s_ch)
        self.eb1 = nn.Embedding(n_spk, s_ch)
        self.src_conditioning = src_conditioning

    def __call__(self, xin, k_t, k_s=None):
        device = xin.device
        B, num_mels, num_frame = xin.shape
        kk_t = k_t*torch.ones(B).to(device, dtype=torch.int64)
        trgspk_emb = self.eb1(kk_t)
        if self.src_conditioning:
            kk_s = k_s*torch.ones(B).to(device, dtype=torch.int64)
            srcspk_emb = self.eb0(kk_s)
        out = xin

        out = out.permute(0,2,1) # (B, num_frame, num_mels)
        if self.src_conditioning: out = md.concat_dim2(out,srcspk_emb) # (B, num_frame, num_mels+add_ch)
        out = self.lrelu0(self.linear0(out))
        if self.src_conditioning: out = md.concat_dim2(out,srcspk_emb) # (B, num_frame, mid_ch+add_ch)
        self.rnn0.flatten_parameters()
        out, _ = self.rnn0(out) # (B, num_frame, mid_ch)
        if self.src_conditioning: out = md.concat_dim2(out,srcspk_emb) # (B, num_frame, mid_ch+add_ch)
        out = self.linear1(out) # (B, num_frame, z_ch)
        out = md.concat_dim2(out,trgspk_emb) # (B, num_frame, z_ch+s_ch)
        out = self.lrelu1(self.linear2(out)) # (B, num_frame, mid_ch)
        out = md.concat_dim2(out,trgspk_emb) # (B, num_frame, mid_ch+s_ch)
        self.rnn1.flatten_parameters()
        out, _ = self.rnn1(out) # (B, num_frame, mid_ch)
        out = md.concat_dim2(out,trgspk_emb) # (B, num_frame, mid_ch+s_ch)
        out = self.linear3(out) # (B, num_frame, in_ch)
        out = out.permute(0,2,1) # (B, in_ch, num_frame)

        return out

class Discriminator1(nn.Module):
    # 1D convolutional architecture
    def __init__(self, in_ch, clsnum, mid_ch, normtype='IN', dor=0.1):
        super(Discriminator1, self).__init__()
        self.le1 = md.ConvGLU1D(in_ch, mid_ch, 11, 1, normtype)
        self.le2 = md.ConvGLU1D(mid_ch, mid_ch, 10, 2, normtype)
        self.le3 = md.ConvGLU1D(mid_ch, mid_ch, 10, 2, normtype)
        self.le4 = md.ConvGLU1D(mid_ch, mid_ch, 7, 1, normtype)
        self.le_adv = nn.Conv1d(mid_ch, 1, 7, stride=1, padding=(7-1)//2, bias=False)
        self.le_cls = nn.Conv1d(mid_ch, clsnum, 7, stride=1, padding=(7-1)//2, bias=False)
        nn.init.xavier_normal_(self.le_adv.weight,gain=0.1)
        nn.init.xavier_normal_(self.le_cls.weight,gain=0.1)
        self.do1 = nn.Dropout(p=dor)
        self.do2 = nn.Dropout(p=dor)
        self.do3 = nn.Dropout(p=dor)
        self.do4 = nn.Dropout(p=dor)

    def __call__(self, xin):
        device = xin.device
        B, n_mels, n_frame_ = xin.shape

        out = xin

        out = self.do1(self.le1(out))
        out = self.do2(self.le2(out))
        out = self.do3(self.le3(out))
        out = self.do4(self.le4(out))

        out_adv = self.le_adv(out)
        out_cls = self.le_cls(out)
        
        return out_adv, out_cls
    
class StarGAN(nn.Module):
    def __init__(self, gen, dis, n_spk, loss_type='wgan'):
        super(StarGAN, self).__init__()
        self.gen = gen
        self.dis = dis
        self.n_spk = n_spk
        self.loss_type = loss_type

    def forward(self, x, k_t, k_s=None):
        device = x.device
        n_frame_ = x.shape[2]
        n_frame = math.ceil(n_frame_/4)*4
        if n_frame > n_frame_:
            x = nn.ReplicationPad1d((0, n_frame-n_frame_))(x)
        return self.gen(x, k_t, k_s)[:,:,0:n_frame_]

    def calc_advloss_g(self, df_adv_ss, df_adv_st, df_adv_tt, df_adv_ts):
        df_adv_ss = df_adv_ss.permute(0,2,1).reshape(-1,1)
        df_adv_st = df_adv_st.permute(0,2,1).reshape(-1,1)
        df_adv_tt = df_adv_tt.permute(0,2,1).reshape(-1,1)
        df_adv_ts = df_adv_ts.permute(0,2,1).reshape(-1,1)

        if self.loss_type=='wgan':
            # Wasserstein GAN with gradient penalty (WGAN-GP)
            AdvLoss_g = (
                torch.sum(-df_adv_ss) +
                torch.sum(-df_adv_st) + 
                torch.sum(-df_adv_tt) + 
                torch.sum(-df_adv_ts)
            ) / (df_adv_ss.numel() + df_adv_st.numel() + df_adv_tt.numel() + df_adv_ts.numel())

        elif self.loss_type=='lsgan':
            # Least squares GAN (LSGAN)
            AdvLoss_g = 0.5 * (
                torch.sum((df_adv_ss - torch.ones_like(df_adv_ss))**2) +
                torch.sum((df_adv_st - torch.ones_like(df_adv_st))**2) +
                torch.sum((df_adv_tt - torch.ones_like(df_adv_tt))**2) +
                torch.sum((df_adv_ts - torch.ones_like(df_adv_ts))**2)
            ) / (df_adv_ss.numel() + df_adv_st.numel() + df_adv_tt.numel() + df_adv_ts.numel())

        elif self.loss_type=='cgan':
            # Regular GAN with the sigmoid cross-entropy criterion (CGAN)
            AdvLoss_g = (
                F.binary_cross_entropy_with_logits(df_adv_ss, torch.ones_like(df_adv_ss), reduction='sum') +
                F.binary_cross_entropy_with_logits(df_adv_st, torch.ones_like(df_adv_st), reduction='sum') +
                F.binary_cross_entropy_with_logits(df_adv_tt, torch.ones_like(df_adv_tt), reduction='sum') +
                F.binary_cross_entropy_with_logits(df_adv_ts, torch.ones_like(df_adv_ts), reduction='sum')
            ) / (df_adv_ss.numel() + df_adv_st.numel() + df_adv_tt.numel() + df_adv_ts.numel())

        return AdvLoss_g

    def calc_clsloss_g(self, df_cls_ss, df_cls_st, df_cls_tt, df_cls_ts, k_s, k_t):
        device = df_cls_ss.device

        df_cls_ss = df_cls_ss.permute(0,2,1).reshape(-1,self.n_spk)
        df_cls_st = df_cls_st.permute(0,2,1).reshape(-1,self.n_spk)
        df_cls_tt = df_cls_tt.permute(0,2,1).reshape(-1,self.n_spk)
        df_cls_ts = df_cls_ts.permute(0,2,1).reshape(-1,self.n_spk)

        cf_ss = k_s*torch.ones(len(df_cls_ss), device=device, dtype=torch.long)
        cf_st = k_t*torch.ones(len(df_cls_st), device=device, dtype=torch.long)
        cf_tt = k_t*torch.ones(len(df_cls_tt), device=device, dtype=torch.long)
        cf_ts = k_s*torch.ones(len(df_cls_ts), device=device, dtype=torch.long)

        ClsLoss_g = (
            F.cross_entropy(df_cls_ss, cf_ss, reduction='sum') + 
            F.cross_entropy(df_cls_st, cf_st, reduction='sum') + 
            F.cross_entropy(df_cls_tt, cf_tt, reduction='sum') + 
            F.cross_entropy(df_cls_ts, cf_ts, reduction='sum')
        ) / (df_cls_ss.numel() + df_cls_st.numel() + df_cls_tt.numel() + df_cls_ts.numel())

        return ClsLoss_g

    def calc_advloss_d(self, x_s, x_t, xf_ts, xf_st, dr_adv_s, dr_adv_t, df_adv_ss, df_adv_st, df_adv_tt, df_adv_ts):
        device = x_s.device
        B_s = len(x_s)
        B_t = len(x_t)
        
        dr_adv_s = dr_adv_s.permute(0,2,1).reshape(-1,1)
        dr_adv_t = dr_adv_t.permute(0,2,1).reshape(-1,1)
        df_adv_ss = df_adv_ss.permute(0,2,1).reshape(-1,1)
        df_adv_st = df_adv_st.permute(0,2,1).reshape(-1,1)
        df_adv_tt = df_adv_tt.permute(0,2,1).reshape(-1,1)
        df_adv_ts = df_adv_ts.permute(0,2,1).reshape(-1,1)

        if self.loss_type=='wgan':
            # Wasserstein GAN with gradient penalty (WGAN-GP)
            AdvLoss_d_r = (
                torch.sum(-dr_adv_s) + 
                torch.sum(-dr_adv_t)
            ) / (dr_adv_s.numel() + dr_adv_t.numel())
            AdvLoss_d_f = (
                torch.sum(df_adv_ss) + 
                torch.sum(df_adv_st) + 
                torch.sum(df_adv_tt) + 
                torch.sum(df_adv_ts)
            ) / (df_adv_ss.numel() + df_adv_st.numel() + df_adv_tt.numel() + df_adv_ts.numel())
            AdvLoss_d = AdvLoss_d_r + AdvLoss_d_f

        elif self.loss_type=='lsgan':
            # Least squares GAN (LSGAN)

            AdvLoss_d_r = 0.5 * (
                torch.sum((dr_adv_s - torch.ones_like(dr_adv_s))**2) + 
                torch.sum((dr_adv_t - torch.ones_like(dr_adv_t))**2)
            ) / (dr_adv_s.numel() + dr_adv_t.numel())
            AdvLoss_d_f = 0.5 * (
                torch.sum(df_adv_ss**2) + 
                torch.sum(df_adv_st**2) + 
                torch.sum(df_adv_tt**2) + 
                torch.sum(df_adv_ts**2)
            ) / (df_adv_ss.numel() + df_adv_st.numel() + df_adv_tt.numel() + df_adv_ts.numel())
            AdvLoss_d = AdvLoss_d_r + AdvLoss_d_f

        elif self.loss_type=='cgan':
            # Regular GAN with sigmoid cross-entropy criterion (CGAN)
            AdvLoss_d_r = (
                F.binary_cross_entropy_with_logits(dr_adv_s, torch.ones_like(dr_adv_s), reduction='sum') +
                F.binary_cross_entropy_with_logits(dr_adv_t, torch.ones_like(dr_adv_t), reduction='sum')
            ) / (dr_adv_s.numel() + dr_adv_t.numel())
            AdvLoss_d_f = (
                F.binary_cross_entropy_with_logits(df_adv_ss, torch.zeros_like(df_adv_ss), reduction='sum') +
                F.binary_cross_entropy_with_logits(df_adv_st, torch.zeros_like(df_adv_st), reduction='sum') +
                F.binary_cross_entropy_with_logits(df_adv_tt, torch.zeros_like(df_adv_tt), reduction='sum') +
                F.binary_cross_entropy_with_logits(df_adv_ts, torch.zeros_like(df_adv_ts), reduction='sum')
            ) / (df_adv_ss.numel() + df_adv_st.numel() + df_adv_tt.numel() + df_adv_ts.numel())
            AdvLoss_d = AdvLoss_d_r + AdvLoss_d_f

        # Gradient penalty loss
        alpha_t = torch.rand(B_t, 1, 1, requires_grad=True).to(device)
        interpolates = alpha_t * x_t + ((1 - alpha_t) * xf_ts)
        interpolates = interpolates.to(device)
        disc_interpolates, _ = self.dis(interpolates)
        disc_interpolates = torch.sum(disc_interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradnorm = torch.sqrt(torch.sum(gradients * gradients, (1, 2)))
        loss_gp_t = ((gradnorm - 1)**2).mean()

        alpha_s = torch.rand(B_s, 1, 1, requires_grad=True).to(device)
        interpolates = alpha_s * x_s + ((1 - alpha_s) * xf_st)
        interpolates = interpolates.to(device)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates, _ = self.dis(interpolates)
        disc_interpolates = torch.sum(disc_interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradnorm = torch.sqrt(torch.sum(gradients * gradients, (1, 2)))
        loss_gp_s = ((gradnorm - 1)**2).mean()

        GradLoss_d = loss_gp_s + loss_gp_t

        return AdvLoss_d, GradLoss_d

    def calc_clsloss_d(self, dr_cls_s, dr_cls_t, k_s, k_t):
        device = dr_cls_s.device

        dr_cls_s = dr_cls_s.permute(0,2,1).reshape(-1,self.n_spk)
        dr_cls_t = dr_cls_t.permute(0,2,1).reshape(-1,self.n_spk)

        cr_s = k_s*torch.ones(len(dr_cls_s), device=device, dtype=torch.long)
        cr_t = k_t*torch.ones(len(dr_cls_t), device=device, dtype=torch.long)

        ClsLoss_d = (
            F.cross_entropy(dr_cls_s, cr_s, reduction='sum') + 
            F.cross_entropy(dr_cls_t, cr_t, reduction='sum')
        ) / (dr_cls_s.numel() + dr_cls_t.numel())

        return ClsLoss_d

    def calc_gen_loss(self, x_s, x_t, k_s, k_t):
        # Generator outputs        
        xf_ss = self.gen(x_s, k_s, k_s)
        xf_ts = self.gen(x_t, k_s, k_t)
        xf_tt = self.gen(x_t, k_t, k_t)
        xf_st = self.gen(x_s, k_t, k_s)

        # Discriminator outputs
        df_adv_ss, df_cls_ss = self.dis(xf_ss)
        df_adv_st, df_cls_st = self.dis(xf_st)
        df_adv_tt, df_cls_tt = self.dis(xf_tt)
        df_adv_ts, df_cls_ts = self.dis(xf_ts)

        # Adversarial loss 
        AdvLoss_g = self.calc_advloss_g(df_adv_ss, df_adv_st, df_adv_tt, df_adv_ts)

        # Classifier loss
        ClsLoss_g = self.calc_clsloss_g(df_cls_ss, df_cls_st, df_cls_tt, df_cls_ts, k_s, k_t)
        
        # Cycle-consistency loss
        CycLoss = (
            torch.sum(torch.abs(x_s - self.gen(xf_st, k_s, k_t))) + 
            torch.sum(torch.abs(x_t - self.gen(xf_ts, k_t, k_s)))
        ) / (x_s.numel() + x_t.numel())

        # Reconstruction loss
        RecLoss = (
            torch.sum(torch.abs(x_s - xf_ss)) + 
            torch.sum(torch.abs(x_t - xf_tt))
        ) / (x_s.numel() + x_t.numel())

        return AdvLoss_g, ClsLoss_g, CycLoss, RecLoss

    def calc_dis_loss(self, x_s, x_t, k_s, k_t):
        device = x_s.device

        # Generator outputs        
        xf_ss = self.gen(x_s, k_s, k_s)
        xf_ts = self.gen(x_t, k_s, k_t)
        xf_tt = self.gen(x_t, k_t, k_t)
        xf_st = self.gen(x_s, k_t, k_s)

        # Discriminator outputs
        dr_adv_s, dr_cls_s = self.dis(x_s)
        dr_adv_t, dr_cls_t = self.dis(x_t)
        df_adv_ss, _ = self.dis(xf_ss)
        df_adv_st, _ = self.dis(xf_st)
        df_adv_tt, _ = self.dis(xf_tt)
        df_adv_ts, _ = self.dis(xf_ts)

        # Adversarial loss
        AdvLoss_d, GradLoss_d = self.calc_advloss_d(x_s, x_t, xf_ts, xf_st, dr_adv_s, dr_adv_t, df_adv_ss, df_adv_st, df_adv_tt, df_adv_ts)

        # Classifier loss
        ClsLoss_d = self.calc_clsloss_d(dr_cls_s, dr_cls_t, k_s, k_t)

        return AdvLoss_d, GradLoss_d, ClsLoss_d
