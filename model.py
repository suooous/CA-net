import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 定义卷积、批归一化和ReLU激活函数
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 定义Patch Self-Attention模块
# 将输入的特征图展开为tokens,然后通过线性层投影得到q,k,v
# 然后通过q,k,v计算注意力权重,然后通过注意力权重计算得到输出
# 最后通过线性层投影得到输出
class PatchSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B,N,C)
        out = self.proj(out)
        return out

# 定义StitchViTBlock模块
# 将输入的特征图分成多个patch,然后通过Patch Self-Attention模块计算注意力权重
# 然后通过注意力权重计算得到输出
# 最后通过线性层投影得到输出
# 输入的特征图在通道维度上分割成多个部分，
# 并对每个部分以不同的采样率进行下采样。然后对下采样的部分应用自注意力，
# 再将注意力结果上采样回原始尺寸并拼接起来。
class StitchViTBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stitch_rates=(2,4), num_heads=4):
        super().__init__()
        self.stitch_rates = stitch_rates
        self.attns = nn.ModuleList()
        splits = [in_ch // len(stitch_rates)] * len(stitch_rates)
        for i in range(in_ch - sum(splits)):
            splits[i] += 1
        self.splits = splits
        for sd in splits:
            self.attns.append(PatchSelfAttention(sd, num_heads=num_heads))
        self.ffn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def sample_patch(self, x, rate):
        return x[:,:,::rate,::rate]

    def forward(self, x):
        B,C,H,W = x.shape
        splits = torch.split(x, self.splits, dim=1)
        outs = []
        for i, part in enumerate(splits):
            rate = self.stitch_rates[i % len(self.stitch_rates)]
            sampled = self.sample_patch(part, rate)
            b,c_i,h_s,w_s = sampled.shape
            tokens = sampled.flatten(2).transpose(1,2)
            attn_out = self.attns[i](tokens)
            attn_map = attn_out.transpose(1,2).reshape(B, c_i, h_s, w_s)
            attn_up = F.interpolate(attn_map, size=(H,W), mode='bilinear', align_corners=False)
            outs.append(attn_up)
        fused = torch.cat(outs, dim=1)
        out = self.ffn(fused)
        return out

# ================== 新增：ASPP 模块 ==================
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context."""
    def __init__(self, in_ch, out_ch, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            if d == 1:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    )
                )
        # image pooling branch
        self.img_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch*(len(dilations)+1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        H, W = x.shape[2:]
        feats = [b(x) for b in self.branches]
        gp = self.img_pool(x)
        gp = F.interpolate(gp, size=(H, W), mode='bilinear', align_corners=False)
        feats.append(gp)
        out = torch.cat(feats, dim=1)
        return self.project(out)

# ================== 新增：轻量自注意力（空间维度） ==================
class LiteSelfAttention2D(nn.Module):
    """
    轻量 2D self-attention：在空间 HxW 维度上做注意力，通道先降维再升回，控制显存开销。
    """
    def __init__(self, in_ch, heads=4, reduction=4):
        super().__init__()
        assert in_ch % reduction == 0, "in_ch 应能被 reduction 整除以便降维"
        self.heads = heads
        self.key_ch = in_ch // reduction
        self.to_q = nn.Conv2d(in_ch, self.key_ch, 1, bias=False)
        self.to_k = nn.Conv2d(in_ch, self.key_ch, 1, bias=False)
        self.to_v = nn.Conv2d(in_ch, self.key_ch, 1, bias=False)
        self.proj = nn.Conv2d(self.key_ch, in_ch, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.to_q(x).view(B, self.heads, self.key_ch // self.heads, H*W)  # B,h,c',N
        k = self.to_k(x).view(B, self.heads, self.key_ch // self.heads, H*W)
        v = self.to_v(x).view(B, self.heads, self.key_ch // self.heads, H*W)

        q = q.transpose(-2, -1)   # B,h,N,c'
        k = k                      # B,h,c',N
        attn = torch.matmul(q, k) * (1.0 / (self.key_ch // self.heads) ** 0.5)  # B,h,N,N
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1))  # B,h,N,c'
        out = out.transpose(-2, -1).contiguous().view(B, self.key_ch, H, W)
        out = self.proj(out)
        return out + x  # 残差

# ================== 替换：CCM（带补全能力） ==================
class CCM(nn.Module):
    """
    Contour Completion Module with ASPP + LiteSelfAttention2D.
    - 接收高层/低层特征，输出补全后的轮廓图 (1通道 sigmoid)；
    - 输出将轮廓注入后的增强低层特征 (out_ch)。
    """
    def __init__(self, in_ch_high, in_ch_low, mid_ch=128, out_ch=64,
                 aspp_out=128, attn_heads=4, attn_reduction=4):
        super().__init__()
        # 对齐通道
        self.proj_high = nn.Conv2d(in_ch_high, mid_ch, 1)
        self.proj_low  = nn.Conv2d(in_ch_low,  mid_ch, 1)

        # encoder
        self.enc1 = ConvBNReLU(mid_ch*2, mid_ch)   # 融合起点
        self.enc2 = ConvBNReLU(mid_ch,   mid_ch)
        # bottleneck with ASPP（多尺度上下文补全）
        self.aspp = ASPP(mid_ch, aspp_out, dilations=(1, 6, 12, 18))
        # 解码
        self.dec1 = ConvBNReLU(aspp_out, mid_ch)
        self.dec2 = ConvBNReLU(mid_ch,   mid_ch//2)

        # 轻量自注意力（全局结构闭合）
        self.attn_refine = LiteSelfAttention2D(mid_ch//2, heads=attn_heads, reduction=attn_reduction)

        # 轮廓头（保持与原实现一致，便于无缝替换）
        self.contour_head = nn.Sequential(
            nn.Conv2d(mid_ch//2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 将轮廓注入低层特征以产生增强特征（供后续解码使用）
        self.enhance = nn.Sequential(
            nn.Conv2d(in_ch_low + 1, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_high, f_low):
        # 对齐 + 融合
        fh = self.proj_high(f_high)                              # [B, mid, Hh, Wh]
        fl = self.proj_low(f_low)                                # [B, mid, Hl, Wl]
        fh_up = F.interpolate(fh, size=fl.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([fh_up, fl], dim=1)                        # [B, 2*mid, Hl, Wl]

        # 编码 + ASPP 补全 + 解码
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.aspp(x)                                         # 多尺度上下文补全
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.attn_refine(x)                                  # 全局结构闭合（轻量注意力）

        # 轮廓预测（补全后的）
        contour = self.contour_head(x)                           # [B,1,Hl,Wl]

        # 轮廓注入增强低层特征
        enhanced = torch.cat([f_low, contour], dim=1)
        enhanced = self.enhance(enhanced)                        # [B,out_ch,Hl,Wl]
        return enhanced, contour


# # CCM: Contour Completion Module
# # 它的作用是接收主干网络的高层和低层特征，预测一个完整的轮廓图。
# # 它内部包含一个U型结构（或类似结构），通过上采样和下采样融合不同尺度的特征。
# # 最终输出一个轮廓预测图，并生成一个将低层特征与轮廓信息融合后的增强特征。
# class CCM(nn.Module):
#     """Contour Completion Module (replaces BEM).
#     Predicts a completed contour map and returns a contour-enhanced encoder feature.
#     """
#     def __init__(self, in_ch_high, in_ch_low, mid_ch=128, out_ch=64):
#         super().__init__()
#         self.proj_high = nn.Conv2d(in_ch_high, mid_ch, 1)
#         self.proj_low  = nn.Conv2d(in_ch_low, mid_ch, 1)
#         self.enc1 = ConvBNReLU(mid_ch*2, mid_ch)
#         self.enc2 = ConvBNReLU(mid_ch, mid_ch)
#         self.dec1 = ConvBNReLU(mid_ch, mid_ch)
#         self.dec2 = ConvBNReLU(mid_ch, mid_ch//2)
#         self.contour_head = nn.Sequential(
#             nn.Conv2d(mid_ch//2, 1, 3, padding=1),
#             nn.Sigmoid()
#         )
#         self.enhance = nn.Sequential(
#             nn.Conv2d(in_ch_low + 1, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, f_high, f_low):
#         fh = self.proj_high(f_high)
#         fl = self.proj_low(f_low)
#         fh_up = F.interpolate(fh, size=fl.shape[2:], mode='bilinear', align_corners=False)
#         x = torch.cat([fh_up, fl], dim=1)
#         x = self.enc1(x)
#         x = self.enc2(x)
#         x = self.dec1(x)
#         x = self.dec2(x)
#         contour = self.contour_head(x)
#         enhanced = torch.cat([f_low, contour], dim=1)
#         enhanced = self.enhance(enhanced)
#         return enhanced, contour

# CIM: Contour Injection Module
# 它的作用是接收一个轮廓图、一个编码器特征和一个前一个解码器特征。
# 它使用一个门控机制，利用轮廓和先前解码器特征来调制编码器特征，然后将所有特征进行融合，
# 从而在解码过程中逐步注入轮廓信息。
class CIM(nn.Module):
    def __init__(self, enc_ch, dec_prev_ch, out_ch):
        super().__init__()
        self.contour_proj = nn.Conv2d(1, enc_ch, 1)
        # 添加编码器特征投影层，将不同通道数统一到out_ch
        self.enc_proj = nn.Conv2d(enc_ch, out_ch, 1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(enc_ch + dec_prev_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(out_ch + dec_prev_ch + enc_ch, out_ch),
            ConvBNReLU(out_ch, out_ch)
        )
        self.out_proj = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, contour, Fc, prev_d):
        cont = self.contour_proj(contour)
        cont = F.interpolate(cont, size=Fc.shape[2:], mode='bilinear', align_corners=False)
        prev_up = F.interpolate(prev_d, size=Fc.shape[2:], mode='bilinear', align_corners=False)
        gate_in = torch.cat([cont, prev_up], dim=1)
        gate = self.gate_conv(gate_in)
        # 投影编码器特征到统一通道数
        Fc_proj = self.enc_proj(Fc)
        gated_enc = Fc_proj * gate
        fused = torch.cat([gated_enc, prev_up, cont], dim=1)
        out = self.fuse(fused)
        out = self.out_proj(out)
        return out

# 它的作用是接收来自主干网络和CCM的融合特征，以及编码器特征，
# 并利用CIM模块逐步进行特征融合和上采样。最终，它在每个解码层都产生一个分割预测，
# 并通过上采样将预测结果恢复到原始输入图像的分辨率
class Decoder(nn.Module):
    def __init__(self, fused_ch, enc_channels, num_levels=3, decode_ch=64, num_classes=3):
        super().__init__()
        self.initial = ConvBNReLU(fused_ch, decode_ch)
        self.cims = nn.ModuleList()
        for i in range(num_levels):
            enc_ch = enc_channels[i]
            self.cims.append(CIM(enc_ch, decode_ch, decode_ch))
        # 修改输出头以支持多类别
        self.head = nn.Conv2d(decode_ch, num_classes, 1)

    def forward(self, fused, enc_feats, contour, input_size):
        d = self.initial(fused)
        preds = []
        for i, cim in enumerate(self.cims):
            enc_feat = enc_feats[i]
            d = F.interpolate(d, size=enc_feat.shape[2:], mode='bilinear', align_corners=False)
            d = cim(contour, enc_feat, d)
            # 移除sigmoid，因为多类别使用softmax
            pred = self.head(d)
            # 上采样到原始输入图像的分辨率
            pred = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=False)
            preds.append(pred)
        return preds

class CANet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, stitch_rates=(2,4), num_classes=3):
        super().__init__()
        # 修复pretrained参数警告
        if pretrained:
            res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            res = models.resnet50(weights=None)
        self.stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4
        self.stitch = StitchViTBlock(in_ch=2048, out_ch=256, stitch_rates=stitch_rates, num_heads=4)
        self.ccm = CCM(in_ch_high=2048, in_ch_low=256, mid_ch=128, out_ch=64)
        self.fuse_conv = nn.Conv2d(2048 + 256, 256, 1)
        enc_channels = [64, 512, 1024]  # note: first enc channel becomes CCM out_ch (64)
        self.decoder = Decoder(fused_ch=256, enc_channels=enc_channels, num_levels=3, decode_ch=64, num_classes=num_classes)

    def forward(self, x):
        s = self.stem(x)
        f1 = self.layer1(s)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        sv = self.stitch(f4)
        f4_proj = F.interpolate(f4, size=sv.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([f4_proj, sv], dim=1)
        fused = self.fuse_conv(fused)
        enhanced_feat, contour_pred = self.ccm(f4, f1)
        # 将轮廓预测上采样到原始输入图像的分辨率
        contour_pred = F.interpolate(contour_pred, size=x.shape[2:], mode='bilinear', align_corners=False)
        enc_feats = [enhanced_feat, f2, f3]
        preds = self.decoder(fused, enc_feats, contour_pred, x.shape[2:])
        return preds, contour_pred
