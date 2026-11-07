
"""TinyBrainNet-P21: unified classification + segmentation + learned attention map, <= 500,000 parameters.
Input  : images [B, 1, 224, 224], canonical preprocessing (see preprocess.py), normalised with the training mean/std.
Outputs: cls_logits [B, 4] (Glioma, Meningioma, Pituitary, No_tumor), seg_logits [B, 1, 224, 224], cam [B, 1, 14, 14]
         (sigmoid of the mask-supervised attention logits), plus att_logits and aux_seg (training only)."""
import torch, torch.nn as nn, torch.nn.functional as F

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__(); se_c = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1); self.fc1 = nn.Conv2d(channels, se_c, 1); self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(se_c, channels, 1); self.act2 = nn.Sigmoid()
    def forward(self, x): return x * self.act2(self.fc2(self.act1(self.fc1(self.pool(x)))))

class InvertedResidualSE(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4, se_reduction=4):
        super().__init__(); hidden = int(round(in_c * expand_ratio)); self.use_residual = (stride == 1 and in_c == out_c); layers = []
        if expand_ratio != 1: layers += [nn.Conv2d(in_c, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
        else: hidden = in_c
        layers += [nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
        self.expand_dw = nn.Sequential(*layers); self.se = SqueezeExcite(hidden, reduction=se_reduction)
        self.project = nn.Sequential(nn.Conv2d(hidden, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
    def forward(self, x):
        out = self.project(self.se(self.expand_dw(x)))
        return out + x if self.use_residual else out

class TinyBrainNetV6Arch(nn.Module):
    """Historical v6 classifier (cell 66), 3-channel input. Used as the Plan-B initialisation and classification teacher."""
    def __init__(self, num_classes=4, width_mult=0.95, dropout=0.25, in_ch=3):
        super().__init__(); c = lambda ch: max(8, int(round(ch * width_mult / 4) * 4))
        C0, C1, C2, C3, C4, C5 = [c(x) for x in [16, 24, 32, 48, 64, 96]]
        self.stem = nn.Sequential(nn.Conv2d(in_ch, C0, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C0), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(InvertedResidualSE(C0, C1, 1, 1))
        self.stage2 = nn.Sequential(InvertedResidualSE(C1, C2, 2, 4), InvertedResidualSE(C2, C2, 1, 4))
        self.stage3 = nn.Sequential(InvertedResidualSE(C2, C3, 2, 4), InvertedResidualSE(C3, C3, 1, 4))
        self.stage4 = nn.Sequential(InvertedResidualSE(C3, C4, 2, 4), InvertedResidualSE(C4, C4, 1, 4), InvertedResidualSE(C4, C4, 1, 4))
        self.stage5 = nn.Sequential(InvertedResidualSE(C4, C5, 1, 4), InvertedResidualSE(C5, C5, 1, 4))
        self.head_conv = nn.Sequential(nn.Conv2d(C5, C5 * 2, 1, bias=False), nn.BatchNorm2d(C5 * 2), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1); self.dropout = nn.Dropout(dropout); self.fc = nn.Linear(C5 * 2, num_classes)
    def forward(self, x):
        x = self.stage5(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x))))))
        return self.fc(self.dropout(self.pool(self.head_conv(x)).flatten(1)))

def _out_c(seq): return seq[-1].project[0].out_channels

class UDSConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(i, i, 3, padding=1, groups=i, bias=False), nn.BatchNorm2d(i), nn.ReLU(inplace=True),
                               nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)

class UUp(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__(); self.c1 = UDSConv(c_in + c_skip, c_out); self.c2 = UDSConv(c_out, c_out)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.c2(self.c1(torch.cat([x, skip], 1)))

class InputAdapter1(nn.Module):
    """OPTIONAL (ablation A4): learned intensity correction x + g(InstanceNorm(x)); zero-initialised -> identity at start."""
    def __init__(self, hid=8):
        super().__init__(); self.norm = nn.InstanceNorm2d(1, affine=False); self.c1 = nn.Conv2d(1, hid, 3, padding=1); self.c2 = nn.Conv2d(hid, 1, 1)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)
    def forward(self, x): return x + self.c2(F.relu(self.c1(self.norm(x))))

class TinyBrainNetP21(nn.Module):
    """Part-16 UNI-NET skeleton with a 1-channel stem, a mask-supervised attention head (-> cam) and an auxiliary 56x56 head."""
    def __init__(self, use_adapter=False, dec=(56, 40, 32, 24), full_ch=16, width_mult=0.95, dropout=0.25):
        super().__init__()
        e = TinyBrainNetV6Arch(4, width_mult, dropout, in_ch=1)
        self.adapter = InputAdapter1() if use_adapter else nn.Identity()
        self.stem, self.stage1, self.stage2, self.stage3, self.stage4, self.stage5 = e.stem, e.stage1, e.stage2, e.stage3, e.stage4, e.stage5
        self.head_conv, self.dropout, self.fc = e.head_conv, e.dropout, e.fc
        c1, c2, c3, c4, c5 = (_out_c(s) for s in (self.stage1, self.stage2, self.stage3, self.stage4, self.stage5))
        self.att = nn.Conv2d(c5 * 2, 1, 1); nn.init.zeros_(self.att.weight); nn.init.zeros_(self.att.bias)
        self.bott = UDSConv(c5 + c4, dec[0]); self.up3 = UUp(dec[0], c3, dec[1]); self.up2 = UUp(dec[1], c2, dec[2])
        self.up1 = UUp(dec[2], c1, dec[3]); self.full = UDSConv(dec[3], full_ch); self.seg_head = nn.Conv2d(full_ch, 1, 1)
        self.aux_head = nn.Conv2d(dec[2], 1, 1)

    def forward(self, x, task="both"):
        if x.dim() != 4 or x.shape[1] != 1: raise ValueError(f"expected [B,1,H,W], got {tuple(x.shape)}")
        H, W = x.shape[-2:]; x = self.adapter(x)
        e1 = self.stage1(self.stem(x)); e2 = self.stage2(e1); e3 = self.stage3(e2); e4 = self.stage4(e3); e5 = self.stage5(e4); out = {}
        if task in ("cls", "both"):
            f = self.head_conv(e5); a = self.att(f)
            w = torch.softmax(a.flatten(2), -1)                                   # attention weights over the 14x14 grid
            pooled = 0.5 * f.mean((2, 3)) + 0.5 * (f.flatten(2) * w).sum(-1)     # equals GAP while att is zero-initialised
            out.update(cls_logits=self.fc(self.dropout(pooled)), att_logits=a, cam=torch.sigmoid(a), feat=pooled)
        if task in ("seg", "both"):
            d = self.bott(torch.cat([e5, e4], 1)); d2 = self.up2(self.up3(d, e3), e2); d1 = self.up1(d2, e1)
            out["seg_logits"] = self.seg_head(self.full(F.interpolate(d1, size=(H, W), mode="bilinear", align_corners=False)))
            out["aux_seg"] = self.aux_head(d2)
        return out

def build_model(use_adapter=False): return TinyBrainNetP21(use_adapter=use_adapter)

def count_params(m, trainable_only=True): return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)
