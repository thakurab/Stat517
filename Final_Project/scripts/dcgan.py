from torch import nn

# NOT USED!!!!!!!!!!
class gen_loss(nn.Module):

    def __init__(self, BCE_weight=1):
        super(gen_loss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.discloss = nn.BCELoss()
        self.BCE_weight = BCE_weight

    def forward(self, netd_out, netd_expect, netg_out, netg_expect):
        BCE = self.discloss(netd_out, netd_expect)
        L1L = self.l1loss(netg_out, netg_expect)

        return 0.3*BCE + 0.7*L1L
