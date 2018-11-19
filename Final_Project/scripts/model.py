# model.py

from torch import nn
import models
import losses


# custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Model:
    def __init__(self, args):
        self.ndf = args.ndf
        self.nef = args.nef
        self.cuda = args.cuda
        self.wkld = args.wkld
        self.gbweight = args.gbweight
        self.nlatent = args.nlatent
        self.nechannels = args.nechannels
        self.ngchannels = args.ngchannels
        
    def setup(self, checkpoints):
        model = {}
        criterion = {}
        #model["netE"] = models.dcgan._netE(self.nechannels, self.nlatent, self.nef)     #i have commented this line
        # self.nlatent is length of z vector. Adding 1 to it for gbtt
        model["netG"] = models.dcgan._netG(self.ngchannels, self.ndf, self.nlatent)
        #criterion["netG"] = losses.ge                                                  #i have commented this line
        #criterion["netE"] = losses.vae_loss(self.wkld, self.gbweight)                  #I have commented this line

        criterion["netG"] = losses.vae_loss(self.wkld, self.gbweight)                   #i have added this line

        if self.cuda:
            #model['netE'] = nn.DataParallel(model['netE']).cuda()                       #i have commented this line
            model['netG'] = nn.DataParallel(model['netG']).cuda()                      #I have commented this line
            #criterion["netE"] = criterion["netE"].cuda()                               #I have commented this line
            criterion["netG"] = criterion["netG"].cuda()                                #I have changed netE to netG in the RHS
            #criterion["netG"] = losses.vae_loss(self.wkld, self.gbweight)

        models_to_resume = checkpoints.latest('resume')
        for name, net in model.items():
            if name in models_to_resume:
                tmp = checkpoints.load(name)
                net.load_state_dict(tmp)
            else:
                net.apply(weights_init)

        return model, criterion
