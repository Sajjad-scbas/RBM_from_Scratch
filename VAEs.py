import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange, tqdm


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = nn.ReLU()(self.fc1(x))
        h = nn.ReLU()(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps*std + mu
        
    def decoder(self, z):
        h = nn.ReLU()(self.fc4(z))
        h = nn.ReLU()(self.fc5(h))
        return nn.Sigmoid()(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
    



class ConvVAE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, latent_size):
        super(ConvVAE, self).__init__()

        #Encoder 
        self.enc1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=2, padding=1)
        self.enc2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, 
                              kernel_size=kernel_size, stride=2, padding=1)
        self.enc3 = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*4, 
                              kernel_size=kernel_size, stride=2, padding=1)        
        self.enc4 = nn.Conv2d(in_channels=out_channels*4, out_channels=64, 
                              kernel_size=2, stride=2, padding=0)
        
        #FC Layers for the reparametrization
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_size)
        self.fc_log_var = nn.Linear(128, latent_size)
        self.fc2 = nn.Linear(latent_size, 64)

        #Decoder 
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(in_channels=64, out_channels=out_channels*8,
                                       kernel_size=kernel_size, stride=1, padding=0),
            nn.ReLU(),
                                    
            nn.ConvTranspose2d(in_channels=out_channels*8, out_channels=out_channels*4,
                                       kernel_size=kernel_size, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=out_channels*4, out_channels=out_channels*2,
                                       kernel_size=kernel_size, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=out_channels*2, out_channels=in_channels,
                                       kernel_size=kernel_size, stride=2, padding=3),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample


    def forward(self, x):
        # encoding
        x = nn.ReLU()(self.enc1(x))
        x = nn.ReLU()(self.enc2(x))
        x = nn.ReLU()(self.enc3(x))
        x = nn.ReLU()(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)
 
        # decoding
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var
    

def loss_criterion(inputs, targets, logvar, mu, criterion):
    bce_loss = criterion(inputs, targets)
    kl_loss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kl_loss


def train(model, data_loader, optimizer, device):

    train_loss = 0

    model.train()

    for (x, _) in tqdm(data_loader, desc="Evaluating", leave=False):

        if model.__class__.__name__ == 'ConvVAE':
            x = x.reshape(x.shape[0],1,28,28).to(device)
            criterion = nn.BCELoss(reduction='sum')
        else:
            x = x.to(device)
            criterion = nn.CrossEntropyLoss( reduction='sum')
        optimizer.zero_grad()

        recon_x, mu, log_var = model(x)

        loss = loss_criterion(recon_x, x, log_var, mu, criterion)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()


    return train_loss / len(data_loader)



def evaluate(model, data_loader, device):

    val_loss = 0

    model.eval()

    with torch.no_grad():

        for (x, _) in tqdm(data_loader, desc="Evaluating", leave=False):

            if model.__class__.__name__ == 'ConvVAE':
                x = x.reshape(x.shape[0],1,28,28).to(device)
                criterion = nn.BCELoss(reduction='sum')
            else:
                x = x.to(device)
                criterion = nn.CrossEntropyLoss( reduction='sum')
            
            recon_x, mu, log_var = model(x)
            
            loss = loss_criterion(recon_x, x, log_var, mu, criterion)
            
            val_loss += loss.item()
            

    return val_loss / len(data_loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


 


