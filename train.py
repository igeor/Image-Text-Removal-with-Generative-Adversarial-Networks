from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision
import torch 
from torch import nn

from networks.ae import AutoEncoder
from utils.train_options import *
from networks.patchgan import *
from networks.deep_unet import *
from networks.unet import *
from networks.unet64 import *
from networks.attunet import *
from datasets.KonIQDataset import *
from datasets.TestDataset import * 
from datasets.SynthTextDataset import *
from datasets.ValidDataset import *

def train():
	device = 'cuda'
	
	#gen = Generator(3,3)
	#gen = UNet()
	#gen = UNet64()
	gen = AttU_Net(img_ch=3, output_ch=3)

	gen.load_state_dict(torch.load('/home/igeorvasilis/thesis_src/checkpoints/latest_Gatt(L1c).pth'))
	#gen.load_state_dict(torch.load('/home/igeorvasilis/thesis_src/checkpoints/latest_G{+11}_synthtext_plus_text.pth'))
	#gen.load_state_dict(torch.load("/home/igeorvasilis/thesis_src/checkpoints/latest_G{+2}_synthtext_plus_text.pth"))
	
	#gen.load_state_dict(torch.load('/home/igeorvasilis/thesis_src/checkpoints/latest_G.pth'))
	#sd = torch.load('/home/igeorvasilis/diploma thesis/noise2noise-pytorch-master/latest_G_opt.pth')
	#print(sd)
	#gen.load_state_dict(torch.load('/home/igeorvasilis/diploma thesis/noise2noise-pytorch-master/latest_G_opt.pth'))
	gen = gen.to(device)
	disc = PatchGAN(3 + 3).to(device)
	

	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Resize((256,256))
	])
	#torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
	#						std=[0.229, 0.224, 0.225])

	dataset = KonIQDataset('/home/igeorvasilis/sdb/KoniQ_dataset/B/train', transform=transform)
	dataset = SynthTextDataset(transform=transform, extra_text=True)
	dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

	TEST_ROOT = "/home/igeorvasilis/diploma thesis/noise2noise-pytorch-master/data/test_out/test"
	test_dataset = TestDataset(TEST_ROOT, transform=transform, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=32)

	transform_val = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Resize((256,512))
	])

	val_dataset = ValidDataset('/home/igeorvasilis/sdb/KoniQ_dataset/A_B/test', transform=transform_val)
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)


	gen_opt = torch.optim.Adam(gen.parameters(), lr=0.0002)
	disc_opt = torch.optim.Adam(disc.parameters(), lr=0.0001)

	recon_criterion = nn.L1Loss().to(device)
	adv_criterion = nn.BCEWithLogitsLoss().to(device)

	mean_generator_loss = 0.0 
	mean_rec_loss = 0.0 
	mean_discriminator_loss = 0.0
	mean_generator_adv_loss = 0.0

	display_dir = '/home/igeorvasilis/sdb/out_train'
	display_step = 200
	lambda_recon = 100
	epochs = 150
	step = 0

	def PSNR(img1, img2):
		"""Peak Signal to Noise Ratio
		img1 and img2 have range [0, 255]"""
		img1 *= 255.0
		img2 *= 255.0 
		
		mse = torch.mean((img1 - img2) ** 2)
		return 20 * torch.log10(255.0 / torch.sqrt(mse))

	def test(unet, test_loader, epoch):
		test_out = "/home/igeorvasilis/diploma thesis/noise2noise-pytorch-master/data/test_out/"
		eval_score = 0.0
		for batch_id, (condition) in enumerate(test_loader):
			
			condition = condition.to(device)

			with torch.no_grad():
				fake = unet(condition)
			out = torch.cat([condition, fake], dim=2).squeeze(1)
			save_image(out, test_out+str(epoch)+".png")   

	def validate(val_loader, gen):

		psnr_score = 0.0
		for batch_id, (input_image) in enumerate(val_loader):
			
			input_image = input_image[0].to(device)

			#print(input_image.shape)
			condition = input_image[:, :, :input_image.shape[2] // 2].unsqueeze(0)
			target = input_image[:, :, input_image.shape[2] // 2 :].unsqueeze(0)
		
			with torch.no_grad():
				fake = gen(condition)
			psnr_score += PSNR(fake, target) / len(val_loader)

		print("psnr: ",psnr_score.item())

    
    
	for epoch in range(epochs):

		torch.save(gen.state_dict(), 'checkpoints/latest_Gattg.pth')
		validate(val_loader, gen)
		#test(gen, test_loader, epoch)

		for real, condition in dataloader:
			condition = condition.to(device)
			real = real.to(device)

			#-----------------
			#train generator
			#-----------------
			gen_opt.zero_grad()
			fake = gen.forward(condition) # generator returns a tensor with shape (batch_size,3,h,w)
			evaluation = disc.forward(fake, condition) # disc returns a tensor with shape (batch_size,1,16,16) that contain the predictions
			gen_adv_loss = adv_criterion(evaluation, torch.ones_like(evaluation)) 
			gen_rec_loss = recon_criterion(real, fake) * lambda_recon 
			gen_loss = 0.5 * gen_adv_loss + gen_rec_loss
			gen_loss.backward() # Update gradients
			gen_opt.step() # Update optimizer

			#-----------------
			#train discriminator
			#-----------------
			disc_opt.zero_grad() # Zero out the gradient before backpropagation
			with torch.no_grad():
				fake = gen.forward(condition)
			disc_fake_hat = disc.forward(fake.detach(), condition) # Detach generator
			disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
			disc_real_hat = disc.forward(real, condition)
			disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat)) 
			disc_loss = (disc_fake_loss + disc_real_loss) / 2
			disc_loss.backward(retain_graph=True) # Update gradients
			disc_opt.step() # Update optimizer

			#-----------------
			#update losses
			#-----------------
			mean_generator_loss += gen_loss.item() / display_step
			mean_rec_loss += gen_rec_loss.item() / display_step
			mean_discriminator_loss += disc_loss.item() / display_step
			mean_generator_adv_loss += gen_adv_loss.item() / display_step

			#-----------------
			#display losses
			#-----------------
			if(step % display_step == 0 and step > 0):
			
				epoch_step = (f"Epoch {epoch}: Step {step}: ")
				display_gen = (f"Generator loss: "+  "{:.2f}".format(mean_generator_loss) +" ")
				display_disc = (f"Discriminator loss: "+  "{:.2f}".format(mean_discriminator_loss)+" ")
				display_rec_loss = (f"rec_loss: "+  "{:.2f}".format(mean_rec_loss)+" ")
				display_adv_loss = (f"adv_loss: "+  "{:.2f}".format(mean_generator_adv_loss)+" ")
				print(epoch_step + display_gen + display_adv_loss + display_rec_loss + display_disc)
				
				mean_generator_loss = 0.0
				mean_rec_loss = 0.0
				mean_discriminator_loss = 0.0
				mean_generator_adv_loss = 0.0

				out = torch.cat([real, condition, fake], dim=2).squeeze(1)
				save_image(out, display_dir+"/"+str(epoch)+"_"+str(step)+".png", normalize=False)

			step += 1
		#lambda_recon *= 0.98

if __name__ == '__main__':
	
	params = parse_args()
	train()

	

