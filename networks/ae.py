import torch.nn as nn
import torch

class AutoEncoder(nn.Module):
	"""Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

	def __init__(self, in_channels=3, out_channels=3):
		"""Initializes U-Net."""

		super(AutoEncoder, self).__init__()

		# Layers: enc_conv0, enc_conv1, pool1
		self._block1 = nn.Sequential(
			nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(48, 48, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2))
		
		self._block2 = nn.Sequential(
			nn.Conv2d(64, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2))
		
		self._block3 = nn.Sequential(
			nn.Conv2d(128, 256, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2))
		
		self._block4 = nn.Sequential(
			nn.Conv2d(256, 512, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2))
		
		self._block5 = nn.Sequential(
			nn.Conv2d(512, 1024, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(1024, 1024, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2))
			
		self._block6 = nn.Sequential(
			nn.Conv2d(1024, 512, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1))
			#nn.Upsample(scale_factor=2, mode='nearest'))
		
		self._block7 = nn.Sequential(
			nn.Conv2d(512, 256, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1))
			#nn.Upsample(scale_factor=2, mode='nearest'))
		
		self._block8 = nn.Sequential(
			nn.Conv2d(256, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1))
			#nn.Upsample(scale_factor=2, mode='nearest'))

		self._block9 = nn.Sequential(
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
			#nn.Upsample(scale_factor=2, mode='nearest'))

		self._block9 = nn.Sequential(
			nn.Conv2d(64, out_channels, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(out_channels, out_channels, 3, stride=2, padding=1, output_padding=1))
			#nn.Upsample(scale_factor=2, mode='nearest'))

		# Initialize weights
		self._init_weights()


	def _init_weights(self):
		"""Initializes weights using He et al. (2015)."""

		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight.data)
				m.bias.data.zero_()


	def forward(self, x):
		"""Through encoder, then decoder by adding U-skip connections. """

		# Encoder
		pool1 = self._block1(x)
		pool2 = self._block2(pool1)
		pool3 = self._block3(pool2)
		pool4 = self._block4(pool3)
		pool5 = self._block5(pool4)

		# Decoder
		upsample5 = self._block6(pool5)
		upsample4 = self._block7(upsample5)
		upsample3 = self._block8(upsample4)
		upsample2 = self._block9(upsample3)
		upsample1 = self._block5(upsample2)
		upsample0 = self._block6(upsample1)
		
		# Final activation
		return upsample0
