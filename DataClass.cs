using TorchSharp;
using static TorchSharp.torch;

namespace ResnetTrain
{
	internal class DataClass : torch.utils.data.Dataset
	{
		private string path = string.Empty;
		long count = 0;
		List<string> foldersName = new List<string>();
		List<string> files = new List<string>();
		int cropWidth = 224;
		int cropHeight = 224;
		int resizeWidth = 256;
		int resizeHeight = 256;
		double[] means = [0.485, 0.456, 0.406], stdevs = [0.229, 0.224, 0.225];

		public DataClass(string path, int resizeWidth = 256, int resizeHeight = 256, int cropWidth = 224, int cropHeight = 224)
		{
			this.path = path;
			string[] files = Directory.GetFiles(path, "*.*", SearchOption.AllDirectories).Where(a => 
			{
				string extension = Path.GetExtension(a).ToLower();
				return (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp");
			}).ToArray();

			this.files = new List<string>(files);
			this.count = files.Length;
			DirectoryInfo[] folders = Directory.GetParent(files[0]).Parent.GetDirectories();
			foreach (var folder in folders)
			{
				foldersName.Add(folder.Name);
			}
			this.cropWidth = cropWidth;
			this.cropHeight = cropHeight;
			this.resizeWidth = resizeWidth;
			this.resizeHeight = resizeHeight;
		}

		public override long Count => count;

		public string GetTagName(long index)
		{
			string file = files[(int)index];
			string parent = Directory.GetParent(file).Name;
			return parent;
		}

		public string GetTagNameByTag(int tagIndex)
		{
			return foldersName[tagIndex];
		}

		public override Dictionary<string, torch.Tensor> GetTensor(long index)
		{
			string file = files[(int)index];
			string parent = Directory.GetParent(file).Name;
			int tagIndex = foldersName.IndexOf(parent);

			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			var transformers = torchvision.transforms.Compose([
				torchvision.transforms.Resize(resizeHeight,resizeWidth),
				torchvision.transforms.CenterCrop(cropHeight,cropWidth),
				torchvision.transforms.Normalize(means, stdevs)]);

			Tensor imgTensor = torchvision.io.read_image(file) / 255.0f;
			imgTensor = transformers.call(imgTensor.unsqueeze(0));
			var labelTensor = torch.tensor(tagIndex, dtype: torch.int64);
			var tensorDataDic = new Dictionary<string, Tensor>();
			tensorDataDic.Add("tag", labelTensor);
			tensorDataDic.Add("img", imgTensor.squeeze(0));
			return tensorDataDic;

		}


		private Tensor Letterbox(Tensor image, int targetWidth, int targetHeight)
		{
			int originalWidth = (int)image.shape[2];
			int originalHeight = (int)image.shape[1];

			float scale = Math.Min((float)targetWidth / originalWidth, (float)targetHeight / originalHeight);

			int scaledWidth = (int)(originalWidth * scale);
			int scaledHeight = (int)(originalHeight * scale);

			int padLeft = (targetWidth - scaledWidth) / 2;
			int padTop = (targetHeight - scaledHeight) / 2;

			// 缩放图像
			Tensor scaledImage = torchvision.transforms.functional.resize(image, scaledHeight, scaledWidth);

			Tensor paddedImage = zeros(new long[] { 3, targetHeight, targetWidth }, image.dtype, image.device);
			paddedImage[TensorIndex.Ellipsis, padTop..(padTop + scaledHeight), padLeft..(padLeft + scaledWidth)].copy_(scaledImage);

			GC.Collect();

			return paddedImage;
		}
	}
}
