using TorchSharp;
using static TorchSharp.torch;

namespace ResnetTrain
{
	internal class DataClass : torch.utils.data.Dataset
	{
		long count = 0;
		List<string> foldersName = new List<string>();
		List<string> files = new List<string>();
		int cropSize = 224;
		int resizeSize = 256;
		double[] means = [0.485, 0.456, 0.406], stdevs = [0.229, 0.224, 0.225];
		bool useTransform_plus = true;

		public DataClass(string root, int resizeSize = 256, int cropSize = 224, bool useTransform_plus = true)
		{
			string[] files = Directory.GetFiles(root, "*.*", SearchOption.AllDirectories).Where(a =>
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
			this.cropSize = cropSize;
			this.resizeSize = resizeSize;
		}

		public DataClass(string root, string fileList, int resizeSize = 256, int cropSize = 224, bool useTransform_plus = true)
		{
			string[] files = File.ReadAllLines(fileList);
			files = files.Select(a => Path.Combine(root, a)).ToArray();

			this.files = new List<string>(files);
			this.count = files.Length;
			DirectoryInfo[] folders = Directory.GetParent(files[0]).Parent.GetDirectories();
			foreach (var folder in folders)
			{
				foldersName.Add(folder.Name);
			}
			this.cropSize = cropSize;
			this.resizeSize = resizeSize;
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

			torchvision.ITransform[] transforms = new torchvision.ITransform[] {
				torchvision.transforms.Resize(resizeSize),
				torchvision.transforms.CenterCrop(cropSize),
				torchvision.transforms.Normalize(means, stdevs)};

			torchvision.ITransform[] transforms_plus = new torchvision.ITransform[] {
				torchvision.transforms.RandomHorizontalFlip(p:0.5),
				torchvision.transforms.RandomRotation(15),
				torchvision.transforms.ColorJitter(brightness:0.4f, contrast:0.4f, saturation:0.4f),
				torchvision.transforms.Resize(resizeSize),
				torchvision.transforms.CenterCrop(cropSize),
				torchvision.transforms.Normalize(means, stdevs)};

			torchvision.ITransform transformers = torchvision.transforms.Compose(this.useTransform_plus ? transforms_plus : transforms);

			Tensor imgTensor = torchvision.io.read_image(file) / 255.0f;
			imgTensor = transformers.call(imgTensor.unsqueeze(0));
			Tensor labelTensor = torch.tensor(tagIndex, dtype: torch.int64);

			return new Dictionary<string, Tensor>
			{
				{ "img", imgTensor.squeeze(0) },
				{ "tag", labelTensor }
			};

		}

	}
}
