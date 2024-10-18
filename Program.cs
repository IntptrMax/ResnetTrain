using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ResnetTrain
{
	internal class Program
	{
		private static int CropWidth = 224;
		private static int CropHeight = 224;
		private static int ResizeWidth = 256;
		private static int ResizeHeight = 256;
		private static int Loop = 10;
		private static int BatchSize = 128;
		private static int Workers = 8;

		private static Device device = new Device(DeviceType.CUDA);
		private static ScalarType scalarType = ScalarType.Float32;

		private static string modelName = "resnet18.bin";
		private static string path = @"..\..\..\Assets\flower_photos";

		static void Train()
		{
			var resnet = torchvision.models.resnet18(5).to(device, scalarType);
			var optimizer = torch.optim.Adam(resnet.parameters());
			var cross_loss = new CrossEntropyLoss();
			DataClass dataClass = new DataClass(path, ResizeWidth, ResizeHeight, CropWidth, CropHeight);
			var trainLoader = new DataLoader(dataClass, batchSize: BatchSize, shuffle: true, device: device, num_worker: Workers);

			for (int loop = 0; loop < Loop; loop++)
			{
				int step = 0;
				foreach (var trainBatch in trainLoader)
				{
					step++;
					var img = trainBatch["img"];
					var tag = trainBatch["tag"];
					optimizer.zero_grad();
					Tensor re = resnet.forward(img);
					var loss = cross_loss.forward(re, tag);
					loss.backward();
					optimizer.step();
					var eval = loss.ToSingle();
					GC.Collect();
					Console.WriteLine("Loop:{0}   Setp:{1} / {2}  ; Loss:{3}", loop, step, trainLoader.Count, eval);
				}

			}
			resnet.save(modelName);
			Console.WriteLine("Train Done.");
		}

		static void Test()
		{
			string testPath = path;
			DataClass dataClass = new DataClass(testPath);
			var data = dataClass.GetTensor(0);
			var img = data["img"].unsqueeze(0).to(scalarType, device);
			var resnet = torchvision.models.resnet18(5,device: device).to(scalarType);
			resnet.load(modelName);
			resnet.eval();
			Tensor re = (Tensor)resnet.forward(img);
			re = torch.softmax(re, 1);
			var (max, index) = re.max(1);
			Console.WriteLine("Predction:{0}\r\nScore:{1}\r\nTag:{2}", dataClass.GetTagNameByTag((int)index.ToInt64()), max.ToSingle(), dataClass.GetTagName(0));

		}


		static void Main(string[] args)
		{
			Train();
			Test();
		}

	}
}
