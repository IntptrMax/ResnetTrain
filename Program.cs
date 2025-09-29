using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ResnetTrain
{
	internal class Program
	{
		private static int CropSize = 224;
		private static int ResizeSize = 256;
		private static int Loop = 100;
		private static int BatchSize = 64;
		private static int Workers = 8;
		private static int ClassNumber = 5;

		private static Device device = new Device(DeviceType.CUDA);
		private static ScalarType scalarType = ScalarType.Float32;

		private static string modelName = "resnet18.bin";
		private static string path = @"..\..\..\Assets\flower_photos";
		private static string trainFileList = @"..\..\..\Assets\labels\train.txt";
		private static string valFileList = @"..\..\..\Assets\labels\val.txt";
		private static string testFileList = @"..\..\..\Assets\labels\test.txt";
		private static CrossEntropyLoss cross_loss = new CrossEntropyLoss();

		static void Train()
		{
			var resnet = torchvision.models.resnet18(ClassNumber).to(device, scalarType);
			var optimizer = torch.optim.SGD(resnet.parameters(), learningRate: 0.02, momentum: 0.9, weight_decay: 5e-4);
			var lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max: 200);
			DataClass trainDataClass = new DataClass(path, trainFileList, ResizeSize, CropSize);
			DataLoader trainLoader = new DataLoader(trainDataClass, batchSize: BatchSize, shuffle: true, device: device, num_worker: Workers);

			DataClass valDataClass = new DataClass(path, valFileList, ResizeSize, CropSize, false);
			DataLoader valLoader = new DataLoader(valDataClass, batchSize: 8, shuffle: true, device: device, num_worker: Workers);

			for (int loop = 0; loop < Loop; loop++)
			{
				int step = 0;
				float lossValue = 0;
				resnet.train();
				foreach (var trainBatch in trainLoader)
				{
					step++;
					Tensor trainImg = trainBatch["img"];
					Tensor trainTag = trainBatch["tag"];
					optimizer.zero_grad();
					Tensor output = resnet.forward(trainImg);
					Tensor loss = cross_loss.forward(output, trainTag);
					loss.backward();
					//optimizer.step();
					optimizer.step();
					float ls = loss.ToSingle();
					lossValue += ls;
					GC.Collect();
					Console.WriteLine("Loop:{0}   Setp:{1} / {2}  ; Loss:{3}", loop, step, trainLoader.Count, ls / trainTag.size(0));
				}
				lossValue /= trainDataClass.Count;
				lr_scheduler.step();
				Console.WriteLine($"\nTrain set: Average loss: {lossValue.ToString("f6")}\n");

				resnet.eval();

				float val_loss = 0;
				int correct = 0;

				using (no_grad())
				{
					foreach (var valBatch in valLoader)
					{
						Tensor valImg = valBatch["img"];
						Tensor valTag = valBatch["tag"];
						Tensor output = resnet.forward(valImg);
						Tensor loss = cross_loss.forward(output, valTag);
						Tensor pred = output.argmax(dim: 1, keepdim: true);
						val_loss += loss.ToSingle();
						correct += pred.eq(valTag.view_as(pred)).sum().ToInt32();
					}
				}

				val_loss /= valDataClass.Count;

				Console.WriteLine($"\nVal set: Average loss: {val_loss.ToString("f6")}, Accuracy: {correct}/{valDataClass.Count} ({(100f * correct / valDataClass.Count).ToString("f2")}%)\n");
			}
			resnet.save(modelName);
			Console.WriteLine("Train Done.");
		}

		static void Test()
		{
			var resnet = torchvision.models.resnet18(5, device: device).to(scalarType);
			resnet.load(modelName);
			resnet.eval();

			DataClass testDataClass = new DataClass(path, testFileList, ResizeSize, CropSize, false);
			DataLoader testLoader = new DataLoader(testDataClass, batchSize: 4, device: device, num_worker: Workers);

			int correct = 0;
			float test_loss = 0;
			using (no_grad())
			{
				foreach (var testBatch in testLoader)
				{
					Tensor testImg = testBatch["img"];
					Tensor testTag = testBatch["tag"];
					Tensor output = resnet.forward(testImg);
					Tensor pred = output.argmax(dim: 1, keepdim: true);
					Tensor loss = cross_loss.forward(output, testTag);
					correct += pred.eq(testTag.view_as(pred)).sum().ToInt32();
					test_loss += loss.ToSingle();
				}
				test_loss /= testDataClass.Count;
			}

			Console.WriteLine($"\nTest set: Average loss: {test_loss.ToString("f6")}, Accuracy: {correct}/{testDataClass.Count} ({(100f * correct / testDataClass.Count).ToString("f2")}%)\n");
		}


		static void Main(string[] args)
		{
			Train();
			Test();
		}

	}
}
