using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Data
{
	internal class YoloDataLoader : DataLoader<Dictionary<string, torch.Tensor>, Dictionary<string, torch.Tensor>>
	{
		public YoloDataLoader(torch.utils.data.Dataset dataset, int batchSize, IEnumerable<long> shuffler, torch.Device device = null, int num_worker = 1, bool drop_last = false, bool disposeBatch = true, bool disposeDataset = true)
			: base((torch.utils.data.Dataset<Dictionary<string, torch.Tensor>>)dataset, batchSize, (Func<IEnumerable<Dictionary<string, torch.Tensor>>, torch.Device, Dictionary<string, torch.Tensor>>)Collate, shuffler, device, num_worker, drop_last, disposeBatch, disposeDataset)
		{
		}

		public YoloDataLoader(torch.utils.data.Dataset dataset, int batchSize, bool shuffle = false, torch.Device device = null, int? seed = null, int num_worker = 1, bool drop_last = false, bool disposeBatch = true, bool disposeDataset = true)
			: base((torch.utils.data.Dataset<Dictionary<string, torch.Tensor>>)dataset, batchSize, (Func<IEnumerable<Dictionary<string, torch.Tensor>>, torch.Device, Dictionary<string, torch.Tensor>>)Collate, shuffle, device, seed, num_worker, drop_last, disposeBatch, disposeDataset)
		{
		}

		private static Dictionary<string, torch.Tensor> Collate(IEnumerable<Dictionary<string, torch.Tensor>> dic, torch.Device device)
		{
			using (torch.NewDisposeScope())
			using (torch.no_grad())
			{
				Dictionary<string, torch.Tensor> dictionary = new Dictionary<string, torch.Tensor>();
				torch.Tensor[] classes = dic.Select((Dictionary<string, torch.Tensor> k) => k["cls"].unsqueeze(0)).ToArray();
				Tensor[] batch_ids = new Tensor[classes.Length];
				for (int i = 0; i < classes.Length; i++)
				{
					batch_ids[i] = torch.full(new long[] { classes[i].shape[1] }, i, classes[i].dtype, device);
				}
				dictionary["batch_idx"] = torch.cat(batch_ids, 0).unsqueeze(-1).MoveToOuterDisposeScope();

				foreach (string x in dic.First().Keys)
				{
					torch.Tensor tensor = torch.cat(dic.Select((Dictionary<string, torch.Tensor> k) => k[x]).ToArray(), 0);
					if (tensor.device_type != device.type || tensor.device_index != device.index)
					{
						tensor = tensor.to(device);
					}

					dictionary[x] = tensor.MoveToOuterDisposeScope();
				}
				return dictionary;
			}
		}
	}
}