using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;

namespace Data
{
	internal class Augment
	{
		internal class LetterBox
		{
			private readonly int width;
			private readonly int height;
			private readonly bool auto;
			private readonly bool scale_fill;
			private readonly bool scaleup;
			private readonly int stride;
			private readonly bool center;

			/// <summary>
			/// Initialize LetterBox object for resizing and padding images.<para/>
			/// This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation 
			/// tasks.It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.
			/// </summary>
			/// <param name="width">Target width for the resized image.</param>
			/// <param name="height">Target height for the resized image.</param>
			/// <param name="auto">If True, use minimum rectangle to resize. If False, use new_shape directly.</param>
			/// <param name="scale_fill">If True, stretch the image to new_shape without padding.</param>
			/// <param name="scaleup">If True, allow scaling up. If False, only scale down.</param>
			/// <param name="center">If True, center the placed image. If False, place image in top-left corner.</param>
			/// <param name="stride">Stride value for ensuring image size is divisible by stride.</param>
			public LetterBox(int width = 640, int height = 640, bool auto = false, bool scale_fill = false, bool scaleup = true, bool center = true, int stride = 32)
			{
				this.width = width;
				this.height = height;
				this.auto = auto;
				this.scale_fill = scale_fill;
				this.scaleup = scaleup;
				this.stride = stride;
				this.center = center; // Put the image in the middle or top-left
			}


		}


	}
}
