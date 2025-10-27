using OpenCvSharp;

namespace YoloSharp.Data
{
	internal class ImageData
	{
		public string ImagePath;
		public int OrgWidth;
		public int OrgHeight;

		public Mat OrgImage => Cv2.ImRead(ImagePath);
		public Mat ResizedImage;
		public List<LabelData> OrgLabels;
		public List<LabelData> ResizedLabels;
	}

	internal class LabelData
	{
		public float CenterX;
		public float CenterY;
		public float Width;
		public float Height;
		public float Radian;
		public int LabelID;
		public Point[] MaskOutLine;
		public Types.KeyPoint[] KeyPoints;

	
	}
}
