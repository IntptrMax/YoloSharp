namespace YoloSharp.Types
{
	public class YoloResult
	{
		public int ClassID;
		public float Score;
		public int CenterX;
		public int CenterY;
		public int Width;
		public int Height;
		public float Radian;
		public byte[,] Mask;
		public KeyPoint[] KeyPoints;

		public int X => CenterX - Width / 2;
		public int Y => CenterY - Height / 2;
	}

}
