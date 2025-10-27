namespace YoloSharp.Types
{
	public class KeyPoint
	{
		public float X;
		public float Y;
		public float VisibilityScore;

		public KeyPoint()
		{

		}

		public KeyPoint(float x, float y, float visibilityScore)
		{
			X = x;
			Y = y;
			VisibilityScore = visibilityScore;
		}
	}

}
