public enum YoloType
{
	Yolov5,
	Yolov5u,
	Yolov8,
	Yolov11,
	Yolov12,
}

public enum YoloSize
{
	s,
	m,
	l,
	x,
	n,
}

public enum ScalarType
{
	Float32 = 6,
	Float16 = 5,
}

public enum DeviceType
{
	CPU = 0,
	CUDA = 1,
}

public enum AttentionType
{
	SelfAttention = 0,
	MultiHeadAttention = 1,
	ScaledDotProductAttention = 2,
	FlashAttention = 3,
}

public enum TaskType
{
	Detection = 0,
	Segmentation = 1,
	Obb = 2,
	Pose = 3,
	Classification = 4,
}

public enum ImageProcessType
{
	Letterbox = 0,
	Mosiac = 1,
}

