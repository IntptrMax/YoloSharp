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

