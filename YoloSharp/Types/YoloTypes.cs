namespace YoloSharp.Types
{
	/// <summary>
	/// Represents the different versions of the YOLO (You Only Look Once) object detection algorithm.
	/// </summary>
	/// <remarks>YOLO is a family of real-time object detection models. This enumeration provides identifiers for
	/// specific versions of the YOLO algorithm, which may differ in architecture, performance, and use cases.</remarks>
	public enum YoloType
	{
		Yolov5,
		Yolov5u,
		Yolov8,
		Yolov11,
		Yolov12,
	}

	/// <summary>
	/// Represents the available sizes for a YOLO (You Only Look Once) model configuration.
	/// </summary>
	/// <remarks>The sizes correspond to different model configurations, typically used to balance performance and
	/// accuracy. Smaller sizes (e.g., <see cref="s"/> or <see cref="n"/>)  are optimized for speed and lower resource
	/// usage, while larger sizes (e.g., <see cref="l"/>  or <see cref="x"/>) are designed for higher accuracy at the cost
	/// of increased computational  requirements.</remarks>
	public enum YoloSize
	{
		s,
		m,
		l,
		x,
		n,
	}

	/// <summary>
	/// Represents the scalar data types supported by the system.
	/// </summary>
	/// <remarks>This enumeration defines scalar types that can be used to represent numeric values with varying
	/// levels of precision.</remarks>
	public enum ScalarType
	{
		Float32 = 6,
		Float16 = 5,
	}

	/// <summary>
	/// Represents the type of device used for computation.
	/// </summary>
	/// <remarks>This enumeration is typically used to specify the hardware device on which a computation or operation
	/// will be performed.</remarks>
	public enum DeviceType
	{
		CPU = 0,
		CUDA = 1,
	}

	/// <summary>
	/// Specifies the types of attention mechanisms used in machine learning models.
	/// </summary>
	/// <remarks>Attention mechanisms are commonly used in neural networks, particularly in natural language
	/// processing (NLP) and computer vision tasks,  to focus on relevant parts of the input data. This enumeration provides
	/// options for different attention strategies.</remarks>
	public enum AttentionType
	{
		SelfAttention = 0,
		MultiHeadAttention = 1,
		ScaledDotProductAttention = 2,
		FlashAttention = 3,
	}

	/// <summary>
	/// Specifies the type of task to be performed in a machine learning or computer vision context.
	/// </summary>
	/// <remarks>This enumeration defines various task types that can be used to categorize or configure operations.
	/// Examples include object detection, image segmentation, and classification.</remarks>
	public enum TaskType
	{
		Detection = 0,
		Segmentation = 1,
		Obb = 2,
		Pose = 3,
		Classification = 4,
	}

	/// <summary>
	/// Specifies the type of image processing to be applied.
	/// </summary>
	/// <remarks>This enumeration defines the available image processing techniques that can be used to modify or transform an image.</remarks>
	public enum ImageProcessType
	{
		Letterbox = 0,
		Mosiac = 1,
	}

}