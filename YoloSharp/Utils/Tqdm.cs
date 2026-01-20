using System.Collections;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Utils
{
	/// <summary>
	/// The Tqdm class offers utility functions to wrap collections and enumerables with a ProgressBar, 
	/// providing a simple and effective way to track and display progress in console applications for various iterative operations.
	/// </summary>
	public class Tqdm : IDisposable, IEnumerable
	{
		// Style of the bar
		public enum BarStyle
		{
			Classic,      // [====>    ] 
			Modern,       // ██████░░░░░░
			Arrow,        // ██████████▶ 
			Circle,       // ●●●○○○○○○○
			Square,       // ■■■□□□□□□□
			Block,        // ▣▣▣▣▢▢▢▢▢▢
			Filled,       // ▮▮▮▮▯▯▯▯▯▯
			Simple,       // # # # - - -
			Bold,         // ⬛⬛⬛⬜⬜⬜⬜⬜
			Star,         // ★★★★★☆☆☆☆☆
			Heart,        // ♥♥♥♡♡♡♡♡♡
			Diamond       // ♦♦♦◊◊◊◊◊◊
		}

		// Colors
		public enum BarColor
		{
			None,
			Green,
			Red,
			Yellow,
			Blue,
			Magenta,
			Cyan,
			White
		}

		private const double MIN_RATE_CALC_INTERVAL = 0.01;
		private const double RATE_SMOOTHING_FACTOR = 0.3;
		private const double MAX_SMOOTHED_RATE = 1000000;
		private const double NONINTERACTIVE_MIN_INTERVAL = 60.0;

		private BarStyle _barStyle = BarStyle.Modern;
		private BarColor _barColor = BarColor.Green;
		private int _barWidth = 25;
		private bool _showPercentage = true;
		private bool _showTime = true;
		private bool _showRate = true;
		private bool _showCounter = true;
		private bool _showBar = true;
		private bool _showBrackets = true;
		private bool _showPartialChar = true;
		private bool _useSpinner = false;

		// Text of the progress bar
		private IEnumerable _iterable;
		private string _desc = "";
		private int? _total;
		private bool _disable;
		private string _unit = "it";
		private bool _unitScale = true;
		private int _unitDivisor = 1000;
		private bool _leave = true;
		private bool _nonInteractive;
		private double _minInterval = 0.1;
		private int _initial = 0;
		private TextWriter _output;


		private int _n;
		private int _lastPrintN;
		private double _lastPrintTime;
		private double _startTime;
		private double _lastRate;
		private bool _closed;
		private bool _isBytes;
		private List<(double divisor, string label)> _scales;
		private int _spinnerIndex = 0;

		private static bool? _isNonInteractiveConsole;
		private static bool? _supportsAnsiColors;

		private static readonly Dictionary<BarStyle, (char filled, char unfilled, char partial)> _styleChars = new()
		{
			[BarStyle.Classic] = ('=', ' ', '>'),
			[BarStyle.Modern] = ('█', '░', '▓'),
			[BarStyle.Arrow] = ('█', '─', '▶'),
			[BarStyle.Circle] = ('●', '○', '◐'),
			[BarStyle.Square] = ('■', '□', '▣'),
			[BarStyle.Block] = ('▣', '▢', '▥'),
			[BarStyle.Filled] = ('▮', '▯', '▭'),
			[BarStyle.Simple] = ('#', '-', '>'),
			[BarStyle.Bold] = ('⬛', '⬜', '⬚'),
			[BarStyle.Star] = ('★', '☆', '⯪'),
			[BarStyle.Heart] = ('♥', '♡', '❥'),
			[BarStyle.Diamond] = ('♦', '◊', '⬦')
		};

		// Loop style with no total count.
		private static readonly string[] _spinners =
		{
		"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"
	};

		// Colors str for console
		private static readonly Dictionary<BarColor, (string start, string end)> _ansiColors = new()
		{
			[BarColor.Green] = ("\x1b[32m", "\x1b[0m"),
			[BarColor.Red] = ("\x1b[31m", "\x1b[0m"),
			[BarColor.Yellow] = ("\x1b[33m", "\x1b[0m"),
			[BarColor.Blue] = ("\x1b[34m", "\x1b[0m"),
			[BarColor.Magenta] = ("\x1b[35m", "\x1b[0m"),
			[BarColor.Cyan] = ("\x1b[36m", "\x1b[0m"),
			[BarColor.White] = ("\x1b[37m", "\x1b[0m")
		};

		public Tqdm(
			IEnumerable iterable = null,
			string desc = null,
			int? total = null,
			bool leave = true,
			TextWriter file = null,
			double mininterval = 0.1,
			bool? disable = null,
			string unit = "it",
			bool unitScale = true,
			int unitDivisor = 1000,
			string barFormat = null,
			int initial = 0,
			BarStyle barStyle = BarStyle.Modern,
			BarColor barColor = BarColor.Green,
			int barWidth = 25,
			bool showPercentage = true,
			bool showTime = true,
			bool showRate = true,
			bool showCounter = true,
			bool showBar = true,
			bool showBrackets = true,
			bool showPartialChar = true,
			bool useSpinner = false)
		{
			_barStyle = barStyle;
			_barColor = barColor;
			_barWidth = Math.Max(10, Math.Min(50, barWidth));
			_showPercentage = showPercentage;
			_showTime = showTime;
			_showRate = showRate;
			_showCounter = showCounter;
			_showBar = showBar;
			_showBrackets = showBrackets;
			_showPartialChar = showPartialChar;
			_useSpinner = useSpinner;

			_disable = disable ?? false;

			_iterable = iterable;
			_desc = desc ?? "";

			if (total == null && iterable != null)
			{
				if (iterable is ICollection collection)
				{
					_total = collection.Count;
				}
				else if (iterable is Array array)
				{
					_total = array.Length;
				}
				else
				{
					_total = null;
				}
			}
			else
			{
				_total = total == 0 ? null : total;
			}

			_unit = unit;
			_unitScale = unitScale;
			_unitDivisor = unitDivisor;
			_leave = leave;
			_nonInteractive = IsNonInteractiveConsole();
			_minInterval = _nonInteractive ? Math.Max(mininterval, NONINTERACTIVE_MIN_INTERVAL) : mininterval;
			_initial = initial;
			_output = file ?? Console.Out;

			_n = _initial;
			_lastPrintN = _initial;
			_lastPrintTime = GetCurrentTime();
			_startTime = GetCurrentTime();
			_lastRate = 0.0;
			_closed = false;
			_isBytes = unitScale && (unit == "B" || unit == "bytes");

			_scales = _isBytes
				? new List<(double, string)> { (1073741824, "GB/s"), (1048576, "MB/s"), (1024, "KB/s") }
				: new List<(double, string)> { (1e9, $"G{_unit}/s"), (1e6, $"M{_unit}/s"), (1e3, $"K{_unit}/s") };

			if (!_disable && _total.HasValue && !_nonInteractive)
			{
				Display();
			}
		}

		private static bool IsNonInteractiveConsole()
		{
			if (_isNonInteractiveConsole.HasValue)
				return _isNonInteractiveConsole.Value;

			var githubActions = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("GITHUB_ACTIONS"));
			var runpodId = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("RUNPOD_POD_ID"));

			_isNonInteractiveConsole = githubActions || runpodId;
			return _isNonInteractiveConsole.Value;
		}

		private static bool SupportsAnsiColors()
		{
			if (_supportsAnsiColors.HasValue)
				return _supportsAnsiColors.Value;

			if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
			{
				try
				{
					_supportsAnsiColors = Environment.OSVersion.Version.Major >= 10;
				}
				catch
				{
					_supportsAnsiColors = false;
				}
			}
			else
			{
				_supportsAnsiColors = true;
			}

			return _supportsAnsiColors.Value;
		}

		private double GetCurrentTime()
		{
			return Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;
		}

		private string FormatRate(double rate)
		{
			if (rate <= 0)
				return "";

			var invRate = 1.0 / rate;

			if (invRate > 1)
				return _isBytes ? $"{invRate:F1}s/B" : $"{invRate:F1}s/{_unit}";

			var fallback = _isBytes ? $"{rate:F1}B/s" : $"{rate:F1}{_unit}/s";
			foreach (var (divisor, label) in _scales)
			{
				if (rate >= divisor)
					return $"{(rate / divisor):F1}{label}";
			}
			return fallback;
		}

		private string FormatNum(double num)
		{
			if (!_unitScale || !_isBytes)
				return ((int)num).ToString();

			var units = new[] { "", "K", "M", "G", "T", "P" };
			var value = num;

			foreach (var unit in units)
			{
				if (Math.Abs(value) < _unitDivisor)
					return unit == "" ? $"{value:F0}B" : $"{value:F1}{unit}B";
				value /= _unitDivisor;
			}
			return $"{value:F1}EB";
		}

		private static string FormatTime(double seconds)
		{
			if (seconds < 60)
				return $"{seconds:F1}s";
			else if (seconds < 3600)
			{
				var minutes = (int)(seconds / 60);
				var secs = (int)(seconds % 60);
				return $"{minutes}:{secs:00}";
			}
			else
			{
				var hours = (int)(seconds / 3600);
				var minutes = (int)((seconds % 3600) / 60);
				var secs = (int)(seconds % 60);
				return $"{hours}:{minutes:00}:{secs:00}";
			}
		}

		private string GenerateBar()
		{
			if (!_showBar)
				return "";

			if (!_total.HasValue)
			{
				if (_useSpinner)
				{
					_spinnerIndex = (_spinnerIndex + 1) % _spinners.Length;
					return _spinners[_spinnerIndex];
				}

				// Show the bar
				var chars = _styleChars[_barStyle];
				var pulseWidth = 20; // With no total count
				var bar = new string(chars.filled, pulseWidth) + new string(chars.unfilled, 3);
				return _showBrackets ? $"[{bar}]" : bar;
			}

			var frac = Math.Min(1.0, (double)_n / _total.Value);
			var filled = (int)(frac * _barWidth);
			var remainder = (frac * _barWidth) - filled; // 

			var style = _styleChars[_barStyle];
			var barBuilder = new StringBuilder();

			for (int i = 0; i < filled; i++)
			{
				barBuilder.Append(style.filled);
			}

			if (_showPartialChar && remainder > 0.3 && filled < _barWidth)
			{
				barBuilder.Append(style.partial);
				for (int i = filled + 1; i < _barWidth; i++)
				{
					barBuilder.Append(style.unfilled);
				}
			}
			else
			{
				for (int i = filled; i < _barWidth; i++)
				{
					barBuilder.Append(style.unfilled);
				}
			}

			var barStr = barBuilder.ToString();
			return _showBrackets ? $"[{barStr}]" : barStr;
		}

		private (string start, string end) GetColorCode()
		{
			if (_barColor == BarColor.None || !SupportsAnsiColors())
				return ("", "");

			return _ansiColors.TryGetValue(_barColor, out var codes) ? codes : ("", "");
		}

		private bool ShouldUpdate(double dt, int dn)
		{
			if (_nonInteractive)
				return false;

			return (_total.HasValue && _n >= _total.Value) || (dt >= _minInterval);
		}

		private void Display(bool final = false)
		{
			if (_disable || (_closed && !final))
				return;

			var currentTime = GetCurrentTime();
			var dt = currentTime - _lastPrintTime;
			var dn = _n - _lastPrintN;

			if (!final && !ShouldUpdate(dt, dn))
				return;

			double rate;
			if (dt > MIN_RATE_CALC_INTERVAL)
			{
				rate = dn / dt;
				if (rate < MAX_SMOOTHED_RATE)
				{
					_lastRate = RATE_SMOOTHING_FACTOR * rate + (1 - RATE_SMOOTHING_FACTOR) * _lastRate;
					rate = _lastRate;
				}
			}
			else
			{
				rate = _lastRate;
			}

			if (_total.HasValue && _n >= _total.Value)
			{
				var overallElapsed = currentTime - _startTime;
				if (overallElapsed > 0)
					rate = _n / overallElapsed;
			}

			_lastPrintN = _n;
			_lastPrintTime = currentTime;
			var elapsed = currentTime - _startTime;

			// Remaining time.
			var remainingStr = "";
			if (_total.HasValue && _n > 0 && _n < _total.Value && elapsed > 0)
			{
				var estRate = rate > 0 ? rate : (_n / elapsed);
				var remaining = (_total.Value - _n) / estRate;
				remainingStr = $"<{FormatTime(remaining)}";
			}

			// Percent
			var progress = _total.HasValue ? (double)_n / _total.Value : 0;
			var color = GetColorCode();

			string nStr = "", tStr = "", percentStr = "";

			if (_showCounter)
			{
				nStr = FormatNum(_n);
				tStr = _total.HasValue ? FormatNum(_total.Value) : "?";
			}

			if (_showPercentage && _total.HasValue)
			{
				percentStr = $"{progress * 100,4:F1}%";
			}

			var bar = GenerateBar();
			var elapsedStr = _showTime ? FormatTime(elapsed) : "";
			var rateStr = _showRate ? FormatRate(rate) : "";

			var sb = new StringBuilder();

			// Add text
			if (!string.IsNullOrEmpty(_desc))
			{
				sb.Append(_desc);
				sb.Append(" ");
				//sb.Append(": ");
			}

			// Add percent
			if (!string.IsNullOrEmpty(percentStr))
			{
				sb.Append($"{percentStr} ");
			}

			// Add bar with color
			if (!string.IsNullOrEmpty(bar))
			{
				sb.Append(color.start);
				sb.Append(bar);
				sb.Append(color.end);
				sb.Append(" ");
			}

			// Add counter
			if (_showCounter)
			{
				if (_total.HasValue)
				{
					if (_isBytes && _n >= _total.Value)
					{
						sb.Append($"{tStr} ");
					}
					else
					{
						sb.Append($"{nStr}/{tStr} ");
					}
				}
				else
				{
					sb.Append($"{nStr} ");
				}
			}

			// Add rate
			if (!string.IsNullOrEmpty(rateStr))
				sb.Append($"{rateStr} ");

			// Add time
			if (!string.IsNullOrEmpty(elapsedStr))
			{
				sb.Append(elapsedStr);

				// Add remaing time
				if (!string.IsNullOrEmpty(remainingStr))
					sb.Append(remainingStr);
			}

			var progressStr = sb.ToString().TrimEnd();

			// Add output text
			try
			{
				if (_nonInteractive)
				{
					_output.Write(progressStr);
					if (final && _leave)
						_output.WriteLine();
				}
				else
				{
					_output.Write($"\r{progressStr}");

					// clear the temp string
					try
					{
						var clearWidth = Console.WindowWidth - progressStr.Length - 1;
						if (clearWidth > 0)
							_output.Write(new string(' ', clearWidth));
					}
					catch { }

					_output.Write("\r");
					_output.Flush();
				}
			}
			catch
			{

			}
		}

		public void Update(int n = 1)
		{
			if (!_disable && !_closed)
			{
				_n += n;
				Display();
			}
		}

		public void SetDescription(string desc = null)
		{
			_desc = desc ?? "";
			if (!_disable)
			{
				Display();
			}
		}

		public void SetPostfix(params (string key, object value)[] items)
		{
			if (items == null || items.Length == 0)
				return;

			var postfix = string.Join(", ", items.Select(x => $"{x.key}={x.value}"));
			var baseDesc = _desc;
			var sepIndex = _desc.IndexOf(" | ");
			if (sepIndex >= 0)
			{
				baseDesc = _desc.Substring(0, sepIndex);
			}

			SetDescription($"{baseDesc} | {postfix}");
		}

		public void Close()
		{
			if (_closed)
				return;

			_closed = true;

			if (!_disable)
			{
				// Last display
				if (_total.HasValue && _n >= _total.Value)
				{
					_n = _total.Value;
					if (_n != _lastPrintN)
					{
						Display(true);
					}
				}
				else
				{
					Display(true);
				}

				// Clean the cache.
				if (_leave)
				{
					_output.WriteLine();
				}

				try
				{
					_output.Flush();
				}
				catch { }
			}
		}

		public void Refresh() => Display();

		public void Clear()
		{
			try
			{
				_output.Write("\r" + new string(' ', Console.WindowWidth - 1) + "\r");
				_output.Flush();
			}
			catch { }
		}

		public static void Write(string s, TextWriter file = null, string end = "\n")
		{
			file ??= Console.Out;
			try
			{
				file.Write(s + end);
				file.Flush();
			}
			catch { }
		}

		public IEnumerator GetEnumerator()
		{
			if (_iterable == null)
				throw new InvalidOperationException("'Null' object is not iterable");

			try
			{
				foreach (var item in _iterable)
				{
					yield return item;
					Update(1);
				}
			}
			finally
			{
				Close();
			}
		}

		public void Dispose() => Close();
	}

	public class Tqdm<T> : Tqdm, IEnumerable<T>
	{
		private IEnumerable<T> _typedIterable;

		public Tqdm(
			IEnumerable<T> iterable = null,
			string desc = null,
			int? total = null,
			bool leave = true,
			TextWriter file = null,
			double mininterval = 0.1,
			bool? disable = null,
			string unit = "it",
			bool unitScale = true,
			int unitDivisor = 1000,
			string barFormat = null,
			int initial = 0,
			BarStyle barStyle = BarStyle.Modern,
			BarColor barColor = BarColor.Green,
			int barWidth = 25,
			bool showPercentage = true,
			bool showTime = true,
			bool showRate = true,
			bool showCounter = true,
			bool showBar = true,
			bool showBrackets = true,
			bool showPartialChar = true,
			bool useSpinner = false)
			: base(iterable, desc, total, leave, file, mininterval, disable, unit,
				  unitScale, unitDivisor, barFormat, initial, barStyle, barColor,
				  barWidth, showPercentage, showTime, showRate, showCounter,
				  showBar, showBrackets, showPartialChar, useSpinner)
		{
			_typedIterable = iterable;
		}

		public new IEnumerator<T> GetEnumerator()
		{
			if (_typedIterable == null)
				throw new InvalidOperationException("'Null' object is not iterable");

			foreach (var item in _typedIterable)
			{
				yield return item;
				Update(1);
			}
			Close();
		}


	}


}

