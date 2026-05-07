namespace Data
{
    internal class Utils
    {
        /// <summary>
        /// Convert image paths to label paths by replacing 'images' with 'labels' and extension with '.txt'.
        /// </summary>
        /// <param name="img_paths"></param>
        /// <returns></returns>
        internal static string[] img2label_paths(string[] img_paths)
        {
            string sep = Path.DirectorySeparatorChar.ToString();
            string sa = $"{sep}images{sep}";
            string sb = $"{sep}labels{sep}";

            return img_paths.Select(imgPath =>
            {
                int lastIndex = imgPath.LastIndexOf(sa, StringComparison.Ordinal);
                if (lastIndex == -1)
                    throw new ArgumentException($"Can't find '{sa}'：{imgPath}");

                string before = imgPath.Substring(0, lastIndex);
                string after = imgPath.Substring(lastIndex + sa.Length);
                string labelPath = before + sb + after;

                labelPath = Path.ChangeExtension(labelPath, "txt");
                return labelPath;
            }).ToArray();
        }

    }
}
