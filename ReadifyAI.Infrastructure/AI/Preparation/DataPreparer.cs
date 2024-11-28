using TorchSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;


namespace Readify.Infrastructure.AI.Preparation;

public class DataPreparer {
    private const int DefaultMaxLength = 256;

    public torch.Tensor TokenizeText(string content, int maxLength = DefaultMaxLength) {
        if (string.IsNullOrWhiteSpace(content)) {
            throw new ArgumentException("Content cannot be null or empty.", nameof(content));
        }

        var tokens = content
            .Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(word => (long)word.GetHashCode() % 10000) // Simple hashing for token IDs
            .Take(maxLength)
            .ToArray();

        return torch.tensor(tokens, torch.ScalarType.Int64).unsqueeze(0); // Add batch dimension
    }

    /// <summary>
    /// Prepares a collection of images as tensors for AI processing.
    /// </summary>
    /// <param name="imagePaths">The paths to the image files.</param>
    /// <returns>A list of tensors representing the preprocessed images.</returns>
    public List<torch.Tensor> PrepareImages(IEnumerable<string> imagePaths) {
        if (imagePaths == null || !imagePaths.Any()) {
            throw new ArgumentException("Image paths cannot be null or empty.", nameof(imagePaths));
        }

        var imageTensors = new List<torch.Tensor>();

        foreach (var path in imagePaths) {
            var tensor = PreprocessImage(File.ReadAllBytes(path));
            imageTensors.Add(tensor);
        }

        return imageTensors;
    }

    /// <summary>
    /// Preprocesses a single image and converts it into a normalized tensor.
    /// </summary>
    /// <param name="imageData">The image data as a byte array.</param>
    /// <returns>A tensor representing the preprocessed image.</returns>
    private torch.Tensor PreprocessImage(byte[] imageData) {
        // Load image using ImageSharp
        using var image = Image.Load<Rgb24>(imageData);

        // Resize image to [224x224] (example for AI models like ResNet)
        const int targetWidth = 224;
        const int targetHeight = 224;
        image.Mutate(x => x.Resize(targetWidth, targetHeight));

        // Create a tensor from image pixels
        var tensor = torch.zeros(new long[] { 3, targetHeight, targetWidth }, torch.ScalarType.Float32); // RGB channels

        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                var pixel = image[x, y];
                tensor[0, y, x] = pixel.R / 255.0f; // Red channel
                tensor[1, y, x] = pixel.G / 255.0f; // Green channel
                tensor[2, y, x] = pixel.B / 255.0f; // Blue channel
            }
        }

        return tensor;
    }
}