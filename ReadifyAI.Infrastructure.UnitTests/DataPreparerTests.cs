using Readify.Infrastructure.AI;
using Readify.Infrastructure.AI.Inferences;
using Readify.Infrastructure.AI.Models;
using Readify.Infrastructure.AI.Preparation;
using Readify.Infrastructure.AI.Trainers;
using SixLabors.ImageSharp;
using TorchSharp;

namespace ReadifyAI.Infrastructure.Tests;

public class DataPreparerTests {
    private readonly DataPreparer _dataPreparer;

    public DataPreparerTests() {
        torch.InitializeDeviceType(DeviceType.CPU);
        _dataPreparer = new DataPreparer();
    }
    
    [Fact]
    public void TokenizeText_ValidInput_ShouldReturnTensor()
    {
        // Arrange
        string content = "This is a sample sentence for testing.";

        // Act
        var tensor = _dataPreparer.TokenizeText(content);

        // Assert
        Assert.NotNull(tensor);
        Assert.Equal(1, tensor.shape[0]); // Batch dimension
        Assert.True(tensor.shape[1] <= 256); // Default max length
    }
    
    [Fact]
    public void TokenizeText_EmptyInput_ShouldThrowArgumentException()
    {
        // Arrange
        string content = "";

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _dataPreparer.TokenizeText(content));
    }

    [Fact]
    public void TokenizeText_NullInput_ShouldThrowArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _dataPreparer.TokenizeText(null!));
    }
    
    [Fact]
    public void PrepareImages_EmptyInput_ShouldThrowArgumentException()
    {
        // Arrange
        var imagePaths = new List<string>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _dataPreparer.PrepareImages(imagePaths));
    }
    
    [Fact]
    public void PrepareImages_InvalidPaths_ShouldThrowFileNotFoundException()
    {
        // Arrange
        var imagePaths = new List<string> { "invalid_image1.jpg", "invalid_image2.jpg" };

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => _dataPreparer.PrepareImages(imagePaths));
    }
    
    [Fact]
    public void PrepareImages_ValidInput_ShouldReturnTensors()
    {
        // Arrange
        string testImagePath1 = "test_image1.jpg";
        string testImagePath2 = "test_image2.jpg";

        // Create dummy image files for testing
        CreateTestImage(testImagePath1);
        CreateTestImage(testImagePath2);

        var imagePaths = new List<string> { testImagePath1, testImagePath2 };

        try
        {
            // Act
            var imageTensors = _dataPreparer.PrepareImages(imagePaths);

            // Assert
            Assert.NotNull(imageTensors);
            Assert.Equal(2, imageTensors.Count); // Two tensors for two images

            foreach (var tensor in imageTensors)
            {
                Assert.Equal(new long[] { 3, 224, 224 }, tensor.shape); // RGB channels with 224x224 dimensions
            }
        }
        finally
        {
            // Cleanup test images
            File.Delete(testImagePath1);
            File.Delete(testImagePath2);
        }
    }
    
    [Fact]
    public void PrepareImages_CorruptedImage_ShouldThrowException()
    {
        // Arrange
        string corruptedImagePath = "corrupted_image.jpg";
        File.WriteAllText(corruptedImagePath, "This is not a valid image file.");

        try
        {
            var imagePaths = new List<string> { corruptedImagePath };

            // Act & Assert
            Assert.Throws<UnknownImageFormatException>(() => _dataPreparer.PrepareImages(imagePaths));
        }
        finally
        {
            // Cleanup
            File.Delete(corruptedImagePath);
        }
    }
    
    
    [Fact]
    public void AIComparison_ShouldReturnSimilarityScore()
    {
        var model = new ContentSimilarityModel();
        var trainer = new ContentSimilarityTrainer();
        var inference = new Inference(model);

        var preparer = new DataPreparer();
        var epubTensor = preparer.TokenizeText("Sample EPUB text.");
        var pdfTensor = preparer.TokenizeText("Sample PDF text.");

        trainer.TrainModel(model, epubTensor, epubTensor);
        var similarity = inference.Predict(pdfTensor);

        Assert.InRange(similarity, 0, 1);
    }
    
    
    /// <summary>
    /// Helper method to create a dummy image for testing.
    /// </summary>
    private void CreateTestImage(string path)
    {
        using var image = Image.LoadPixelData<SixLabors.ImageSharp.PixelFormats.Rgb24>(
            new byte[224 * 224 * 3], 224, 224);
        image.SaveAsJpeg(path);
    }
}