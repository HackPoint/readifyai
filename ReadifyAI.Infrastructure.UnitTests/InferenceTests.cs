using Readify.Infrastructure.AI.Inferences;
using Readify.Infrastructure.AI.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace ReadifyAI.Infrastructure.Tests;

public class InferenceTests {
    public InferenceTests() {
        // Initialize TorchSharp for the CPU backend
        torch.InitializeDeviceType(DeviceType.CPU);
    }

    /// <summary>
    /// Tests the Predict method with a valid input tensor.
    /// </summary>
    [Fact]
    public void Predict_ValidInput_ShouldReturnValidOutput() {
        // Arrange
        var fakeSimilarityModel = new ContentSimilarityModel();
        var inputTensor = tensor([1.0f, 2.0f, 3.0f], dtype: ScalarType.Int64).unsqueeze(0);
        var inference = new Inference(fakeSimilarityModel);

        // Act
        var output = inference.Predict(inputTensor);

        // Assert
        Assert.InRange(output, 0.0f, 1.0f); 
    }

    [Fact]
    public void Compare_ShouldReturnCosineSimilarity() {
        // Arrange
        var fakeSimilarityModel = new ContentSimilarityModel();

        // Input tensors
        var input1 = tensor(new long[] { 1, 2, 3 }, dtype: ScalarType.Int64).unsqueeze(0);
        var input2 = tensor(new long[] { 4, 5, 6 }, dtype: ScalarType.Int64).unsqueeze(0);

        // Instantiate the Inference class with the fake similarity model
        var inference = new Inference(fakeSimilarityModel);

        // Act
        var result = inference.Compare(input1, input2);

        // Calculate expected similarity manually
        float expectedSimilarity;
        using (no_grad()) {
            var embedding1 = fakeSimilarityModel.Forward(input1);
            var embedding2 = fakeSimilarityModel.Forward(input2);

            // Calculate cosine similarity
            var similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim: 0);
            expectedSimilarity = similarity.item<float>();
        }

        // Assert
        Assert.Equal(expectedSimilarity, result, precision: 5); // Allow small rounding differences
    }
}