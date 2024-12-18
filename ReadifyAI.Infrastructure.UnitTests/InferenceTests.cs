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

internal class FakeContentSimilarityModel : ContentSimilarityModel {
    public override Tensor Forward(torch.Tensor input) {
        // Ensure the input tensor is in the correct type
        if (input.dtype != ScalarType.Int64) {
            input = input.to(ScalarType.Int64);
        }

        // Predefined inputs for comparison
        var input1 = tensor(new long[] { 1, 2, 3 }, dtype: ScalarType.Int64).unsqueeze(0); // Shape [1, 3]
        var input2 = tensor(new long[] { 4, 5, 6 }, dtype: ScalarType.Int64).unsqueeze(0); // Shape [1, 3]

        // Compare tensors using .all().to_bool_scalar() safely
        if (input.equal(input1).all().ToBoolean()) {
            return tensor([0.1f, 0.2f, 0.3f], dtype: ScalarType.Float32).unsqueeze(0); // Shape [1, 3]
        }
        else if (input.equal(input2).all().ToBoolean()) {
            return tensor([0.4f, 0.5f, 0.6f], dtype: ScalarType.Float32).unsqueeze(0); // Shape [1, 3]
        }

        throw new InvalidOperationException("Unexpected input tensor");
    }
}