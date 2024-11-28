using Readify.Infrastructure.AI.Inferences;
using Readify.Infrastructure.AI.Models;
using TorchSharp;

namespace ReadifyAI.Infrastructure.Tests;

public class InferenceTests {
    private readonly ContentSimilarityModel _testModel;

    public InferenceTests()
    {
        // Initialize TorchSharp for the CPU backend
        torch.InitializeDeviceType(DeviceType.CPU);
        _testModel = new ContentSimilarityModel();
    }
    
    /// <summary>
    /// Tests the Predict method with a valid input tensor.
    /// </summary>
    [Fact]
    public void Predict_ValidInput_ShouldReturnValidOutput()
    {
        // Arrange
        var inputTensor = torch.tensor([1.0f, 2.0f, 3.0f]).unsqueeze(0);
        var inference = new Inference(_testModel);

        // Act
        var output = inference.Predict(inputTensor);

        // Assert
        Assert.Equal(6.0f, output); // Simulated behavior: sum of inputs (see TestContentSimilarityModel)
    }
}