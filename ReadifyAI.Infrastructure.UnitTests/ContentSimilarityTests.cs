using Readify.Infrastructure.AI.Inferences;
using Readify.Infrastructure.AI.Models;
using TorchSharp;

namespace ReadifyAI.Infrastructure.Tests;

public class ContentSimilarityTests {
    [Fact]
    public void Predict_ShouldReturnScoreBetweenZeroAndOne()
    {
        // Arrange
        var model = new ContentSimilarityModel();
        var inference = new Inference(model);
        var input = torch.tensor(new long[] { 1, 2, 3 }, torch.ScalarType.Int64).unsqueeze(0);

        // Act
        var score = inference.Predict(input);

        // Assert
        Assert.InRange(score, 0.0, 1.0);
    }
    
    [Fact]
    public void Compare_ShouldReturnHighSimilarityForIdenticalInputs()
    {
        // Arrange
        var model = new ContentSimilarityModel();
        var inference = new Inference(model);
        var input1 = torch.tensor(new long[] { 1, 2, 3 }, torch.ScalarType.Int64).unsqueeze(0);
        var input2 = torch.tensor(new long[] { 1, 2, 3 }, torch.ScalarType.Int64).unsqueeze(0);

        // Act
        var similarity = inference.Compare(input1, input2);

        // Assert
        Assert.True(similarity > 0.9);
    }
}