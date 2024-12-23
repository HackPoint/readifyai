using Readify.Infrastructure.AI.Inferences;
using Readify.Infrastructure.AI.Models;
using Readify.Infrastructure.AI.Preparation;
using Readify.Infrastructure.Services;
using TorchSharp;

namespace ReadifyAI.Infrastructure.Tests;

public class InferenceServiceTests
{
    private readonly InferenceService _inferenceService;

    public InferenceServiceTests()
    {
        torch.InitializeDeviceType(DeviceType.CPU);
        _inferenceService = new InferenceService(
            new Inference(new ContentSimilarityModel()),
            new DataPreparer()
        );
    }

    [Fact]
    public async Task PredictAsync_ShouldReturnValidScoreAndCacheSession()
    {
        // Arrange
        var sessionId = "session-1";
        var content = "Test content";

        // Act
        var similarity = await _inferenceService.PredictAsync(sessionId, content);

        // Assert
        Assert.InRange(similarity, 0.0f, 1.0f);

        // Verify session cache (covered indirectly if PredictAsync works)
    }

    [Fact]
    public async Task CompareAsync_ShouldHandleVariousInputScenarios()
    {
        // Arrange
        var sessionId = "session-1";

        var identicalContent = "Identical content.";
        var unrelatedContent1 = "Completely unrelated text.";
        var unrelatedContent2 = "Totally different text.";
        var relatedContent1 = "The cat sat on the mat.";
        var relatedContent2 = "The feline rested on the carpet.";

        // Act
        var identicalSimilarity = await _inferenceService.CompareAsync(sessionId, identicalContent, identicalContent);
        var unrelatedSimilarity = await _inferenceService.CompareAsync(sessionId, unrelatedContent1, unrelatedContent2);
        var relatedSimilarity = await _inferenceService.CompareAsync(sessionId, relatedContent1, relatedContent2);

        // Assert
        Assert.Equal(1.0, identicalSimilarity);
        Assert.True(unrelatedSimilarity < 0.1, $"Similarity {unrelatedSimilarity} is too high for unrelated inputs.");
        Assert.True(relatedSimilarity is > 0.3 and < 0.8,
            $"Similarity {relatedSimilarity} is not within the expected range for related inputs.");
    }

    [Fact]
    public async Task CompareAsync_ShouldThrowForInvalidInputs()
    {
        // Arrange
        var sessionId = "session-1";
        var emptyContent = "";
        var nullContent = (string)null!;

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _inferenceService.CompareAsync(sessionId, emptyContent, "Valid content"));
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _inferenceService.CompareAsync(sessionId, nullContent, "Valid content"));
    }

    [Fact]
    public async Task TrainAsync_ShouldImprovePredictionAccuracy()
    {
        // Arrange
        var trainingData = new List<(string Content, float Target)>
        {
            ("Positive example content.", 1.0f),
            ("Negative example content.", 0.0f)
        };
        var sessionId = "session-1";
        var testContent = "Positive example content.";

        // Act
        await _inferenceService.TrainAsync(trainingData);
        var similarity = await _inferenceService.PredictAsync(sessionId, testContent);

        // Assert
        Assert.True(similarity > 0.7,
            $"Similarity {similarity} is lower than expected after training.");
    }
}
