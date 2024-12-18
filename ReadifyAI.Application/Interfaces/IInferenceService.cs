using System.Text.Json.Serialization;

namespace ReadifyAI.Application.Interfaces;

public interface IInferenceService {
    Task<float> PredictAsync(string sessionId, string content);
    Task<double> CompareAsync(string sessionId, string content1, string content2);
    Task TrainAsync(IEnumerable<(string Content, float Target)> trainingData);
}

public record TrainingRequest(string Content, float Target);
public record PredictionRequest(string SessionId, string Content);
public record ComparisonRequest(
    [property: JsonPropertyName("sessionId")] string SessionId,
    [property: JsonPropertyName("content1")] string Content1,
    [property: JsonPropertyName("content2")] string Content2);
public record PredictionResponse(float Similarity);