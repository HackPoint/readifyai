using System.Collections.Concurrent;
using Readify.Infrastructure.AI.Inferences;
using Readify.Infrastructure.AI.Models;
using Readify.Infrastructure.AI.Preparation;
using Readify.Infrastructure.AI.Trainers;
using ReadifyAI.Application.Interfaces;
using TorchSharp;

namespace Readify.Infrastructure.Services;

public class InferenceService(Inference inference, IModelDataPreparer modelDataPreparer) : IInferenceService {
    private readonly ContentSimilarityTrainer _trainer = new();
    private readonly ConcurrentDictionary<string, torch.Tensor> _sessionCache = new();

    public Task<float> PredictAsync(string sessionId, string content) {
        var inputTensor = modelDataPreparer.TokenizeText(content);
        if (!_sessionCache.TryAdd(sessionId, inputTensor)) {
            _sessionCache[sessionId] = inputTensor;
        }

        return Task.FromResult(inference.Predict(inputTensor));
    }

    public Task<double> CompareAsync(string sessionId, string content1, string content2) {
        if(string.IsNullOrWhiteSpace(content1) || string.IsNullOrWhiteSpace(content2))
            throw new InvalidOperationException("Unable to Compare content with empty string.");
        
        var tensorContent1 = modelDataPreparer.TokenizeText(content1);
        var tensorContent2 = modelDataPreparer.TokenizeText(content2);
        var compare = inference.Compare(tensorContent1, tensorContent2);
        
        return Task.FromResult(compare);
    }

    public Task TrainAsync(IEnumerable<(string Content, float Target)> trainingData) {
        foreach ((string content, float target) in trainingData) {
            var inputTensor = modelDataPreparer.TokenizeText(content);
            var targetTensor = torch.tensor([target], torch.ScalarType.Float32);
            var model = new ContentSimilarityModel();
            _trainer.TrainModel(model, inputTensor, targetTensor);
        }

        return Task.CompletedTask;
    }
}