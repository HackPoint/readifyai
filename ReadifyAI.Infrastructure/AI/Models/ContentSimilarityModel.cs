using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Readify.Infrastructure.AI.Models;

public class ContentSimilarityModel : nn.Module {
    private readonly Embedding _embeddingLayer;
    private readonly Linear _similarityLayer;
    private readonly int _vocabularySize;

    public ContentSimilarityModel() : base("ContentSimilarityModel") {
        _vocabularySize = 10000;
        _embeddingLayer = nn.Embedding(_vocabularySize, 256);
        _similarityLayer = nn.Linear(256, 1);

        RegisterComponents();
    }

    public Tensor Forward(Tensor input) {
        var clippedInput = input.clamp(0, _vocabularySize - 1);
        var embedded = _embeddingLayer.forward(clippedInput);
        var meanEmbedding = embedded.mean([1]);
        var output = _similarityLayer.forward(meanEmbedding);
        
        return nn.functional.sigmoid(output);
    }
}