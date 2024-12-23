using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Readify.Infrastructure.AI.Models;

public sealed class ContentSimilarityModel : nn.Module {
    private readonly Embedding _embeddingLayer;
    private readonly Linear _similarityLayer;
    private readonly int _vocabularySize;

    public ContentSimilarityModel() : base("ContentSimilarityModel") {
        _vocabularySize = 10000;
        _embeddingLayer = nn.Embedding(_vocabularySize, 256);
        _similarityLayer = nn.Linear(256, 1);

        RegisterComponents();
        InitializeParameters();
    }

    private void InitializeParameters() {
        foreach (var param in parameters()) {
            if (param.requires_grad) {
                if (param.ndim >= 2) {
                    // Xavier initialization for tensors with at least 2 dimensions
                    nn.init.xavier_uniform_(param);
                }
                else {
                    // Use a small constant initialization for biases or 1D tensors
                    nn.init.constant_(param, 0.01);
                }
            }
        }
    }

    public Tensor Forward(Tensor input) {
        // var clippedInput = input.clamp(0, _vocabularySize - 1);
        // var embedded = _embeddingLayer.forward(clippedInput);
        // var meanEmbedding = embedded.mean([1]);
        // var output = _similarityLayer.forward(meanEmbedding);
        //
        // return nn.functional.sigmoid(output);

        var numEmbeddings = _embeddingLayer.weight!.shape[0];
        var clippedInput = input.clamp(0, numEmbeddings - 1);
        var embedded = _embeddingLayer.forward(clippedInput);
        var meanEmbedding = embedded.mean([1], keepdim: false);
        return _similarityLayer.forward(meanEmbedding);
    }
}