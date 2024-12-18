using Readify.Infrastructure.AI.Models;
using TorchSharp;

namespace Readify.Infrastructure.AI.Inferences;

public class Inference {
    private readonly ContentSimilarityModel _model;

    public Inference(ContentSimilarityModel model) {
        _model = model;
    }

    public float Predict(torch.Tensor input) {
        _model.eval();

        using (torch.no_grad()) {
            // Ensure the input tensor has the correct type
            if (input.dtype != torch.ScalarType.Int64)
            {
                input = input.to(torch.ScalarType.Int64);
            }

            var output = _model.Forward(input);
            return output.item<float>();
        }
    }

    /// <summary>
    /// Compares two inputs using the similarity model.
    /// </summary>
    public double Compare(torch.Tensor input1, torch.Tensor input2) {
        _model.eval();

        using (torch.no_grad()) {
            var embedding1 = _model.Forward(input1);
            var embedding2 = _model.Forward(input2);

            // Calculate cosine similarity
            var similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim: 0);
            return similarity.item<float>();
        }
    }
}