using Readify.Infrastructure.AI.Models;
using TorchSharp;

namespace Readify.Infrastructure.AI.Inferences;

public class Inference(ContentSimilarityModel model) {
    public float Predict(torch.Tensor input) {
        model.eval();

        using (torch.no_grad()) {
            // Ensure the input tensor has the correct type
            if (input.dtype != torch.ScalarType.Int64) {
                input = input.to(torch.ScalarType.Int64);
            }

            if (input.is_sparse) {
                input = input.to_dense();
            }

            var output = model.Forward(input);
            return output.clamp(0.0f, 1.0f).item<float>();
        }
    }

    /// <summary>
    /// Compares two inputs using the similarity model.
    /// </summary>
    public double Compare(torch.Tensor input1, torch.Tensor input2) {
        model.eval();

        using (torch.no_grad()) {
            // Process inputs through the model
            var embedding1 = model.Forward(input1);
            var embedding2 = model.Forward(input2);

            // Debugging: Print tensor shapes
            Console.WriteLine($"Embedding1 Shape: {string.Join(", ", embedding1.shape)}");
            Console.WriteLine($"Embedding2 Shape: {string.Join(", ", embedding2.shape)}");

            // Validate dimensions
            if (embedding1.dim() != 2 || embedding2.dim() != 2) {
                throw new InvalidOperationException($"Expected 2D tensor but got {embedding1.dim()}D tensor.");
            }

            if (!embedding1.shape.SequenceEqual(embedding2.shape)) {
                throw new InvalidOperationException("Embeddings must have the same shape.");
            }

            // Calculate cosine similarity
            var dotProduct = torch.sum(embedding1 * embedding2, dim: -1); // Element-wise dot product
            var magnitude1 = embedding1.norm(dim: 1, keepdim: true);
            var magnitude2 = embedding2.norm(dim: 1, keepdim: true);
            var similarity = (dotProduct / (magnitude1 * magnitude2));
            return similarity.mean().item<float>();
        }
    }
}