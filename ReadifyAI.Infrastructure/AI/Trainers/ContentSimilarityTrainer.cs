using Readify.Infrastructure.AI.Models;
using static TorchSharp.torch;

namespace Readify.Infrastructure.AI.Trainers;

public class ContentSimilarityTrainer {

    public void TrainModel(ContentSimilarityModel model, Tensor input, Tensor target, int epochs = 10) {
        var optimizer = optim.Adam(model.parameters(), lr: 0.001);
        var lossFunction = nn.functional.binary_cross_entropy_with_logits;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            model.train();
            optimizer.zero_grad();

            // Forward pass
            var output = model.Forward(input);
            var targetReshaped = target.view(output.shape);
            if (targetReshaped.is_sparse) {
                targetReshaped = targetReshaped.to_dense();
            }

            // Compute loss
            var loss = lossFunction(output, targetReshaped);

            // Backward pass and optimize
            loss.backward();
            optimizer.step();

            Console.WriteLine($"Epoch {epoch}, Loss: {loss.item<float>()}");
        }
    }
}