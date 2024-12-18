using Readify.Infrastructure.AI.Models;
using TorchSharp;

namespace Readify.Infrastructure.AI.Trainers;

public class ContentSimilarityTrainer {
    public void TrainModel(ContentSimilarityModel model, torch.Tensor input, torch.Tensor target, int epochs = 10) {
        var optimizer = torch.optim.Adam(model.parameters(), lr: 0.001);
        var lossFunction = torch.nn.functional.mse_loss;
        
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            model.train();
            optimizer.zero_grad();

            // Forward pass
            var output = model.Forward(input);
            var targetFloat = target.to(torch.float32);
            
            // Compute loss
            var loss = lossFunction(output, targetFloat);

            // Backward pass and optimize
            loss.backward();
            optimizer.step();
        }
    }
}