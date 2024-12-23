using Readify.Infrastructure.AI.Models;
using Readify.Infrastructure.Extensions;
using TorchSharp;

namespace ReadifyAI.Infrastructure.Tests;

public class TorchExtensionsTest {
    [Fact]
    public void Normalize_ValidInput_ReturnsNormalizedTensor() {
        // Arrange
        var input = torch.tensor(new float[,] { { 3, 4 }, { 6, 8 } });

        // Act
        var normalized = input.Normalize(dim: 1);

        // Assert
        for (int i = 0; i < input.shape[0]; i++) // Iterate over rows
        {
            var row = normalized[i];
            var normValue = torch.norm(row, p: 2).item<float>(); // Calculate L2 norm of the row
            Assert.Equal(1.0f, normValue, precision: 5); // Each row should be normalized to 1
        }
    }
}