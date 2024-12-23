using TorchSharp;

namespace Readify.Infrastructure.Extensions;

public static class TorchExtensions {
    public static torch.Tensor Normalize(this torch.Tensor input, int dim, float p = 2f, double epsilon = 1e-9) {
        // Calculate the norm along the specified dimension
        var normValue = torch.norm(input, dim, keepdim: true, p);

        // Avoid division by zero
        normValue = torch.clamp(normValue, min: epsilon);

        // Normalize the tensor
        return input / normValue;
    }
}