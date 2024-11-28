namespace ReadifyAI.Core.Interfaces;

public interface IAiComparer {
    Task<double> CompareContentAsync(string input, string output);
}