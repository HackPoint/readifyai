namespace ReadifyAI.Core.Interfaces;

public interface IConverter {
    Task ConvertAsync(string epubPath, string pdfPath);
}