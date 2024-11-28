using EpubSharp;
using ReadifyAI.Core.Interfaces;
using PdfSharp.Pdf;

namespace Readify.Infrastructure.Converters;

public class EpubPdfService : IConverter {
    public Task ConvertAsync(string epubPath, string pdfPath) {
        var epubBook = EpubReader.Read(epubPath);
        var pdfDocument = new PdfDocument {
            Info = { Title = epubBook.Title }
        };

        object pdfLock = new object();
        Parallel.ForEach(epubBook.TableOfContents, chapter => { });
        return Task.CompletedTask;
    }
}