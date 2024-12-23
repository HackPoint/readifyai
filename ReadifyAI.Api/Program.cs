using System.Text.Json.Serialization;
using Readify.Infrastructure.AI.Inferences;
using Readify.Infrastructure.AI.Models;
using Readify.Infrastructure.AI.Preparation;
using Readify.Infrastructure.Services;
using ReadifyAI.Application.Interfaces;
namespace ReadifyAI.Api;
using Microsoft.Extensions.Hosting;

public class Program {
    public static void Main(string[] args) {
        var builder = WebApplication.CreateSlimBuilder(args);
        builder.AddServiceDefaults();
        
        builder.Services.ConfigureHttpJsonOptions(options =>
        {
            options.SerializerOptions.TypeInfoResolver = AppJsonSerializerContext.Default;
        });
        
        builder.Services.AddSingleton<ContentSimilarityModel>();
        builder.Services.AddScoped<IInferenceService, InferenceService>();
        builder.Services.AddScoped<IModelDataPreparer, DataPreparer>();
        builder.Services.AddScoped<Inference>(sp => {
            var model = sp.GetRequiredService<ContentSimilarityModel>();
            return new Inference(model);
        });
        
        var app = builder.Build();
        app.MapGet("/", () => "API is running!");
        
        var aiApi = app.MapGroup("/api/ai");
        aiApi.MapPost("/predict", async (PredictionRequest request, IInferenceService service) => {
            var result = await service.PredictAsync(request.SessionId, request.Content);
            return Results.Ok(result);
        });

        aiApi.MapPost("/compare", async (ComparisonRequest request, IInferenceService service) => {
            var result = await service.CompareAsync(request.SessionId, request.Content1, request.Content2);
            return Results.Ok(result);
        });

        aiApi.MapPost("/train", async (IEnumerable<TrainingRequest> trainingData, IInferenceService service) => {
            await service.TrainAsync(trainingData.Select(td => (td.Content, td.Target)));
            return Results.Ok("Training complete.");
        });
        app.Run();
    }
}

[JsonSerializable(typeof(PredictionRequest))]
[JsonSerializable(typeof(PredictionResponse))]
[JsonSerializable(typeof(ComparisonRequest))]
[JsonSerializable(typeof(bool))]
[JsonSerializable(typeof(int))]
[JsonSerializable(typeof(double))]
internal partial class AppJsonSerializerContext : JsonSerializerContext {
}