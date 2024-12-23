using Microsoft.Extensions.DependencyInjection;

var builder = DistributedApplication.CreateBuilder(args);
builder.AddProject<Projects.ReadifyAI_Api>("ai-service")
    .WithExternalHttpEndpoints()
    .ApplicationBuilder
    .Services
    .AddHealthChecks();

builder.Build().Run();