var builder = DistributedApplication.CreateBuilder(args);

builder.AddProject<Projects.ReadifyAI_Api>("ai-service")
    .WithExternalHttpEndpoints();
    
builder.Build().Run();